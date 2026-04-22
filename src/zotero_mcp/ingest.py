"""
M3 batch ingest — drives the full Zotero → KG pipeline.

Flow per paper:
  1. Skip if already in SQLite (resume-safe) or tagged ``kg:skip`` / ``kg:duplicate_of:*``
  2. Fetch parent item + its attachments via Zotero API
  3. Resolve PDF bytes:
       a. Zotero attachment with contentType=application/pdf → download via /items/<key>/file
          (Zotero cloud) **only as fallback** — user's real storage is WebDAV
       b. Preferred: Jianguoyun ``/dav/zotero/<attachmentKey>.zip`` (per hard rule
          in feedback_zotero_pdf_webdav.md) — unzip, take the single PDF
  4. MinerU parse → markdown
  5. Kimi extract → ExtractedPaper JSON
  6. Write: SQLite (raw md + JSON) + Qdrant (dense + BM25 chunks) + Neo4j (nodes + edges)

Designed to run as ``python -m zotero_mcp.ingest --limit 10 [--dry-run]`` for
smoke tests, or ``python -m zotero_mcp.ingest`` for full library ingest.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load ~/.claude/.env.local so `nohup python -m zotero_mcp.ingest ...` works
# from any shell (no need to pre-`source` the env). Without this, a cold-shell
# launch crashes on KeyError('NEO4J_ZOTERO_PASSWORD') (or silently empty
# KIMI_API_KEY / Z_AI_API_KEY, which is worse).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env.local"), override=False)
except ImportError:
    pass

from .extractor import extract_structured
from .kg_store import IMG_REF_RE, make_writers
from .parser_factory import convert_to_markdown_smart_with_images

logger = logging.getLogger("zotero_mcp.ingest")

# ======================================================================
# Figure handling helpers (T0)
# ======================================================================
# MinerU writes md refs like ``![](images/<64-hex>.jpg)`` where the 64-hex
# is its internal content-addressed filename. We reuse filename+extension
# as ``mineru_name`` (same form as appears in md refs + WebDAV path). The
# ref regex itself is shared from ``kg_store.IMG_REF_RE`` — don't
# duplicate. Caption heuristics below match the 2 common MinerU layouts:
# ``![]() ⏎ Figure N:`` or ``Figure N: ⏎ ![]()``.
_CAPTION_AFTER_RE = re.compile(
    r'!\[[^\]]*\]\(images/(?P<mn>[a-f0-9]{64}\.\w{2,4})\)\s*\n+\s*'
    r'(?P<cap>(?:Figure|Fig\.?)\s*\d+[:.][^\n]{1,400})',
    re.IGNORECASE,
)
_CAPTION_BEFORE_RE = re.compile(
    r'(?P<cap>(?:Figure|Fig\.?)\s*\d+[:.][^\n]{1,400})\s*\n+\s*'
    r'!\[[^\]]*\]\(images/(?P<mn>[a-f0-9]{64}\.\w{2,4})\)',
    re.IGNORECASE,
)
FIG_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB → trigger compression per spec §4
WEBDAV_FIG_CB_LIMIT = int(os.environ.get("INGEST_FIG_CB_LIMIT", "3"))


# ======================================================================
# Stable identifier helpers (T1)
# ======================================================================
# DOI RFC 3986 path-like; accept URL + prefix variants + any case. Neo4j
# comparisons are case-sensitive, so normalize to lowercase on write.
_DOI_URL_PREFIXES = (
    "https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
    "http://dx.doi.org/", "doi:", "DOI:",
)
_ARXIV_RE = re.compile(
    r"\barxiv:\s*(?P<id>\d{4}\.\d{4,5})(?P<v>v\d+)?\b",
    re.IGNORECASE,
)
_ARXIV_ALIAS_DOI_RE = re.compile(
    r"10\.48550/arxiv\.(?P<id>\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
# arxiv_id alone is allowed in extra field too — e.g. when user manually
# typed just the id without "arXiv:" prefix
_ARXIV_BARE_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})$")


def normalize_doi(raw) -> str | None:
    """Return a lowercased canonical DOI (``10.<prefix>/<suffix>``) or None."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    for pref in _DOI_URL_PREFIXES:
        if s.lower().startswith(pref.lower()):
            s = s[len(pref):]
            break
    s = s.strip().lower()
    if not s.startswith("10.") or "/" not in s:
        return None
    return s


def normalize_arxiv(raw) -> str | None:
    """Return bare ``YYMM.NNNNN`` arXiv id (no version suffix) or None.

    Accepts ``arXiv:2501.18799 [eess]`` / ``arXiv:2501.18799v2`` /
    ``10.48550/arXiv.2501.18799`` / plain ``2501.18799``.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    m = _ARXIV_ALIAS_DOI_RE.search(s)
    if m:
        return m.group("id")
    m = _ARXIV_RE.search(s)
    if m:
        return m.group("id")
    m = _ARXIV_BARE_RE.match(s)
    if m:
        return m.group("id")
    return None


def _parse_paper_ids(item: dict) -> dict:
    """Extract ``{doi, arxiv_id}`` from a Zotero item.data dict.

    Both values are optional; missing ones are ``None``. When the Zotero
    ``DOI`` field itself is a ``10.48550/arXiv.*`` alias, the bare arxiv id
    is ALSO populated — downstream SAME_WORK_AS scans will match preprint
    and published versions regardless of which attribute Zotero recorded.
    """
    doi = normalize_doi(item.get("DOI", ""))
    # arxiv from extra is common; fall back to scanning DOI if it's an alias
    arxiv = normalize_arxiv(item.get("extra", ""))
    if not arxiv and doi:
        # doi may be an arxiv alias → mirror into arxiv_id
        m = _ARXIV_ALIAS_DOI_RE.search(doi)
        if m:
            arxiv = m.group("id")
    return {"doi": doi, "arxiv_id": arxiv}


# ======================================================================
# Non-paper URL fetch helper (T3)
# ======================================================================
# Zotero ``webpage`` / ``blogPost`` / ``computerProgram`` / etc. items
# carry ``item.data.url``. We fetch the URL with a browser User-Agent
# (markitdown's default UA gets 403'd on Wikipedia / StackExchange) and
# parse to markdown via markitdown's stream mode. Single timeout-capped
# path so a stuck fetch can't hang the whole ingest batch.
#
# Zhihu and other SPA / cookie-walled sites fail → caller falls back to
# ``_metadata_only_markdown``. v1 accepts the ~<10% metadata-only share;
# Playwright / Tavily escalation is deferred to T5.
T3_NON_PAPER_TYPES = frozenset({
    "webpage", "blogPost", "computerProgram",
    "encyclopediaArticle", "forumPost", "software",
})
T3_UA_BROWSER = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
T3_FETCH_TIMEOUT = int(os.environ.get("T3_FETCH_TIMEOUT", "30"))
T3_MD_ACCEPT_LEN = int(os.environ.get("T3_MD_ACCEPT_LEN", "500"))

# T3 Tier 1.5: cookie-aware Playwright fetch for login-walled domains.
# Maps {hostname suffix → path to Cookie-Editor JSON export}. Only domains
# listed here go through the headless-browser + cookies path; all others
# stay on the plain urllib tier. This keeps browser launch overhead off the
# hot path for the 90%+ of URLs that don't need it.
T3_AUTH_COOKIE_MAP_DEFAULT = {
    "zhihu.com": "~/.config/zotero-kg/zhihu_cookies.json",
}


def _parse_cookie_map_env() -> dict[str, str]:
    """Optional env override for T3_AUTH_COOKIES:
        T3_AUTH_COOKIES="zhihu.com:~/.config/zotero-kg/zhihu.json;weibo.com:/etc/..."
    """
    raw = os.environ.get("T3_AUTH_COOKIES", "").strip()
    if not raw:
        return dict(T3_AUTH_COOKIE_MAP_DEFAULT)
    out: dict[str, str] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        host, path = entry.split(":", 1)
        out[host.strip()] = path.strip()
    return out


def _auth_cookies_for_url(url: str) -> str | None:
    """Return cookie-JSON path if URL's host matches an auth-cookie entry."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    for suffix, path in _parse_cookie_map_env().items():
        if host == suffix or host.endswith("." + suffix):
            expanded = os.path.expanduser(path)
            if os.path.isfile(expanded):
                return expanded
    return None


def _convert_cookies_chrome_to_playwright(raw: list[dict]) -> list[dict]:
    """Translate Cookie-Editor JSON (Chrome extension format) into the
    shape ``BrowserContext.add_cookies`` expects."""
    out = []
    ss_map = {"no_restriction": "None", "unspecified": "Lax",
              "lax": "Lax", "strict": "Strict"}
    for c in raw:
        pc = {
            "name": c["name"], "value": c["value"],
            "domain": c["domain"], "path": c.get("path", "/"),
            "httpOnly": c.get("httpOnly", False),
            "secure": c.get("secure", False),
            "sameSite": ss_map.get(c.get("sameSite", "unspecified"), "Lax"),
        }
        pc["expires"] = (c["expirationDate"]
                         if not c.get("session") and "expirationDate" in c
                         else -1)
        out.append(pc)
    return out


def fetch_markdown_via_playwright_cookies(url: str, cookie_json_path: str) -> str:
    """T3 Tier 1.5: headless Chrome with user-supplied cookies → markdown.

    Only invoked when ``_auth_cookies_for_url(url)`` returns a path (i.e.
    host matches a known cookie-walled site like zhihu.com). Cookies are
    exported by the user via the Cookie-Editor Chrome extension from a
    browser where they are actively logged in; stored at a 600-perm file
    under ~/.config/zotero-kg/.

    Blocking constraint: the cookie file may expire (Zhihu z_c0 is ~6-12
    months; auto-renewed on user activity). On auth failure (login wall
    in body text) we return "" so caller falls through to
    snapshot / metadata-only fallback.

    Page-load strategy: ``wait_until='domcontentloaded'`` + 3s settle.
    networkidle hangs because Zhihu's ad/tracker beacons never stop.
    """
    import json as _json
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("T3 Playwright not available, skipping cookie fetch")
        return ""
    try:
        cookies_raw = _json.loads(Path(cookie_json_path).read_text())
    except Exception as e:
        logger.warning("T3 cookie file unreadable (%s): %s", cookie_json_path, e)
        return ""
    cookies = _convert_cookies_chrome_to_playwright(cookies_raw)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                "--no-sandbox", "--disable-blink-features=AutomationControlled",
            ])
            ctx = browser.new_context(
                user_agent=T3_UA_BROWSER,
                locale="zh-CN",
                viewport={"width": 1280, "height": 800},
            )
            ctx.add_cookies(cookies)
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(3000)   # let React render article body
                body_text = page.evaluate("() => document.body.innerText")
            finally:
                browser.close()
    except Exception as e:
        logger.warning("T3 Playwright fetch fail %s: %s", url, e)
        return ""

    # Auth-expired detection: common "please log in" markers in first 500 chars
    head = (body_text or "")[:500]
    if any(m in head for m in ("请您登录", "立即登录", "请登录后")):
        logger.warning("T3 Playwright %s hit login wall — cookies may be expired", url)
        return ""
    return body_text or ""


def fetch_snapshot_markdown(item_key: str) -> str:
    """T3 Tier 2: load a Zotero child HTML snapshot attachment and parse it.

    Looks for a child attachment with ``contentType='text/html'`` (produced
    by Zotero Connector's "Save Page with Snapshot" feature). If present,
    downloads the archived HTML bytes via WebDAV and parses them with
    markitdown — same output shape as ``fetch_markdown_via_markitdown``.

    Preferred fallback order in ``ingest_one``:
        live URL fetch → snapshot attachment → metadata-only

    Returns "" if no snapshot exists or any step fails. Snapshot content may
    be stale (saved long ago) — that's the tradeoff for getting past
    login-walled sites (Zhihu etc.): user's logged-in session captured it,
    we just replay it.
    """
    try:
        children = _z_get_json(f"/items/{item_key}/children", {"format": "json"})
    except Exception as e:
        logger.debug("T3 snapshot: /children fetch fail %s: %s", item_key, e)
        return ""
    # Find the first HTML attachment (Zotero snapshots use contentType=text/html)
    att = next(
        (c for c in children
         if c["data"].get("itemType") == "attachment"
         and c["data"].get("contentType") == "text/html"),
        None,
    )
    if not att:
        return ""
    att_key = att["key"]
    html_bytes: bytes | None = None
    # Source A: WebDAV (our own zotero-mcp ingest stores md/html here).
    # Browser Zotero Connector snapshots may leave a 0-byte placeholder
    # since Connector uploads to Zotero Cloud, not WebDAV — we treat
    # missing or empty-zip WebDAV as "not there" and fall through to B.
    try:
        from zotero_mcp import webdav as _webdav
        if _webdav.webdav_enabled():
            html_bytes = _webdav.fetch_attachment_bytes(att_key)
    except Exception as e:
        logger.debug("T3 snapshot WebDAV fail %s (att %s): %s",
                     item_key, att_key, e)
    # Source B: Zotero Cloud (where browser Connector uploads by default)
    if not html_bytes:
        try:
            html_bytes = _zotero_cloud_fetch(att_key)
        except Exception as e:
            logger.debug("T3 snapshot Zotero Cloud fail %s (att %s): %s",
                         item_key, att_key, e)
    if not html_bytes:
        logger.debug("T3 snapshot: no bytes from WebDAV or Cloud for %s", item_key)
        return ""
    try:
        import io as _io
        from markitdown import MarkItDown, StreamInfo
        r = MarkItDown().convert(
            _io.BytesIO(html_bytes), stream_info=StreamInfo(extension=".html"))
        md = r.text_content or ""
        if md:
            logger.info("T3 snapshot loaded for %s: md len=%d", item_key, len(md))
        return md
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("T3 snapshot markitdown fail %s: %s", item_key, e)
        return ""
    except Exception as e:  # noqa: BLE001
        logger.warning("T3 snapshot unexpected %s: %s", type(e).__name__, e)
        return ""


def fetch_markdown_via_markitdown(url: str) -> str:
    """Fetch ``url`` with browser UA, parse HTML → markdown via markitdown.

    Returns markdown string (possibly empty on any failure; caller should
    fall back to metadata-only markdown). Single-path with a 30s hard
    timeout cap — avoids markitdown's own ``requests`` layer hanging on
    slow / SPA endpoints.
    """
    # Normalize: treat whitespace-only / None as empty (Finding #3). Avoids
    # urllib.request.Request raising ValueError on malformed URLs, which
    # isn't caught by the HTTPError/URLError/TimeoutError/OSError tuple
    # below and would otherwise bubble up and crash the per-item ingest.
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    try:
        import io as _io
        from markitdown import MarkItDown, StreamInfo
    except ImportError as e:
        logger.warning("markitdown missing: %s — T3 URL fetch disabled", e)
        return ""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": T3_UA_BROWSER})
    except ValueError as e:
        logger.warning("T3 fetch bad url %r: %s", url, e)
        return ""
    try:
        with urllib.request.urlopen(req, timeout=T3_FETCH_TIMEOUT) as r:
            body = r.read()
            ctype = r.headers.get("Content-Type", "")
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, OSError) as e:
        logger.warning("T3 fetch fail %s: %s", url, e)
        return ""
    ext = ".html" if "html" in ctype else ""
    try:
        r = MarkItDown().convert(
            _io.BytesIO(body), stream_info=StreamInfo(extension=ext))
        return r.text_content or ""
    # Finding #2: markitdown's thrown types vary (ValueError for bad HTML,
    # RuntimeError for converter issues, sometimes its own MarkItDownException).
    # Narrow enough to surface real bugs in test runs, catch enough for real
    # URLs with weird encodings.
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("T3 markitdown parse fail %s (%s): %s",
                       url, type(e).__name__, e)
        return ""
    except Exception as e:   # noqa: BLE001 — last-resort to keep batch alive
        logger.warning("T3 markitdown unexpected %s on %s: %s",
                       type(e).__name__, url, e)
        return ""


def _build_caption_map(md: str) -> dict[str, str]:
    """Return ``{mineru_name → caption}`` for all image refs in ``md``.

    AFTER (``![]() ⏎ Figure N:``) has priority over BEFORE (``Figure N: ⏎
    ![]()``): this prevents cross-contamination when md alternates
    ``img · caption · img`` — the caption belongs to the preceding image
    (AFTER's hit) and shouldn't also be assigned to the next image via
    BEFORE's pattern. Unmapped images simply don't appear in the result.
    """
    result: dict[str, str] = {}
    claimed_spans: set[tuple[int, int]] = set()
    for m in _CAPTION_AFTER_RE.finditer(md):
        mn = m.group("mn")
        if mn not in result:
            result[mn] = m.group("cap").strip()
            claimed_spans.add(m.span("cap"))
    for m in _CAPTION_BEFORE_RE.finditer(md):
        mn = m.group("mn")
        if mn in result:
            continue
        if m.span("cap") in claimed_spans:
            continue  # this caption already bound to an earlier image
        result[mn] = m.group("cap").strip()
    return result


def _caption_for(mn: str, md: str) -> str | None:
    """Single-name wrapper around ``_build_caption_map`` — used by tests and
    direct callers. For bulk builds inside ingest_one, prefer
    ``_build_caption_map`` to avoid O(N) re-scan of md."""
    return _build_caption_map(md).get(mn)


def _maybe_downscale(img_bytes: bytes) -> tuple[bytes, bool, int, int]:
    """Return ``(bytes_out, downscaled_flag, width, height)``.

    - If raw ≤ 10 MB: keep bytes as-is. Read only dims via PIL.
    - If raw > 10 MB: PIL thumbnail → longest_edge 2048 → JPEG q=85. If
      still > 10 MB: q=70. If STILL > 10 MB: return the q=70 bytes anyway
      and let caller decide (WebDAV PUT may reject, but SQLite row still
      gets created with webdav_path=None — figure still has metadata).
    """
    from PIL import Image
    import io as _io
    if len(img_bytes) <= FIG_SIZE_LIMIT:
        im = Image.open(_io.BytesIO(img_bytes))
        w, h = im.size
        return img_bytes, False, w, h
    im = Image.open(_io.BytesIO(img_bytes))
    im.thumbnail((2048, 2048))
    im_rgb = im.convert("RGB") if im.mode != "RGB" else im
    for quality in (85, 70):
        buf = _io.BytesIO()
        im_rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= FIG_SIZE_LIMIT:
            return out, True, im_rgb.size[0], im_rgb.size[1]
    # Still oversize — return the q=70 version; caller may skip WebDAV.
    return out, True, im_rgb.size[0], im_rgb.size[1]


def _build_figures(md: str, images_raw: dict[str, bytes]) -> list[dict]:
    """Turn MinerU's ``{filename.jpg: raw_bytes}`` dict into SQLite-ready
    rows + the bytes to upload.

    Each row has:
        mineru_name   (filename with extension, e.g. ``<64hex>.jpg``)
        content_sha   (sha256 of the BYTES WE'LL UPLOAD, post-downscale)
        caption       (parsed from md if adjacent, else None)
        bytes_len, width, height, mime
        downscaled    (bool)
        bytes_for_upload  — not persisted in SQLite, consumed by Stage A
    """
    import hashlib as _h
    caption_map = _build_caption_map(md)
    out: list[dict] = []
    for name, raw in images_raw.items():
        try:
            final_bytes, was_down, w, h = _maybe_downscale(raw)
        except Exception as e:
            # PIL unsupported format / corrupt — skip image, keep paper going
            logger.warning("figure %s decode/downscale err: %s", name, e)
            continue
        out.append({
            "mineru_name": name,
            "content_sha": _h.sha256(final_bytes).hexdigest(),
            "caption":     caption_map.get(name),
            "bytes_len":   len(final_bytes),
            "width":       w,
            "height":      h,
            "mime":        "image/jpeg",
            "downscaled":  was_down,
            "webdav_path": None,    # filled by Stage A
            "bytes_for_upload": final_bytes,
        })
    return out

# ======================================================================
# Zotero API helpers (minimal, just what ingest needs)
# ======================================================================
Z_HDR: dict[str, str] = {}
Z_BASE = ""


def _z_init():
    global Z_HDR, Z_BASE
    key = os.environ["ZOTERO_API_KEY"]
    lib = os.environ["ZOTERO_LIBRARY_ID"]
    Z_HDR = {"Zotero-API-Key": key, "Zotero-API-Version": "3"}
    Z_BASE = f"https://api.zotero.org/users/{lib}"


def _z_get_json(path: str, params: dict | None = None) -> list | dict:
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    req = urllib.request.Request(Z_BASE + path + q, headers=Z_HDR)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def list_ingest_candidates(
    *,
    item_types: set[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Yield top-level items eligible for ingest (not tagged skip/duplicate/orphan).

    T3 extends the default whitelist to cover non-academic sources
    (``webpage``, ``blogPost``, ``computerProgram``, ``encyclopediaArticle``,
    ``forumPost``) — these are routed to ``:Source:Webpage`` or
    ``:Source:CodeRepo`` in Neo4j (see ``kg_store.ITEM_TYPE_TO_LABELS``).
    """
    item_types = item_types or {
        # paper family
        "journalArticle", "preprint", "conferencePaper",
        "thesis", "bookSection", "book", "report",
        # T3 additions (Wiki §2.1 "webpage → markitdown fallback")
        "webpage", "blogPost", "computerProgram",
        "encyclopediaArticle", "forumPost",
    }
    out: list[dict] = []
    start = 0
    while True:
        batch = _z_get_json("/items/top", {"format": "json", "start": start, "limit": 100})
        if not batch:
            break
        for it in batch:
            d = it["data"]
            if d.get("itemType") not in item_types:
                continue
            tags = {t["tag"] for t in d.get("tags", [])}
            if "kg:skip" in tags or "kg:orphan" in tags:
                continue
            if any(t.startswith("kg:duplicate_of:") for t in tags):
                continue
            out.append(d)
            if limit and len(out) >= limit:
                return out
        start += len(batch)
    return out


def resolve_pdf_bytes(item: dict) -> bytes | None:
    """Return PDF bytes for an item, preferring WebDAV, falling back to Zotero cloud."""
    tags = {t["tag"] for t in item.get("tags", [])}
    # 1. kg:pdf_attached tag means we uploaded the PDF to WebDAV ourselves (M0.4)
    #    or Zotero has a child attachment already syncing via WebDAV.
    children = _z_get_json(f"/items/{item['key']}/children", {"format": "json"})
    pdf_attachments = [
        c for c in children
        if c["data"].get("itemType") == "attachment"
        and c["data"].get("contentType") == "application/pdf"
    ]
    # WebDAV preferred path: /dav/zotero/<attachmentKey>.zip
    if pdf_attachments:
        for att in pdf_attachments:
            b = _webdav_fetch(att["key"])
            if b:
                return b
    # fallback: Zotero cloud (/items/<key>/file) for items user hasn't WebDAV'd yet
    if pdf_attachments:
        return _zotero_cloud_fetch(pdf_attachments[0]["key"])
    return None


def _webdav_fetch(attachment_key: str) -> bytes | None:
    """Download <key>.zip from the configured WebDAV root and return the inner PDF.

    Delegates to ``zotero_mcp.webdav`` so WebDAV host + creds come from env
    (ZOTERO_WEBDAV_URL / ZOTERO_WEBDAV_USER / ZOTERO_WEBDAV_PASS), not hardcoded.
    """
    from zotero_mcp import webdav as _webdav
    if not _webdav.webdav_enabled():
        return None
    raw = _webdav.fetch_attachment_bytes(attachment_key)
    # ingest pipeline specifically wants the PDF bytes; WebDAV helper returns
    # the first file in the zip (Zotero's single-file convention), which is
    # the PDF here.
    return raw


def _zotero_cloud_fetch(attachment_key: str) -> bytes | None:
    req = urllib.request.Request(f"{Z_BASE}/items/{attachment_key}/file", headers=Z_HDR)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except Exception as e:
        logger.debug("Zotero cloud miss %s: %s", attachment_key, e)
        return None


# ======================================================================
# Main pipeline
# ======================================================================
def _metadata_only_markdown(item: dict) -> str:
    """Synthesize a minimal markdown doc from Zotero metadata + abstract.

    Used when no PDF is reachable. Better than dropping the item entirely —
    the abstract alone lets Kimi pull concepts, datasets, and (sometimes)
    methods; the citation graph still gets its :Source node.
    """
    parts = [f"# {item.get('title','')}", ""]
    creators = item.get("creators") or []
    if creators:
        authors = ", ".join(
            (c.get("lastName") or c.get("name","")).strip()
            for c in creators if c.get("creatorType") == "author"
        )
        parts += [f"**Authors:** {authors}", ""]
    date = item.get("date") or ""
    venue = item.get("publicationTitle") or item.get("conferenceName") or ""
    if date or venue:
        parts += [f"**Venue:** {venue}  **Year:** {date}", ""]
    doi = item.get("DOI") or ""
    if doi:
        parts += [f"**DOI:** {doi}", ""]
    abstract = item.get("abstractNote") or ""
    if abstract:
        parts += ["## Abstract", abstract, ""]
    extra = item.get("extra") or ""
    if extra:
        parts += ["## Notes", extra, ""]
    return "\n".join(parts)


def _qdrant_has_chunks(qd, paper_id: str) -> bool:
    """Cheap probe: is there ≥1 Qdrant point for this paper_id? Used to
    distinguish "already_done and fully wired" from "Stage C failed last
    time so retry Qdrant only". Does NOT count — we only need exists-or-not."""
    import urllib.request as _u, urllib.error as _ue, json as _j
    url = (f"{qd._qdrant_base}/collections/{qd.collection}/points/scroll")
    body = _j.dumps({
        "filter": {"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
        "limit": 1, "with_payload": False, "with_vector": False,
    }).encode()
    req = _u.Request(url, data=body, method="POST",
                     headers={"Content-Type": "application/json"})
    try:
        with _u.urlopen(req, timeout=15) as r:
            data = _j.loads(r.read())
    except (_ue.HTTPError, _ue.URLError, TimeoutError, OSError) as e:
        # Fail-open: treat as "has chunks" to avoid spurious retries when
        # Qdrant is flaky. Circuit breaker catches persistent outages.
        logger.warning("qdrant probe for %s failed (%s), assuming has_chunks",
                       paper_id, e)
        return True
    return bool(data.get("result", {}).get("points"))


def _stage_a_upload_figures(sql, figures: list[dict], paper_id: str) -> int:
    """Stage A: upload every figure to the independent WebDAV tree.

    Mutates each figure dict in place, setting ``webdav_path`` on success.
    Per-figure failures are logged + recorded in ``failures`` but don't stop
    the other figures. A streak of ``WEBDAV_FIG_CB_LIMIT`` consecutive
    outage-class errors trips a circuit breaker: the remaining figures get
    ``webdav_path=None`` and can be retried later by
    ``backfill_pending_figures``. Returns count of successfully uploaded.
    """
    from zotero_mcp import webdav as _wd
    uploaded = 0
    consec_out = 0
    tripped = False
    for f in figures:
        if tripped:
            # leave webdav_path=None; backfill will retry
            continue
        try:
            rel = _wd.put_figure(paper_id, f["mineru_name"], f["bytes_for_upload"])
            f["webdav_path"] = rel
            uploaded += 1
            consec_out = 0
        except _wd.WebDAVOutageError as e:
            sql.record_failure(paper_id, "webdav_figures",
                               f"{f['mineru_name']}: {e!r}"[:400])
            consec_out += 1
            if consec_out >= WEBDAV_FIG_CB_LIMIT:
                logger.error(
                    "WebDAV figures CB trip on %s after %d consecutive outages — "
                    "remaining %d figures deferred to backfill",
                    paper_id, consec_out,
                    sum(1 for g in figures if g.get("webdav_path") is None) - 1,
                )
                tripped = True
        except Exception as e:
            # non-outage (413, config, etc): log per-figure, keep going
            sql.record_failure(paper_id, "webdav_figures",
                               f"{f['mineru_name']}: {e!r}"[:400])
            logger.warning("webdav put_figure %s/%s failed: %s",
                           paper_id, f["mineru_name"], e)
            consec_out = 0
    return uploaded


def _backfill_pending_figures(
    sql, paper_id: str, tmp_pdf_path: Path, work_tmp: Path,
) -> dict:
    """Re-run only Stage A for figures that previously failed WebDAV upload.

    Re-fetches the PDF + re-runs MinerU to get image bytes (we don't persist
    bytes anywhere else), then uploads only the figures whose SQLite row
    has ``webdav_path IS NULL``. Marks them uploaded via
    ``mark_figure_uploaded``. Returns a stats dict.
    """
    pending = sql.pending_webdav_figures(paper_id)
    if not pending:
        return {"backfilled": 0, "remaining": 0}
    wanted = {row["mineru_name"] for row in pending}
    logger.info("backfill %s: %d figures pending", paper_id, len(wanted))
    # Re-parse to get bytes. This is OK — MinerU filenames are cross-parse
    # stable per spec §0.1 so we can address the same figure by mineru_name.
    _, images_raw = convert_to_markdown_smart_with_images(tmp_pdf_path)
    # Only upload the ones that are actually still pending.
    from zotero_mcp import webdav as _wd
    backfilled = 0
    for name in wanted:
        raw = images_raw.get(name)
        if raw is None:
            continue  # figure disappeared between parses — shouldn't happen
        bytes_out, _down, _w, _h = _maybe_downscale(raw)
        try:
            rel = _wd.put_figure(paper_id, name, bytes_out)
            sql.mark_figure_uploaded(paper_id, name, rel)
            backfilled += 1
        except Exception as e:
            logger.warning("backfill put_figure %s/%s failed: %s",
                           paper_id, name, e)
    return {"backfilled": backfilled, "remaining": len(wanted) - backfilled}


def ingest_one(item: dict, *, sql, qd, ne, work_tmp: Path) -> dict[str, str | float | int]:
    """Process a single item. Returns stats dict.

    T0 pipeline (see plans/zotero-kg-t0-spec.md):
      1. Resolve PDF bytes (WebDAV → Zotero cloud fallback)
      2. MinerU parse → (md, images_dict); images_dict empty for non-PDF
      3. Downscale >10MB images; compute content_sha + caption
      4. LLM extract with figures (VLM chain → text fallback)
      5. Stage A  WebDAV PUT each figure into /dav/zotero-kg-figures/<pid>/
      6. Stage B  SQLite atomic save (figures + papers in one txn)
      7. Stage C  Qdrant upsert (chunks carry figure_refs per payload)
      8. Neo4j write (independent, per-paper retry-able)
      9. Optional: md attachment upload (legacy Zotero sync)

    Fast-path skip: if a paper is "fully done" (SQLite papers row exists +
    Qdrant has chunks + no pending figures), return ``skip_done``. If only
    subset is done, run only the missing stages.
    """
    pid = item["key"]
    stats: dict[str, str | float | int] = {"pid": pid, "status": "ok"}
    t_start = time.time()

    # Fast-path: papers row exists. Investigate whether all downstream stores
    # have this paper; if anything's missing, we still have work to do below.
    if sql.already_done(pid):
        has_qdrant = _qdrant_has_chunks(qd, pid)
        pending_figs = sql.pending_webdav_figures(pid)
        if has_qdrant and not pending_figs:
            stats["status"] = "skip_done"
            return stats
        # Partial state — flag what's missing so downstream logic knows to
        # only backfill Stage A / Stage C rather than full re-parse.
        stats["partial"] = {
            "pending_figures": len(pending_figs),
            "qdrant_missing": not has_qdrant,
        }
        # Intentional fall-through: re-run the full pipeline, which is
        # idempotent at every stage (WebDAV PUT overwrite, SQLite
        # INSERT OR REPLACE + DELETE+INSERT figures, Qdrant upsert via
        # deterministic UUIDv5 point ids, Neo4j MERGE).

    pdf = resolve_pdf_bytes(item)
    mineru_secs = 0.0
    metadata_only = False
    md: str = ""
    images_raw: dict[str, bytes] = {}

    if pdf is None:
        # T3: for non-paper types, the fallback chain is:
        #   Tier 1 — live URL fetch via markitdown (fresh content)
        #   Tier 2 — Zotero Connector HTML snapshot (user-saved, possibly
        #            logged-in session → catches Zhihu / paywalled pages,
        #            but may be stale)
        #   Tier 3 — metadata-only markdown (title + abstractNote)
        #
        # Paper types skip Tiers 1+2 and go straight to metadata (their URLs
        # typically point at publisher paywalls that don't parse usefully).
        url = (item.get("url") or "").strip()
        item_type = item.get("itemType", "")
        is_t3 = item_type in T3_NON_PAPER_TYPES
        md_candidate = ""
        md_source = None   # "url_fetch" | "snapshot" | None

        if is_t3:
            # Tier 1: live URL fetch (plain urllib+UA → markitdown)
            if url:
                fetched = fetch_markdown_via_markitdown(url)
                fetched_len = len(fetched.strip())
                stats["fetched_url_len"] = fetched_len
                if fetched and fetched_len >= T3_MD_ACCEPT_LEN:
                    md_candidate = fetched
                    md_source = "url_fetch"
                else:
                    stats["fetch_status"] = ("too_short" if fetched else
                                              ("fail" if url else "no_url"))

            # Tier 1.5: cookie-aware Playwright for login-walled domains
            # (zhihu.com etc.). Only runs when Tier 1 didn't produce usable
            # content AND the URL's host has a configured auth-cookie file.
            if not md_candidate and url:
                cookie_path = _auth_cookies_for_url(url)
                if cookie_path:
                    pw_md = fetch_markdown_via_playwright_cookies(url, cookie_path)
                    pw_len = len(pw_md.strip())
                    stats["playwright_len"] = pw_len
                    if pw_md and pw_len >= T3_MD_ACCEPT_LEN:
                        md_candidate = pw_md
                        md_source = "playwright_cookies"

            # Tier 2: snapshot attachment (only when Tiers 1 + 1.5 didn't
            # produce usable content). Might be stale but at least gets past
            # 403s / login walls for sites where we DIDN'T configure cookies
            # OR where our cookies expired.
            if not md_candidate:
                snap = fetch_snapshot_markdown(pid)
                snap_len = len(snap.strip())
                stats["snapshot_len"] = snap_len
                if snap and snap_len >= T3_MD_ACCEPT_LEN:
                    md_candidate = snap
                    md_source = "snapshot"

        if md_candidate:
            md = md_candidate
            stats["mode"] = md_source   # "url_fetch" or "snapshot"
            if md_source == "url_fetch":
                stats["fetch_status"] = "ok"
        else:
            # Tier 3 fallback
            md = _metadata_only_markdown(item)
            metadata_only = True
            if len(md.strip()) < 120:
                sql.record_failure(pid, "no_content",
                                   "no PDF and no abstract/metadata body")
                stats["status"] = "skip_no_content"
                return stats
            stats["mode"] = "meta_only"
    else:
        tmp_pdf = work_tmp / f"{pid}.pdf"
        tmp_pdf.write_bytes(pdf)
        t_mineru = time.time()
        try:
            md, images_raw = convert_to_markdown_smart_with_images(tmp_pdf)
        except Exception as e:
            sql.record_failure(pid, "parse", repr(e)[:400])
            stats["status"] = "parse_err"
            return stats
        finally:
            tmp_pdf.unlink(missing_ok=True)
        mineru_secs = time.time() - t_mineru
        if not md or len(md) < 500:
            md = _metadata_only_markdown(item)
            metadata_only = True
            images_raw = {}  # drop any garbage images if md was also garbage
            stats["mode"] = "meta_fallback"
            if len(md.strip()) < 120:
                sql.record_failure(pid, "no_content",
                                   f"parse empty ({len(md) if md else 0}) and no metadata")
                stats["status"] = "skip_no_content"
                return stats

    # Step 3 — build figures (downscale + caption + content_sha)
    figures = _build_figures(md, images_raw) if images_raw else []
    stats["figures"] = len(figures)

    # Step 4 — LLM extract. Pass figures so VLM chain fires; extract_structured
    # returns (paper, provider_name). None → extract failed after retries.
    vlm_input = [
        {"bytes": f["bytes_for_upload"], "caption": f.get("caption"),
         "mime": f.get("mime", "image/jpeg")}
        for f in figures
    ]
    t_llm = time.time()
    paper, extract_provider = extract_structured(
        md, title=item.get("title", ""), paper_id=pid,
        figures=vlm_input or None,
        item_type_hint=item.get("itemType"),
    )
    llm_secs = time.time() - t_llm
    if paper is None:
        sql.record_failure(pid, "extract", "LLM extract returned None")
        stats["status"] = "extract_err"
        return stats
    stats["provider"] = extract_provider or "unknown"

    # Step 5 — Stage A: WebDAV PUT each figure (with per-paper circuit
    # breaker after WEBDAV_FIG_CB_LIMIT consecutive outages). Per-figure
    # failures leave ``webdav_path=None`` in Stage B; later retries pick
    # those up via ``_backfill_pending_figures``.
    if figures:
        stats["figs_uploaded"] = _stage_a_upload_figures(sql, figures, pid)

    # Step 6 — Stage B: atomic SQLite save (figures + papers in one txn).
    # Strip bytes_for_upload before handing to SQLite — that's Stage A's scratch.
    sql_figures = [{k: v for k, v in f.items() if k != "bytes_for_upload"}
                   for f in figures]
    try:
        sql.save_paper_with_figures(
            paper, item_type=item["itemType"], md_text=md,
            figures=sql_figures,
            extract_provider=extract_provider or "unknown",
            mineru_secs=mineru_secs, llm_secs=llm_secs,
        )
    except Exception as e:
        sql.record_failure(pid, "sqlite", repr(e)[:400])
        stats["status"] = "sqlite_err"
        logger.warning("sqlite save_paper_with_figures %s failed: %s", pid, e)
        return stats

    # Step 7 — Stage C: Qdrant (S6 will enrich payload.figure_refs)
    try:
        chunks = qd.upsert_paper(paper, item_type=item["itemType"], markdown=md)
        stats["chunks"] = chunks
    except Exception as e:
        sql.record_failure(pid, "qdrant", repr(e)[:400])
        logger.warning("qdrant upsert failed for %s: %s", pid, e)

    # Step 8 — Neo4j
    paper_ids = _parse_paper_ids(item)
    try:
        ne.write_paper(paper, item_type=item["itemType"],
                       authors=item.get("creators", []),
                       ids=paper_ids)
        if paper_ids["doi"] or paper_ids["arxiv_id"]:
            stats["ids"] = paper_ids
    except Exception as e:
        sql.record_failure(pid, "neo4j", repr(e)[:400])
        logger.warning("neo4j write failed for %s: %s", pid, e)

    # Step 9 — optional md attachment upload (Zotero-level sync, unrelated to
    # figures tree). Gated by ZOTERO_MCP_ATTACH_MD=1 for back-compat.
    if os.environ.get("ZOTERO_MCP_ATTACH_MD", "1") == "1" and not metadata_only:
        try:
            from scripts.backfill_md_attachments import (
                upload_md_as_attachment, add_parent_tag, parent_has_tag, TAG_DONE,
            )
            if not parent_has_tag(pid, TAG_DONE):
                att_key = upload_md_as_attachment(pid, md)
                if att_key:
                    add_parent_tag(pid, TAG_DONE)
                    stats["md_att"] = att_key
        except Exception as e:
            sql.record_failure(pid, "md_attach", repr(e)[:300])
            logger.warning("md attach failed for %s: %s", pid, e)

    stats.update({"mineru_s": round(mineru_secs, 1), "llm_s": round(llm_secs, 1),
                  "total_s": round(time.time() - t_start, 1),
                  "refs": len(paper.references)})
    return stats


def main():
    # force=True: qdrant-client / pyzotero import earlier call basicConfig at
    # WARNING, making our INFO line (extractor active providers, fell back to
    # zai OK) disappear. force=True lets us take over.
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sqlite", default=os.path.expanduser("~/.cache/zotero-mcp/kg.sqlite"))
    ap.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_ZOTERO_URI"))
    ap.add_argument("--neo4j-user", default=os.environ.get("NEO4J_ZOTERO_USER", "neo4j"))
    ap.add_argument("--qdrant-host",
                    default=os.environ.get("QDRANT_HOST", "127.0.0.1"),
                    help="Qdrant host (env: QDRANT_HOST)")
    ap.add_argument("--qdrant-port", type=int,
                    default=int(os.environ.get("QDRANT_PORT", "6333")),
                    help="Qdrant port (env: QDRANT_PORT)")
    ap.add_argument("--qdrant-collection",
                    default=os.environ.get("QDRANT_COLLECTION", "zotero_library"),
                    help="Qdrant collection name (env: QDRANT_COLLECTION)")
    ap.add_argument("--keys", nargs="+", default=None, help="specific Zotero keys")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("INGEST_WORKERS", "3")),
                    help="Concurrent pipeline workers (MinerU API caps at 3)")
    args = ap.parse_args()

    _z_init()
    password = os.environ["NEO4J_ZOTERO_PASSWORD"]
    sql, qd, ne = make_writers(
        sqlite_path=args.sqlite,
        neo4j_uri=args.neo4j_uri, neo4j_user=args.neo4j_user, neo4j_password=password,
        qdrant_host=args.qdrant_host, qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
    )

    if args.keys:
        # fetch just those items
        items = []
        for k in args.keys:
            items.append(_z_get_json(f"/items/{k}", {"format": "json"})["data"])
    else:
        items = list_ingest_candidates(limit=args.limit)
    logger.info("ingest plan: %d items", len(items))

    work_tmp = Path("/tmp/zotero_kg_work")
    work_tmp.mkdir(exist_ok=True)

    stats_per_status: dict[str, int] = {}
    t0 = time.time()
    from zotero_mcp.webdav import WebDAVOutageError
    CB_LIMIT = int(os.environ.get("INGEST_WEBDAV_OUTAGE_LIMIT", "8"))

    def _is_webdav_outage_crash(err_repr: str) -> bool:
        return "WebDAVOutageError" in err_repr or "webdav put unavailable" in err_repr

    if args.workers <= 1:
        for i, it in enumerate(items, 1):
            s = ingest_one(it, sql=sql, qd=qd, ne=ne, work_tmp=work_tmp)
            stats_per_status[s["status"]] = stats_per_status.get(s["status"], 0) + 1
            logger.info("[%d/%d] %s %s", i, len(items), it["key"], s)
    else:
        logger.info("running with %d concurrent workers", args.workers)
        consecutive_webdav_out = 0
        tripped = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(ingest_one, it, sql=sql, qd=qd, ne=ne,
                                work_tmp=work_tmp): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                it = futs[fut]
                try:
                    s = fut.result()
                    consecutive_webdav_out = 0
                except WebDAVOutageError as e:
                    s = {"pid": it["key"], "status": "webdav_outage", "err": repr(e)[:200]}
                    consecutive_webdav_out += 1
                except Exception as e:
                    err_repr = repr(e)[:200]
                    if _is_webdav_outage_crash(err_repr):
                        consecutive_webdav_out += 1
                        s = {"pid": it["key"], "status": "webdav_outage", "err": err_repr}
                    else:
                        consecutive_webdav_out = 0
                        s = {"pid": it["key"], "status": "crash", "err": err_repr}
                stats_per_status[s["status"]] = stats_per_status.get(s["status"], 0) + 1
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(items) - i) / rate if rate > 0 else 0
                logger.info("[%d/%d @ %.0f%%] %s %s  rate=%.2f/s  ETA=%.0fm",
                            i, len(items), 100*i/len(items), it["key"], s,
                            rate, eta/60)
                if consecutive_webdav_out >= CB_LIMIT and not tripped:
                    tripped = True
                    logger.error("CIRCUIT BREAKER: %d consecutive WebDAV outages — "
                                 "aborting remaining %d futures. Re-run when WebDAV recovers.",
                                 consecutive_webdav_out, len(futs) - i)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    sql.close()
    ne.close()
    logger.info("DONE in %.1fm. status breakdown: %s", (time.time()-t0)/60,
                stats_per_status)


if __name__ == "__main__":
    main()

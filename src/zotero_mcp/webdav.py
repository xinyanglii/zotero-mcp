"""
Zotero WebDAV helpers.

Zotero's built-in WebDAV sync uses a specific format: each attachment is
stored as ``<attachmentKey>.zip`` (zipped file) + ``<attachmentKey>.prop``
(XML metadata) under the user's WebDAV root (typically
``https://dav.jianguoyun.com/dav/zotero/``).

This module provides small, dependency-free helpers for:

- Downloading an attachment file from WebDAV given its Zotero key
- Uploading a file (PDF / markdown / anything) as a Zotero-compatible
  attachment pair

Enable by setting ``ZOTERO_WEBDAV_USER`` + ``ZOTERO_WEBDAV_PASS`` +
(optional) ``ZOTERO_WEBDAV_URL`` env vars. ``ZOTERO_WEBDAV_URL`` defaults
to ``https://dav.jianguoyun.com/dav/zotero`` (trailing slash optional,
trimmed).

The ``create_zotero_webdav_attachment`` function combines Zotero item
creation + WebDAV upload + md5/mtime patch in one call — use it from
``tools/_helpers.py:_download_and_attach_pdf`` and from the ingest pipeline
when auto-attaching extracted markdown.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile

logger = logging.getLogger(__name__)


# Jianguoyun paid plan: ≤1500 requests / 30min ≈ 0.83 QPS global.
# 2026-04-24 incident: concurrent zkg (w=4) + batch-ingest (w=4) + 2nd zkg (w=2)
# burst to ~20 QPS. Jianguoyun returned 503 for every request with the account's
# BASIC auth; unauth'd PROPFIND to the same URL returned 401 cleanly, proving
# it was account-level throttle, not 全站故障 as earlier memories recorded.
# This limiter caps per-process rate; run at most 2 concurrent processes to
# stay under the hard limit with margin.
_RATE_LOCK = threading.Lock()
_LAST_WEBDAV_CALL = [0.0]


def _rate_limit_wait() -> None:
    min_interval = float(os.environ.get("WEBDAV_MIN_INTERVAL_S", "3.0"))
    with _RATE_LOCK:
        now = time.monotonic()
        wait = min_interval - (now - _LAST_WEBDAV_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_WEBDAV_CALL[0] = time.monotonic()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def webdav_enabled() -> bool:
    """True when the env has WebDAV credentials configured."""
    return bool(os.environ.get("ZOTERO_WEBDAV_USER")
                and os.environ.get("ZOTERO_WEBDAV_PASS"))


def _webdav_root() -> str:
    return os.environ.get("ZOTERO_WEBDAV_URL",
                          "https://dav.jianguoyun.com/dav/zotero").rstrip("/")


def _auth_header() -> str:
    user = os.environ["ZOTERO_WEBDAV_USER"]
    passwd = os.environ["ZOTERO_WEBDAV_PASS"]
    return "Basic " + base64.b64encode(f"{user}:{passwd}".encode()).decode()


class WebDAVOutageError(Exception):
    """Raised when WebDAV is persistently unavailable (403/429/5xx) after
    all retries. Callers that loop over many papers should treat this as a
    signal to pause rather than crash individual items."""


def _retry_urlopen(req, timeout: int, max_attempts: int = 4):
    """urlopen wrapper with exponential backoff on 429/5xx and Jianguoyun's
    soft-throttle 403. 404 is surfaced unchanged. Final failure raises
    WebDAVOutageError so outer loops can circuit-break.

    Defaults: 4 attempts with 5/10/20/40s backoff (~75s total). Each attempt
    is gated by the per-process rate limiter (see _rate_limit_wait) so we
    stay under Jianguoyun's 1500-req/30min ceiling — exceeding that returns
    503 on every request for ~30min until the rolling window drains.
    Circuit-breaker in ingest.py catches persistent outage and pauses the
    whole pipeline."""
    delay = 5.0
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        _rate_limit_wait()
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code in (403, 429) or 500 <= e.code < 600:
                last_err = e
                ra_raw = e.headers.get("Retry-After") if e.headers else None
                try:
                    sleep_s = int(ra_raw) if ra_raw else delay
                except ValueError:
                    sleep_s = delay
                sleep_s = min(sleep_s, 40)
                logger.warning("WebDAV %d on attempt %d/%d — sleeping %.1fs",
                               e.code, attempt, max_attempts, sleep_s)
                time.sleep(sleep_s)
                delay = min(delay * 2, 40)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            logger.warning("WebDAV transport err on attempt %d/%d (%s) — sleeping %.1fs",
                           attempt, max_attempts, type(e).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 40)
    raise WebDAVOutageError(
        f"WebDAV unavailable after {max_attempts} attempts: {last_err!r}")


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def fetch_attachment_bytes(attachment_key: str, timeout: int = 60) -> bytes | None:
    """Download ``<key>.zip`` from WebDAV and return the inner file's bytes.

    Returns ``None`` on 404 / missing / bad zip. Raises ``WebDAVOutageError``
    on persistent 403/429/5xx so callers can pause rather than mis-mark
    the paper as permanently failed.
    """
    url = f"{_webdav_root()}/{attachment_key}.zip"
    req = urllib.request.Request(url, headers={"Authorization": _auth_header()})
    try:
        with _retry_urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("WebDAV miss %s (404)", attachment_key)
            return None
        raise
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # Skip directory entries; Zotero's sync zip contains a single
            # file but a hostile or malformed zip could include traversal
            # paths or dir entries that would raise IsADirectoryError here.
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                return zf.read(name)
            return None
    except zipfile.BadZipFile:
        logger.warning("WebDAV %s returned non-zip content", attachment_key)
        return None


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
def upload_attachment_bytes(
    attachment_key: str,
    filename: str,
    file_bytes: bytes,
    *,
    timeout: int = 120,
) -> tuple[str, int]:
    """Upload ``<key>.zip`` + ``<key>.prop`` following Zotero sync protocol.

    Args:
        attachment_key: The Zotero attachment item key (not the parent paper's!).
        filename: Filename inside the zip (must match the attachment's
            ``filename`` field so Zotero desktop lines them up).
        file_bytes: Raw bytes of the file to attach.

    Returns:
        ``(md5_hex, mtime_ms)`` — caller should PATCH these onto the Zotero
        attachment item so the desktop client recognizes the upload.
    """
    root = _webdav_root()
    md5_hex = hashlib.md5(file_bytes).hexdigest()
    mtime_ms = int(time.time() * 1000)

    # zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, file_bytes)
    zip_bytes = buf.getvalue()

    # .prop XML
    prop_xml = (
        f'<properties version="1">\n'
        f'<mtime>{mtime_ms}</mtime>\n'
        f'<hash>{md5_hex}</hash>\n'
        f'</properties>\n'
    ).encode()

    headers = {"Authorization": _auth_header()}
    uploaded: list[str] = []
    try:
        for suffix, body in ((".zip", zip_bytes), (".prop", prop_xml)):
            req = urllib.request.Request(
                f"{root}/{attachment_key}{suffix}",
                data=body, method="PUT", headers=headers,
            )
            with _retry_urlopen(req, timeout=timeout) as r:
                r.read()
            uploaded.append(suffix)
    except Exception:
        # Partial upload cleanup: if .zip succeeded but .prop failed, delete
        # the orphan .zip so Zotero desktop doesn't see a broken attachment.
        for suffix in uploaded:
            try:
                req = urllib.request.Request(
                    f"{root}/{attachment_key}{suffix}",
                    method="DELETE", headers=headers,
                )
                urllib.request.urlopen(req, timeout=timeout).read()
            except Exception as cleanup_err:
                logger.debug("orphan cleanup of %s%s failed: %s",
                             attachment_key, suffix, cleanup_err)
        raise
    return md5_hex, mtime_ms


# ---------------------------------------------------------------------------
# Figures — independent WebDAV tree, NOT routed through Zotero attachments.
# Path layout:
#     <figures_root>/<paper_id>/<mineru_name>.jpg
# ``figures_root`` defaults to ``<zotero_webdav_url sibling>/zotero-kg-figures``
# (e.g. Jianguoyun ``/dav/zotero-kg-figures``). Override via env
# ``ZOTERO_KG_FIGURES_URL``. MKCOL on parent + per-paper dir is idempotent
# (201 or 405 both OK). PUT retries via ``_retry_urlopen``.
# ---------------------------------------------------------------------------
def _figures_root() -> str:
    override = os.environ.get("ZOTERO_KG_FIGURES_URL", "").rstrip("/")
    if override:
        return override
    # Sibling of Zotero's WebDAV root. For Jianguoyun this resolves to
    # https://dav.jianguoyun.com/dav/zotero-kg-figures (tested via MKCOL 201).
    zotero_root = _webdav_root()
    parent, _, _ = zotero_root.rpartition("/")
    if not parent:
        raise RuntimeError(
            "cannot derive figures root from ZOTERO_WEBDAV_URL="
            f"{zotero_root!r} — set ZOTERO_KG_FIGURES_URL explicitly")
    return f"{parent}/zotero-kg-figures"


_figures_root_ensured = False


def _mkcol(url: str, *, timeout: int = 20) -> None:
    """Idempotent MKCOL — 201 Created / 405 Already Exists both OK; raise on
    other errors. Does NOT use ``_retry_urlopen`` because MKCOL semantics are
    different: most errors here mean a permissions / path problem, not a
    transient outage worth waiting on."""
    headers = {"Authorization": _auth_header()}
    req = urllib.request.Request(url, method="MKCOL", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        if e.code == 405:
            return  # already exists
        raise


def _ensure_figures_root() -> str:
    """Lazy one-time MKCOL on the figures root. Returns the root URL."""
    global _figures_root_ensured
    root = _figures_root()
    if not _figures_root_ensured:
        _mkcol(root)
        _figures_root_ensured = True
    return root


def put_figure(
    paper_id: str, mineru_name: str, img_bytes: bytes,
    *,
    content_type: str = "image/jpeg",
    timeout: int = 60,
) -> str:
    """PUT one figure into the independent KG figures tree.

    Returns the WebDAV path (relative to figures root) on success. Raises
    ``WebDAVOutageError`` after retries exhausted. Caller decides whether to
    record the failure or let the stage fail.

    - Idempotent: re-running overwrites the same path (content-addressed).
    - Ensures ``<figures_root>`` + ``<figures_root>/<paper_id>`` exist
      (MKCOL once per process for the root, once per paper for its subdir).
    """
    root = _ensure_figures_root()
    paper_dir = f"{root}/{paper_id}"
    _mkcol(paper_dir)  # cheap; Jianguoyun returns 405 if it already exists
    rel_path = f"{paper_id}/{mineru_name}"
    url = f"{root}/{rel_path}"
    headers = {
        "Authorization": _auth_header(),
        "Content-Type": content_type,
        "Content-Length": str(len(img_bytes)),
    }
    req = urllib.request.Request(url, data=img_bytes, method="PUT", headers=headers)
    with _retry_urlopen(req, timeout=timeout) as r:
        r.read()
    return rel_path


def put_figures_zip(
    paper_id: str, figures: list[dict], *,
    timeout: int = 120,
) -> dict[str, str]:
    """Batch-upload ALL figures of one paper as a single zip.

    Replaces 30-50 independent put_figure calls (each ~5-7s including network
    upload + 1.5s rate limit) with one PUT of <paper_id>.zip containing
    every figure under its mineru_name. Typical savings: 90%+ of wall-clock
    WebDAV time per paper. See 2026-04-24 incident memo.

    Returns {mineru_name: "<paper_id>.zip::<mineru_name>"} mapping, which
    callers write into SQLite `figures.webdav_path`. The "::" separator
    flags the entry as living inside a zip (vs the legacy independent-file
    layout used by put_figure, which stays untouched).

    Raises WebDAVOutageError on all-attempts-fail; all-or-nothing (no
    partial upload — callers mark every figure pending and retry later).
    """
    import io
    import zipfile
    root = _ensure_figures_root()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in figures:
            zf.writestr(f["mineru_name"], f["bytes_for_upload"])
    zip_bytes = buf.getvalue()
    rel = f"{paper_id}.zip"
    url = f"{root}/{rel}"
    headers = {
        "Authorization": _auth_header(),
        "Content-Type": "application/zip",
        "Content-Length": str(len(zip_bytes)),
    }
    req = urllib.request.Request(url, data=zip_bytes, method="PUT", headers=headers)
    with _retry_urlopen(req, timeout=timeout) as r:
        r.read()
    return {f["mineru_name"]: f"{rel}::{f['mineru_name']}" for f in figures}


def delete_figure(paper_id: str, mineru_name: str, *, timeout: int = 30) -> None:
    """Best-effort DELETE of a single figure — used by tests and by a
    potential ``zkg.py dedupe-figures`` tool. 404 is swallowed (nothing to
    delete is not an error)."""
    root = _figures_root()
    url = f"{root}/{paper_id}/{mineru_name}"
    req = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": _auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return
        raise


# ---------------------------------------------------------------------------
# combined: create Zotero attachment item + WebDAV upload + md5/mtime patch
# ---------------------------------------------------------------------------
def create_zotero_webdav_attachment(
    zot,
    *,
    parent_key: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    title: str | None = None,
    extra_tags: list[str] | None = None,
) -> str | None:
    """Create an imported_file attachment in Zotero + upload via WebDAV.

    Args:
        zot: A ``pyzotero.zotero.Zotero`` write client.
        parent_key: The parent paper/item's Zotero key.
        file_bytes: Raw bytes to attach.
        filename: Filename to register (e.g. ``"paper.pdf"`` or ``"X.md"``).
        content_type: MIME type (``application/pdf`` / ``text/markdown``).
        title: Optional display title for the attachment.
        extra_tags: Optional tags to add to the attachment.

    Returns:
        The new attachment's Zotero key on success, ``None`` on failure.
    """
    if not webdav_enabled():
        logger.warning("WebDAV not configured (ZOTERO_WEBDAV_USER/PASS missing)")
        return None

    template = zot.item_template("attachment", "imported_file")
    template["parentItem"] = parent_key
    template["filename"] = filename
    template["contentType"] = content_type
    template["title"] = title or filename
    if extra_tags:
        template["tags"] = [{"tag": t} for t in extra_tags]

    result = zot.create_items([template])
    if not result.get("successful"):
        logger.warning("Zotero attachment create failed: %s", result.get("failed"))
        return None
    att = list(result["successful"].values())[0]
    att_key = att["key"]
    att_version = att.get("version")

    try:
        md5_hex, mtime_ms = upload_attachment_bytes(att_key, filename, file_bytes)
    except Exception as e:
        logger.exception("WebDAV upload failed for %s: %s", att_key, e)
        return None

    # patch md5 + mtime so Zotero desktop recognizes the upload.
    # Use direct REST PATCH — pyzotero's ``update_item`` expects a specific
    # dict shape that diverges across versions, and its internal error
    # handling swallows 412s we want to surface. Direct urllib keeps the
    # semantics explicit.
    try:
        api_base = zot.endpoint.rstrip("/")
        # pyzotero stores library_type already plural ("users" / "groups"),
        # NOT "user"/"group" — do NOT append another "s".
        url = f"{api_base}/{zot.library_type}/{zot.library_id}/items/{att_key}"
        body = json.dumps({"md5": md5_hex, "mtime": mtime_ms}).encode()
        headers = {
            "Zotero-API-Key": zot.api_key,
            "Zotero-API-Version": "3",
            "Content-Type": "application/json",
        }
        if att_version is not None:
            headers["If-Unmodified-Since-Version"] = str(att_version)
        req = urllib.request.Request(url, data=body, method="PATCH", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception as e:
        # md5/mtime patch is load-bearing (Zotero desktop won't recognize
        # the upload without it). Log at warning, not debug.
        logger.warning("md5/mtime PATCH failed for %s: %s", att_key, e)

    return att_key

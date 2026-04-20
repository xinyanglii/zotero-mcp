"""
Citation resolver: cross-matches extracted references to Semantic Scholar /
OpenAlex → returns canonical IDs (DOI, S2, OpenAlex) + optional Zotero key if
the cited paper is already in the user's library.

Strategy (per reference, first hit wins):
  1. Semantic Scholar /paper/search by title → filter by first-author last
     name and year (if available)
  2. OpenAlex /works?search=<title> → same filter
  3. Give up: keep as ExternalRef node with title + author_year only

Caches successful lookups in a JSON file keyed by `title|first_author|year`
to avoid re-hitting S2 on re-runs. S2 polite-pool: 1 req/sec without key.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_PATH = Path(os.getenv(
    "ZOTERO_CITE_CACHE",
    str(Path.home() / ".cache" / "zotero-mcp" / "citation_cache.json"),
))
S2_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = os.getenv(
    "ZOTERO_USER_AGENT",
    "zotero-mcp-kg/0.1 (mailto:lxymario@hotmail.com)",
)


# ---------------- cache ----------------
def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _cache_key(title: str, author_year: str) -> str:
    t = re.sub(r"\s+", " ", title.lower().strip())[:120]
    return f"{t}|{author_year.lower().strip()}"


# ---------------- HTTP ----------------
def _http_json(url: str, timeout: int = 15) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        logger.debug("http_json %s -> %s", url[:100], e)
        return None


# ---------------- lookup ----------------
def _parse_author_year(s: str) -> tuple[str | None, int | None]:
    """'Liu 2023' → ('liu', 2023); returns (None, None) if no match."""
    m = re.match(r"^([A-Za-z\u4e00-\u9fff\-]+)\s+(\d{4})", s.strip())
    if not m:
        return (None, None)
    return (m.group(1).lower(), int(m.group(2)))


def _lookup_s2(title: str, first_author: str | None, year: int | None) -> dict | None:
    q = urllib.parse.quote(title[:200])
    url = (
        f"{S2_BASE}/paper/search?query={q}&limit=5"
        "&fields=title,externalIds,year,authors,venue,openAccessPdf"
    )
    data = _http_json(url)
    if not data or not data.get("data"):
        return None
    for hit in data["data"]:
        # title similarity filter (cheap)
        if _title_similar(hit.get("title", ""), title) < 0.85:
            continue
        if year and hit.get("year") and abs(int(hit["year"]) - year) > 1:
            continue
        if first_author and not _has_author(hit.get("authors", []), first_author):
            continue
        return {
            "source": "s2",
            "s2_id": hit.get("paperId"),
            "doi": (hit.get("externalIds") or {}).get("DOI"),
            "arxiv": (hit.get("externalIds") or {}).get("ArXiv"),
            "title": hit.get("title"),
            "year": hit.get("year"),
            "venue": hit.get("venue"),
            "oa_pdf": (hit.get("openAccessPdf") or {}).get("url"),
        }
    return None


def _lookup_openalex(title: str, first_author: str | None, year: int | None) -> dict | None:
    q = urllib.parse.quote(title[:200])
    url = f"{OPENALEX_BASE}/works?search={q}&per_page=5"
    data = _http_json(url)
    if not data or not data.get("results"):
        return None
    for hit in data["results"]:
        if _title_similar(hit.get("title", ""), title) < 0.85:
            continue
        if year and hit.get("publication_year") and abs(hit["publication_year"] - year) > 1:
            continue
        if first_author:
            authorships = hit.get("authorships") or []
            names = [(a.get("author") or {}).get("display_name", "") for a in authorships]
            if not any(first_author in n.lower() for n in names):
                continue
        oa_pdf = ((hit.get("best_oa_location") or {}).get("pdf_url")
                  or (hit.get("primary_location") or {}).get("pdf_url"))
        return {
            "source": "openalex",
            "openalex_id": hit.get("id"),
            "doi": (hit.get("doi") or "").replace("https://doi.org/", "") or None,
            "title": hit.get("title"),
            "year": hit.get("publication_year"),
            "oa_pdf": oa_pdf,
        }
    return None


def _title_similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    na = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    nb = re.sub(r"[^a-z0-9]+", " ", b.lower()).strip()
    return SequenceMatcher(None, na, nb).ratio()


def _has_author(authors: list[dict], first_author_last: str) -> bool:
    for a in authors:
        name = (a.get("name") or "").lower()
        if first_author_last in name:
            return True
    return False


# ---------------- main API ----------------
def resolve_reference(
    cited_title: str,
    cited_author_year: str,
    *,
    use_cache: bool = True,
    polite_delay: float = 1.0,
) -> dict[str, Any] | None:
    """Resolve a single reference to a canonical record.

    Returns dict with keys from _lookup_s2/_openalex, or None if unresolvable.
    """
    author, year = _parse_author_year(cited_author_year)
    ck = _cache_key(cited_title, cited_author_year)
    cache = _load_cache() if use_cache else {}
    if ck in cache:
        return cache[ck]

    result = _lookup_s2(cited_title, author, year)
    if polite_delay > 0:
        time.sleep(polite_delay)
    if not result:
        result = _lookup_openalex(cited_title, author, year)
        if polite_delay > 0:
            time.sleep(polite_delay / 2)

    if use_cache and result is not None:
        cache[ck] = result
        _save_cache(cache)
    return result


def build_zotero_doi_index(zotero_client) -> dict[str, str]:
    """Return dict {lowercased_doi: zotero_itemKey} for quick in-library matching.

    `zotero_client` must be a `pyzotero.Zotero` instance.
    """
    index: dict[str, str] = {}
    for item in zotero_client.everything(zotero_client.items(format="json", limit=100)):
        d = item.get("data") or {}
        doi = (d.get("DOI") or "").strip().lower()
        if doi:
            index[doi] = d["key"]
    return index

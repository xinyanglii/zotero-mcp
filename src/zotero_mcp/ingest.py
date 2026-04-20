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
from pathlib import Path

from .extractor import extract_structured
from .kg_store import make_writers
from .parser_factory import convert_to_markdown_smart

logger = logging.getLogger("zotero_mcp.ingest")

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
    """Yield top-level items eligible for ingest (not tagged skip/duplicate/orphan)."""
    item_types = item_types or {
        "journalArticle", "preprint", "conferencePaper",
        "thesis", "bookSection", "book", "report",
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


def _webdav_auth_header() -> str:
    creds = f"{os.environ['ZOTERO_WEBDAV_USER']}:{os.environ['ZOTERO_WEBDAV_PASS']}"
    return "Basic " + base64.b64encode(creds.encode()).decode()


def _webdav_fetch(attachment_key: str) -> bytes | None:
    """Download <key>.zip from /dav/zotero/ and return the inner PDF."""
    url = f"https://dav.jianguoyun.com/dav/zotero/{attachment_key}.zip"
    req = urllib.request.Request(url, headers={"Authorization": _webdav_auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            zbytes = r.read()
    except Exception as e:
        logger.debug("WebDAV miss %s: %s", attachment_key, e)
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".pdf"):
                    return zf.read(name)
    except Exception as e:
        logger.warning("zip parse err %s: %s", attachment_key, e)
        return None
    return None


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
def ingest_one(item: dict, *, sql, qd, ne, work_tmp: Path) -> dict[str, str | float | int]:
    """Process a single item. Returns stats dict."""
    pid = item["key"]
    stats: dict[str, str | float | int] = {"pid": pid, "status": "ok"}
    t_start = time.time()

    if sql.already_done(pid):
        stats["status"] = "skip_done"
        return stats

    pdf = resolve_pdf_bytes(item)
    if pdf is None:
        sql.record_failure(pid, "pdf_resolve", "no PDF available")
        stats["status"] = "skip_no_pdf"
        return stats
    tmp_pdf = work_tmp / f"{pid}.pdf"
    tmp_pdf.write_bytes(pdf)

    t_mineru = time.time()
    try:
        md = convert_to_markdown_smart(tmp_pdf)
    except Exception as e:
        sql.record_failure(pid, "parse", repr(e)[:400])
        stats["status"] = "parse_err"
        return stats
    finally:
        tmp_pdf.unlink(missing_ok=True)
    mineru_secs = time.time() - t_mineru
    if not md or len(md) < 500:
        sql.record_failure(pid, "parse", f"md too short: {len(md) if md else 0}")
        stats["status"] = "parse_empty"
        return stats

    t_llm = time.time()
    paper = extract_structured(md, title=item.get("title", ""), paper_id=pid)
    llm_secs = time.time() - t_llm
    if paper is None:
        sql.record_failure(pid, "extract", "LLM extract returned None")
        stats["status"] = "extract_err"
        return stats

    try:
        chunks = qd.upsert_paper(paper, item_type=item["itemType"], markdown=md)
        stats["chunks"] = chunks
    except Exception as e:
        sql.record_failure(pid, "qdrant", repr(e)[:400])
        logger.warning("qdrant upsert failed for %s: %s", pid, e)

    try:
        ne.write_paper(paper, item_type=item["itemType"],
                       authors=item.get("creators", []))
    except Exception as e:
        sql.record_failure(pid, "neo4j", repr(e)[:400])
        logger.warning("neo4j write failed for %s: %s", pid, e)

    sql.save_paper(paper, item_type=item["itemType"],
                   md_text=md, mineru_secs=mineru_secs, llm_secs=llm_secs)
    stats.update({"mineru_s": round(mineru_secs, 1), "llm_s": round(llm_secs, 1),
                  "total_s": round(time.time() - t_start, 1),
                  "refs": len(paper.references)})
    return stats


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sqlite", default=os.path.expanduser("~/.cache/zotero-mcp/kg.sqlite"))
    ap.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_ZOTERO_URI"))
    ap.add_argument("--neo4j-user", default=os.environ.get("NEO4J_ZOTERO_USER", "neo4j"))
    ap.add_argument("--qdrant-host", default="100.68.195.10")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-collection", default="zotero_library")
    ap.add_argument("--keys", nargs="+", default=None, help="specific Zotero keys")
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
    for i, it in enumerate(items, 1):
        t = time.time()
        s = ingest_one(it, sql=sql, qd=qd, ne=ne, work_tmp=work_tmp)
        stats_per_status[s["status"]] = stats_per_status.get(s["status"], 0) + 1
        logger.info("[%d/%d] %s %s", i, len(items), it["key"], s)

    sql.close()
    ne.close()
    logger.info("DONE. status breakdown: %s", stats_per_status)


if __name__ == "__main__":
    main()

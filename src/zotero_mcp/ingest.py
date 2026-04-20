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


def ingest_one(item: dict, *, sql, qd, ne, work_tmp: Path) -> dict[str, str | float | int]:
    """Process a single item. Returns stats dict."""
    pid = item["key"]
    stats: dict[str, str | float | int] = {"pid": pid, "status": "ok"}
    t_start = time.time()

    if sql.already_done(pid):
        stats["status"] = "skip_done"
        return stats

    pdf = resolve_pdf_bytes(item)
    mineru_secs = 0.0
    metadata_only = False
    md = None

    if pdf is None:
        # graceful degrade: build metadata-only md from abstract + Zotero fields
        md = _metadata_only_markdown(item)
        metadata_only = True
        if len(md.strip()) < 120:  # title-only with no abstract → really nothing
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
            md = convert_to_markdown_smart(tmp_pdf)
        except Exception as e:
            sql.record_failure(pid, "parse", repr(e)[:400])
            stats["status"] = "parse_err"
            return stats
        finally:
            tmp_pdf.unlink(missing_ok=True)
        mineru_secs = time.time() - t_mineru
        if not md or len(md) < 500:
            # parse produced nothing useful → fall through to metadata-only
            md = _metadata_only_markdown(item)
            metadata_only = True
            stats["mode"] = "meta_fallback"
            if len(md.strip()) < 120:
                sql.record_failure(pid, "no_content",
                                   f"parse empty ({len(md) if md else 0}) and no metadata")
                stats["status"] = "skip_no_content"
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

    # Attach the extracted markdown as a Zotero child attachment so downstream
    # agents can read the parsed text directly, without re-running MinerU.
    # Gated by ZOTERO_MCP_ATTACH_MD=1 to preserve backwards compatibility.
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
    logging.basicConfig(level=logging.INFO,
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
    if args.workers <= 1:
        for i, it in enumerate(items, 1):
            s = ingest_one(it, sql=sql, qd=qd, ne=ne, work_tmp=work_tmp)
            stats_per_status[s["status"]] = stats_per_status.get(s["status"], 0) + 1
            logger.info("[%d/%d] %s %s", i, len(items), it["key"], s)
    else:
        logger.info("running with %d concurrent workers", args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(ingest_one, it, sql=sql, qd=qd, ne=ne,
                                work_tmp=work_tmp): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                it = futs[fut]
                try:
                    s = fut.result()
                except Exception as e:
                    s = {"pid": it["key"], "status": "crash", "err": repr(e)[:200]}
                stats_per_status[s["status"]] = stats_per_status.get(s["status"], 0) + 1
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(items) - i) / rate if rate > 0 else 0
                logger.info("[%d/%d @ %.0f%%] %s %s  rate=%.2f/s  ETA=%.0fm",
                            i, len(items), 100*i/len(items), it["key"], s,
                            rate, eta/60)

    sql.close()
    ne.close()
    logger.info("DONE in %.1fm. status breakdown: %s", (time.time()-t0)/60,
                stats_per_status)


if __name__ == "__main__":
    main()

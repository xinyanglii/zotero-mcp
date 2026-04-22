#!/usr/bin/env python3
"""T1-b backfill: populate Paper.doi / Paper.arxiv_id + SAME_WORK_AS edges for
papers already in Neo4j, by re-reading Zotero item.data.

Usage:
    python scripts/backfill_paper_identifiers.py [--limit N] [--resume]
                                                  [--retry-missing] [--dry-run]

Idempotent: MERGE + SET + SAME_WORK_AS MERGE all safe to re-run. Progress
persisted to ~/.cache/zotero-mcp/t1_backfill.sqlite so --resume picks up
where a prior run stopped (explicit Ctrl-C or crash).

See plans/zotero-kg-t1-spec.md §3 for semantics.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Load env early so ZOTERO / NEO4J env are available before imports
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env.local"), override=False)
except ImportError:
    pass

# Make `zotero_mcp` importable when run as a standalone script
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from zotero_mcp.ingest import _parse_paper_ids  # noqa: E402
from zotero_mcp.kg_store import Neo4jWriter  # noqa: E402

logger = logging.getLogger("t1_backfill")
logging.basicConfig(
    level=logging.INFO, force=True,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

PROGRESS_DB = Path(os.getenv(
    "T1_BACKFILL_DB",
    os.path.expanduser("~/.cache/zotero-mcp/t1_backfill.sqlite"),
))
BATCH = 50  # Zotero API /items?itemKey=... hard cap
TYPES = "preprint || journalArticle || conferencePaper"


# ---------------------------------------------------------------------------
# Progress SQLite (shared with T1-c enrichment; separate table)
# ---------------------------------------------------------------------------
def _init_progress_db() -> sqlite3.Connection:
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROGRESS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_progress (
            paper_id    TEXT PRIMARY KEY,
            status      TEXT NOT NULL,  -- done / zotero_missing / neo4j_err / no_ids
            doi         TEXT,
            arxiv_id    TEXT,
            same_work_doi_links    INTEGER DEFAULT 0,
            same_work_arxiv_links  INTEGER DEFAULT 0,
            updated_at  REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


def _mark(conn, paper_id: str, status: str, doi=None, arxiv_id=None,
          doi_links=0, arxiv_links=0):
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO paper_progress
               (paper_id, status, doi, arxiv_id,
                same_work_doi_links, same_work_arxiv_links, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (paper_id, status, doi, arxiv_id,
             doi_links, arxiv_links, time.time()),
        )


# ---------------------------------------------------------------------------
# Zotero helpers
# ---------------------------------------------------------------------------
def _z_hdr():
    return {
        "Zotero-API-Key": os.environ["ZOTERO_API_KEY"],
        "Zotero-API-Version": "3",
    }


def _z_base():
    return f"https://api.zotero.org/users/{os.environ['ZOTERO_LIBRARY_ID']}"


def fetch_paper_keys() -> list[str]:
    """Pull all paper-type top-level Zotero keys via the `keys` format (fast)."""
    url = (f"{_z_base()}/items/top?format=keys"
           f"&itemType={urllib.parse.quote(TYPES)}")
    req = urllib.request.Request(url, headers=_z_hdr())
    with urllib.request.urlopen(req, timeout=120) as r:
        keys = r.read().decode().split()
    return [k for k in keys if k]


def fetch_batch(keys: list[str]) -> dict[str, dict]:
    """Return ``{key: item_data}`` for at most 50 keys. Missing keys simply
    don't appear in the dict — spec §3 step 6."""
    url = (f"{_z_base()}/items?format=json&itemKey={','.join(keys)}")
    req = urllib.request.Request(url, headers=_z_hdr())
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.loads(r.read())
    return {it["key"]: it["data"] for it in items if "key" in it}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run(args) -> None:
    conn = _init_progress_db()

    if args.limit and args.limit > 0:
        all_keys = fetch_paper_keys()[: args.limit]
    else:
        all_keys = fetch_paper_keys()
    logger.info("Zotero paper-type keys to process: %d", len(all_keys))

    # Resume filter
    done_statuses = {"done"}
    if not args.retry_missing:
        done_statuses.add("zotero_missing")
    if not args.retry_noid:
        done_statuses.add("no_ids")
    if args.resume:
        cur = conn.execute(
            "SELECT paper_id FROM paper_progress WHERE status IN ("
            + ",".join("?" * len(done_statuses)) + ")",
            tuple(done_statuses),
        )
        already = {r[0] for r in cur.fetchall()}
        all_keys = [k for k in all_keys if k not in already]
        logger.info("resume: %d remaining after skipping prior done/skip",
                    len(all_keys))

    if args.dry_run:
        logger.info("DRY RUN — would process %d papers, no writes", len(all_keys))
        return

    # Neo4j writer
    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )

    # Process batches
    stats = {"done": 0, "zotero_missing": 0, "neo4j_err": 0, "no_ids": 0,
             "same_work_links": 0}
    t0 = time.time()
    try:
        for bi in range(0, len(all_keys), BATCH):
            batch_keys = all_keys[bi: bi + BATCH]
            try:
                data_by_key = fetch_batch(batch_keys)
            except Exception as e:
                logger.warning("batch fetch error (%d keys): %s — retry once",
                               len(batch_keys), e)
                time.sleep(2)
                try:
                    data_by_key = fetch_batch(batch_keys)
                except Exception as e2:
                    logger.error("batch fetch failed twice, skipping batch: %s", e2)
                    for k in batch_keys:
                        _mark(conn, k, "zotero_missing")
                    stats["zotero_missing"] += len(batch_keys)
                    continue

            for k in batch_keys:
                d = data_by_key.get(k)
                if d is None:
                    _mark(conn, k, "zotero_missing")
                    stats["zotero_missing"] += 1
                    continue
                ids = _parse_paper_ids(d)
                if not (ids["doi"] or ids["arxiv_id"]):
                    _mark(conn, k, "no_ids")
                    stats["no_ids"] += 1
                    continue

                # MERGE SET identifiers + SAME_WORK_AS scan
                try:
                    with ne.driver.session() as s:
                        s.run(
                            """MERGE (p:Source:Paper {id: $pid})
                               SET p.doi = $doi, p.arxiv_id = $arxiv_id""",
                            pid=k, doi=ids["doi"], arxiv_id=ids["arxiv_id"],
                        )
                    link_counts = ne._link_same_work_as(k)
                except Exception as e:
                    _mark(conn, k, "neo4j_err")
                    stats["neo4j_err"] += 1
                    logger.warning("neo4j error on %s: %s", k, e)
                    continue

                total_links = (link_counts.get("doi_links", 0)
                               + link_counts.get("arxiv_links", 0))
                stats["same_work_links"] += total_links
                _mark(conn, k, "done", ids["doi"], ids["arxiv_id"],
                      doi_links=link_counts.get("doi_links", 0),
                      arxiv_links=link_counts.get("arxiv_links", 0))
                stats["done"] += 1

            # Throttle between batches (be polite to Zotero API)
            time.sleep(0.3)
            if (bi // BATCH) % 10 == 0:
                elapsed = time.time() - t0
                logger.info("progress: %d/%d papers (%.0fs elapsed, stats=%s)",
                            min(bi + BATCH, len(all_keys)), len(all_keys),
                            elapsed, stats)
    finally:
        ne.close()
        conn.close()

    elapsed = time.time() - t0
    logger.info("DONE in %.0fs. stats=%s", elapsed, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N papers (smoke)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip papers already marked done/zotero_missing/no_ids")
    ap.add_argument("--retry-missing", action="store_true",
                    help="Retry papers previously marked zotero_missing")
    ap.add_argument("--retry-noid", action="store_true",
                    help="Retry papers previously marked no_ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan, no writes")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

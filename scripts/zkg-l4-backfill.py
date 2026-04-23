#!/usr/bin/env python3
"""T5 L4 one-shot backfill: compute title_hash for every existing :Paper,
write it back, then trigger _link_same_work_as to create SAME_WORK_AS edges
for any now-matching pairs.

Idempotent: SET p.title_hash overwrites; second run is a no-op in effect.
No checkpoint table by design (reviewer call-out): 4k papers × single-threaded
≈ 5-10 min; if the script dies, re-run.

Safe to run against live production: writes are all single-statement MERGE/SET
per paper, no table-level locks.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Load ~/.claude/.env.local so NEO4J_ZOTERO_* vars are available.
for line in Path(os.path.expanduser("~/.claude/.env.local")).read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Allow `from zotero_mcp...` import from the src/ layout.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zotero_mcp.kg_store import Neo4jWriter, compute_title_hash  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("l4_backfill")


def fetch_papers_with_authors(sess) -> list[tuple]:
    """Return [(pid, title, year, first_author_last), ...] in a SINGLE
    Cypher round-trip. Earlier version did 4353 sequential sess.run() calls
    for first_author_last which hung after ~5s under a write-transaction
    pool contention — now we fetch everything up front."""
    out = []
    q = """
    MATCH (p:Paper)
    OPTIONAL MATCH (a:Author)-[r:AUTHORED]->(p)
    WITH p, a, r
    ORDER BY coalesce(r.order, 9999) ASC
    WITH p, head(collect(coalesce(a.name, a.canonical_name))) AS first_author_last
    RETURN p.id AS pid, p.title AS title, p.year AS year, first_author_last
    """
    for rec in sess.run(q):
        out.append((rec["pid"], rec["title"], rec["year"],
                    rec["first_author_last"]))
    return out


def batch_write_title_hashes(sess, updates: list[dict]) -> None:
    """Single UNWIND write for a batch of {pid, title_hash} pairs. Replaces
    4353 individual SET statements with ~9 round-trips at batch=500."""
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (p:Source {id: row.pid})
        SET p.title_hash = row.title_hash
        """,
        rows=updates,
    )


def main():
    print(f"[{time.strftime('%H:%M:%S')}] backfill starting", flush=True)
    writer = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ["NEO4J_ZOTERO_USER"],
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] Neo4j driver ok", flush=True)

    t0 = time.time()
    stats = {"seen": 0, "hashed": 0, "skipped_weak": 0,
             "doi_links": 0, "arxiv_links": 0, "title_hash_links": 0}

    print(f"[{time.strftime('%H:%M:%S')}] fetching all papers + first-authors (single query)...", flush=True)
    with writer.driver.session() as sess:
        papers = fetch_papers_with_authors(sess)
    print(f"[{time.strftime('%H:%M:%S')}] fetched {len(papers)} papers", flush=True)

    # Phase 1: compute hashes in Python, then batch-UNWIND write.
    # Old: 4353 sequential sess.run() calls — hung on write-tx pool under load.
    # New: 1 fetch + ~9 batch writes (batch=500) = ~10 Neo4j round-trips total.
    updates = []
    for pid, title, year, author_last in papers:
        stats["seen"] += 1
        h = compute_title_hash(title, author_last, year)
        if h is None:
            stats["skipped_weak"] += 1
        else:
            stats["hashed"] += 1
        updates.append({"pid": pid, "title_hash": h})

    print(f"[{time.strftime('%H:%M:%S')}] computed hashes: hashed={stats['hashed']} weak={stats['skipped_weak']}", flush=True)
    BATCH = 500
    with writer.driver.session() as sess:
        for i in range(0, len(updates), BATCH):
            batch = updates[i:i + BATCH]
            batch_write_title_hashes(sess, batch)
            print(f"[{time.strftime('%H:%M:%S')}] phase 1 wrote {i + len(batch)}/{len(updates)}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] phase 1 done in {time.time() - t0:.1f}s", flush=True)

    # Phase 2: re-run _link_same_work_as for every paper so title_hash-based
    # edges get MERGE'd (also updates kind_sources on existing doi/arxiv edges
    # if the same pair matches multiple ways — ON MATCH apoc.coll.toSet append).
    t1 = time.time()
    for i, (pid, _title, _year, _al) in enumerate(papers, 1):
        try:
            link_counts = writer._link_same_work_as(pid)
            for k, v in link_counts.items():
                stats[k] += v
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] link failed for {pid}: {e}", flush=True)
        if i % 200 == 0:
            rate = i / max(time.time() - t1, 0.01)
            print(f"[{time.strftime('%H:%M:%S')}] phase 2 {i}/{len(papers)} @ {rate:.0f}/s  dl={stats['doi_links']} al={stats['arxiv_links']} thl={stats['title_hash_links']}", flush=True)

    writer.close()
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {time.time() - t0:.1f}s stats={stats}", flush=True)


if __name__ == "__main__":
    main()

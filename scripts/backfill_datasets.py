#!/usr/bin/env python3
"""T8: backfill benchmark names (ImageNet, MNIST, KITTI, etc.) that kimi
text-extract chain missed. Updates SQLite papers.extracted_json + new
paper_datasets_backfill table + Neo4j :Dataset + :EVALUATES_ON edges.

Spec: plans/zotero-kg-t4-t8-spec.md §2 (v2, subagent-reviewed).

Key design points (subagent findings #7, #8, #10, #11):
  - Anchored alias regex so ``^mnist`` doesn't false-hit ``fashionmnist``.
  - ``GLUE`` regex fixed (previous (?!\\s*benchmark) lookahead was reversed).
  - Neo4j canonical_name uses ``Neo4jWriter._canon`` (SPACE-preserving) so
    T8 writes hit the SAME Dataset nodes ingest already created — no
    split "pascal voc" vs "pascalvoc" nodes.
  - New SQLite table ``paper_datasets_backfill`` records what T8 added,
    so future ingest re-runs can (optionally) merge these back before
    INSERT OR REPLACE overwrites extracted_json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env.local"), override=False)
except ImportError:
    pass

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zotero_mcp.kg_store import Neo4jWriter  # noqa: E402

logger = logging.getLogger("t8_backfill")
logging.basicConfig(
    level=logging.INFO, force=True,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

KG_DB = Path(os.getenv(
    "ZOTERO_KG_DB",
    os.path.expanduser("~/.cache/zotero-mcp/kg.sqlite"),
))


# ---------------------------------------------------------------------------
# Canonical benchmark list (subagent #7, #8, #9 fixes on alias regex)
# ---------------------------------------------------------------------------
# Each tuple: (display_name, md_regex, alias_regex_against_collapsed_existing)
CANONICAL_BENCHMARKS: list[tuple[str, str, str]] = [
    ("CIFAR-10",       r"\bCIFAR-?10\b(?!\d)",   r"^cifar10$"),
    ("CIFAR-100",      r"\bCIFAR-?100\b",        r"^cifar100$"),
    # MNIST alias must NOT hit fashionmnist/mnistfor variants
    # MNIST alias: '^mnist' — anchor start. 'fashionmnist' doesn't start
    # with mnist (starts with 'fashion') → won't match → MNIST still added
    # for md hits. 'mnistforneuralnetworkillustration' starts with mnist
    # → matches → MNIST correctly deduped. Simpler + correct.
    ("MNIST",          r"\bMNIST\b",             r"^mnist"),
    ("Fashion-MNIST",  r"\bFashion-?MNIST\b",    r"^fashion-?mnist"),
    ("ImageNet",       r"\bImageNet\b",          r"^imagenet|^ilsvrc"),
    ("COCO",           r"\bMS-?COCO\b|\bCOCO\b", r"^(?:ms)?coco$|^coco20"),
    ("Pascal VOC",     r"\bPascal\s*VOC\b",      r"^pascal(?:voc)?|^voc20"),
    # KITTI alias matches ^kitti$ / variants but NOT kitti-mots (different benchmark)
    ("KITTI",          r"\bKITTI\b",
        r"^kitti$|^kittiodometry|^kittitracking|^kittiraw"),
    ("LibriSpeech",    r"\bLibriSpeech\b",       r"^librispeech"),
    ("SQuAD",          r"\bSQuAD\b",             r"^squad"),
    ("GLUE",           r"\bGLUE\b",              r"^glue$"),  # #7: removed faulty (?!\s*benchmark)
    ("nuScenes",       r"\bnuScenes\b",          r"^nuscenes"),
    ("Cityscapes",     r"\bCityscapes\b",        r"^cityscapes"),
    ("ADE20K",         r"\bADE20K\b",            r"^ade20k"),
    ("Waymo Open",     r"\bWaymo\s*Open\s*Dataset\b", r"^waymo"),
    ("DeepMIMO",       r"\bDeepMIMO\b",          r"^deepmimo"),
    ("QuaDRiGa",       r"\bQuaDRiGa\b",          r"^quadriga"),
]


def _collapse(s: str) -> str:
    """Strip non-word chars + lowercase, for alias matching against
    LLM-extracted dataset names like ``imagenetilsvrc2012``."""
    return re.sub(r"[^\w]+", "", (s or "").lower())


def detect_new_benchmarks(md_text: str,
                           existing: list[str]) -> list[str]:
    """Return list of display_name for benchmarks mentioned in md_text
    but NOT already covered (via alias) in existing dataset list."""
    existing_collapsed = [_collapse(e) for e in existing]
    newly_added: list[str] = []
    for display, md_re, alias_re in CANONICAL_BENCHMARKS:
        if not re.search(md_re, md_text, re.IGNORECASE):
            continue
        # Is any existing name matching the alias pattern? If yes → skip.
        if any(re.search(alias_re, ec) for ec in existing_collapsed):
            continue
        newly_added.append(display)
        existing_collapsed.append(_collapse(display))   # prevent duplicate in same paper
    return newly_added


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def _ensure_backfill_table(conn: sqlite3.Connection) -> None:
    """Create paper_datasets_backfill if missing (subagent #11 fix).
    Records what T8 added so future re-ingest can merge."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_datasets_backfill (
            paper_id         TEXT NOT NULL,
            dataset_display  TEXT NOT NULL,
            dataset_canon    TEXT NOT NULL,
            source           TEXT NOT NULL,
            added_at         REAL NOT NULL,
            PRIMARY KEY (paper_id, dataset_canon)
        );
        CREATE INDEX IF NOT EXISTS idx_backfill_paper ON paper_datasets_backfill(paper_id);
    """)
    conn.commit()


def _record_backfill(conn, paper_id, display, canon, source="regex_backfill"):
    conn.execute(
        """INSERT OR REPLACE INTO paper_datasets_backfill
           (paper_id, dataset_display, dataset_canon, source, added_at)
           VALUES (?,?,?,?,?)""",
        (paper_id, display, canon, source, time.time()))


def _update_extracted_json(conn, paper_id, new_display_names):
    row = conn.execute(
        "SELECT extracted_json FROM papers WHERE paper_id=?", (paper_id,)
    ).fetchone()
    if not row:
        return
    try:
        j = json.loads(row[0])
    except Exception:
        return
    ds = j.get("datasets") or []
    ds.extend(new_display_names)
    j["datasets"] = ds
    conn.execute("UPDATE papers SET extracted_json=? WHERE paper_id=?",
                  (json.dumps(j, ensure_ascii=False), paper_id))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run(args) -> None:
    conn = sqlite3.connect(str(KG_DB))
    _ensure_backfill_table(conn)

    # Pull all papers with md_text long enough
    q = ("SELECT paper_id, md_text, extracted_json FROM papers "
         "WHERE extracted_json IS NOT NULL AND md_chars >= 500")
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    rows = list(conn.execute(q))
    logger.info("scanning %d papers", len(rows))

    ne = None
    canon_fn = None
    if not args.dry_run:
        ne = Neo4jWriter(
            os.environ["NEO4J_ZOTERO_URI"],
            os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
            os.environ["NEO4J_ZOTERO_PASSWORD"],
        )
        # subagent #10 fix: reuse ingest's canonical function to avoid
        # split Dataset nodes
        canon_fn = Neo4jWriter._canon

    stats = {"scanned": 0, "patched": 0, "total_adds": 0,
             "dry_run_samples_added": []}
    t0 = time.time()
    for pid, md_text, extracted_json in rows:
        stats["scanned"] += 1
        try:
            existing_datasets = (json.loads(extracted_json) or {}).get("datasets", [])
        except Exception:
            existing_datasets = []
        new_names = detect_new_benchmarks(md_text or "", existing_datasets)
        if not new_names:
            continue

        stats["patched"] += 1
        stats["total_adds"] += len(new_names)
        if args.dry_run:
            if len(stats["dry_run_samples_added"]) < 15:
                stats["dry_run_samples_added"].append(
                    (pid, existing_datasets[:5], new_names))
            continue

        # SQLite: append to extracted_json + record in backfill table
        with conn:
            _update_extracted_json(conn, pid, new_names)
            for display in new_names:
                _record_backfill(conn, pid, display, _collapse(display))

        # Neo4j: MERGE Dataset + EVALUATES_ON edge
        try:
            with ne.driver.session() as s:
                for display in new_names:
                    cn = canon_fn(display)  # SPACE-preserving canonical
                    s.run("""
                        MERGE (d:Dataset {canonical_name: $cn})
                          SET d.name = coalesce(d.name, $display)
                        WITH d
                        MATCH (p:Source:Paper {id: $pid})
                        MERGE (p)-[r:EVALUATES_ON]->(d)
                          ON CREATE SET r.source_tags = ['regex_backfill'],
                                        r.created_at = timestamp()
                          ON MATCH  SET r.source_tags = apoc.coll.toSet(
                                          coalesce(r.source_tags, []) + ['regex_backfill'])
                    """, cn=cn, display=display, pid=pid)
        except Exception as e:
            logger.warning("neo4j err on %s: %s", pid, e)

        if stats["patched"] % 50 == 0:
            logger.info("progress: %d scanned / %d patched / %d total adds",
                        stats["scanned"], stats["patched"], stats["total_adds"])

    if ne:
        ne.close()
    conn.close()

    logger.info("DONE in %.0fs.  scanned=%d, patched=%d, total_dataset_adds=%d",
                time.time() - t0, stats["scanned"], stats["patched"],
                stats["total_adds"])
    if args.dry_run:
        print("\n=== DRY-RUN sample of patches ===")
        for pid, existing, adds in stats["dry_run_samples_added"]:
            print(f"\n  {pid}")
            print(f"    existing datasets: {existing}")
            print(f"    would add:         {adds}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

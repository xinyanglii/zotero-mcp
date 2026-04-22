#!/usr/bin/env python3
"""T1-c CrossRef enrichment: resolve DOI for :Source:ExternalRef nodes via
CrossRef API and build SAME_WORK_AS edges.

See plans/zotero-kg-t1-spec.md §4 for semantics (failure state machine,
Jaccard thresholds with subset-rescue + short-title fallback, SQLite
progress table with resume support).

Usage:
    # dry run 100 to calibrate thresholds
    python scripts/crossref_enrich_refs.py --dry-run --limit 100

    # full run (defaults: 1 worker, accept >=0.5, ambig 0.25-0.5)
    python scripts/crossref_enrich_refs.py --resume

    # faster with 3 concurrent workers (polite pool X-Concurrency-Limit=3)
    python scripts/crossref_enrich_refs.py --resume --workers 3
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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env.local"), override=False)
except ImportError:
    pass

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from zotero_mcp.ingest import normalize_doi  # noqa: E402
from zotero_mcp.kg_store import Neo4jWriter  # noqa: E402

logger = logging.getLogger("t1_crossref")
logging.basicConfig(
    level=logging.INFO, force=True,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

PROGRESS_DB = Path(os.getenv(
    "T1_BACKFILL_DB",
    os.path.expanduser("~/.cache/zotero-mcp/t1_backfill.sqlite"),
))
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "lxymario@hotmail.com")
USER_AGENT = f"zotero-kg/1.0 ({CROSSREF_MAILTO})"
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Similarity (spec §4 with subset-rescue + short-title fallback)
# ---------------------------------------------------------------------------
def _tokens(s: str) -> list[str]:
    return re.sub(r"[^\w]+", " ", s.lower()).split()


def title_similar(a: str, b: str) -> float:
    """Return 0..1 title similarity tuned for short / extended-title robustness.

    Rules:
      - If either's token set is a subset of the other → 1.0 (扩写版/前缀保护)
      - If shorter is < 4 tokens → token-set (k=1) Jaccard
      - Otherwise → 3-shingle Jaccard
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    sa_tok, sb_tok = set(ta), set(tb)
    if sa_tok <= sb_tok or sb_tok <= sa_tok:
        return 1.0
    if len(ta) < 4 or len(tb) < 4:
        return len(sa_tok & sb_tok) / len(sa_tok | sb_tok)

    def shingles(t, k=3):
        return {tuple(t[i:i + k]) for i in range(len(t) - k + 1)}
    sa, sb = shingles(ta), shingles(tb)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Progress SQLite — separate table sharing same DB as T1-b
# ---------------------------------------------------------------------------
def _init_progress_db() -> sqlite3.Connection:
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROGRESS_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS crossref_progress (
            ref_id          TEXT PRIMARY KEY,
            status          TEXT NOT NULL,  -- done / no_match / pending_review
                                            -- / retry / permanent_fail / neo4j_err
            crossref_doi    TEXT,
            crossref_score  REAL,
            title_sim       REAL,
            crossref_title  TEXT,
            attempts        INTEGER NOT NULL DEFAULT 0,
            last_err        TEXT,
            updated_at      REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cr_status ON crossref_progress(status);
    """)
    conn.commit()
    return conn


_db_lock = Lock()


def _mark(conn, ref_id, status, *, doi=None, score=None, sim=None,
          cr_title=None, attempts=0, last_err=None):
    with _db_lock:
        with conn:
            conn.execute(
                """INSERT INTO crossref_progress
                   (ref_id, status, crossref_doi, crossref_score, title_sim,
                    crossref_title, attempts, last_err, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(ref_id) DO UPDATE SET
                       status=excluded.status,
                       crossref_doi=excluded.crossref_doi,
                       crossref_score=excluded.crossref_score,
                       title_sim=excluded.title_sim,
                       crossref_title=excluded.crossref_title,
                       attempts=excluded.attempts,
                       last_err=excluded.last_err,
                       updated_at=excluded.updated_at""",
                (ref_id, status, doi, score, sim, cr_title,
                 attempts, last_err, time.time()),
            )


def _get_attempts(conn, ref_id) -> int:
    with _db_lock:
        row = conn.execute(
            "SELECT attempts FROM crossref_progress WHERE ref_id=?",
            (ref_id,)).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# CrossRef client (with polite rate limit + token bucket)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple token bucket for the CrossRef 10 req/s polite pool. Shared
    across workers — each call blocks until a slot is free."""

    def __init__(self, rate_per_sec: int = 10):
        self.rate = rate_per_sec
        self._lock = Lock()
        self._times: list[float] = []

    def acquire(self):
        with self._lock:
            now = time.time()
            # drop requests older than 1s
            self._times = [t for t in self._times if now - t < 1.0]
            if len(self._times) >= self.rate:
                sleep_s = 1.0 - (now - self._times[0]) + 0.01
                if sleep_s > 0:
                    time.sleep(sleep_s)
                now = time.time()
                self._times = [t for t in self._times if now - t < 1.0]
            self._times.append(now)


_limiter: RateLimiter | None = None


def _extract_first_author_lastname(author_year: str) -> str | None:
    """Extract first author's last name from '<First Last> <Year>' /
    '<Last, First> (Year)' / 'Smith et al. 2020' kinds of strings."""
    if not author_year:
        return None
    s = author_year.strip()
    # If "Last, First" format
    if "," in s:
        candidate = s.split(",", 1)[0].strip()
        if candidate and len(candidate.split()) <= 3:
            return candidate
    # "Smith et al. 2020" → "Smith"
    m = re.match(r"([A-Z][a-zA-Z-]+)", s)
    return m.group(1) if m else None


def crossref_query(title: str, author_year: str = "",
                   timeout: int = 30, rows: int = 3) -> list[dict]:
    """Execute a CrossRef query. Returns list of result items (may be empty).
    Raises urllib.error.HTTPError / URLError / TimeoutError on failure."""
    _limiter.acquire()
    params = [
        ("query.title", title),
        ("rows", str(rows)),
        ("mailto", CROSSREF_MAILTO),
    ]
    author = _extract_first_author_lastname(author_year)
    if author:
        params.append(("query.author", author))
    qs = urllib.parse.urlencode(params)
    url = f"https://api.crossref.org/works?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("message", {}).get("items", [])


def classify(title: str, author_year: str, conn, ref_id,
             accept_thresh: float, ambig_thresh: float) -> dict:
    """Run one enrichment attempt. Returns dict with status + details.
    Side-effects: writes to progress DB via _mark."""
    prior_attempts = _get_attempts(conn, ref_id)
    try:
        items = crossref_query(title, author_year)
    except urllib.error.HTTPError as e:
        attempts = prior_attempts + 1
        status = "permanent_fail" if attempts >= MAX_ATTEMPTS else "retry"
        _mark(conn, ref_id, status, attempts=attempts,
              last_err=f"HTTP {e.code}")
        return {"status": status, "error": f"HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        attempts = prior_attempts + 1
        status = "permanent_fail" if attempts >= MAX_ATTEMPTS else "retry"
        _mark(conn, ref_id, status, attempts=attempts,
              last_err=f"{type(e).__name__}: {e}")
        return {"status": status, "error": str(e)}

    if not items:
        _mark(conn, ref_id, "no_match", attempts=prior_attempts + 1)
        return {"status": "no_match"}

    best = items[0]  # CrossRef already returns score-sorted
    cr_title = (best.get("title") or [""])[0]
    cr_doi = normalize_doi(best.get("DOI", ""))
    sim = title_similar(title, cr_title)
    score = float(best.get("score", 0.0))

    if sim >= accept_thresh and cr_doi:
        return {"status": "done", "doi": cr_doi, "score": score, "sim": sim,
                "cr_title": cr_title}
    elif sim >= ambig_thresh and cr_doi:
        _mark(conn, ref_id, "pending_review",
              doi=cr_doi, score=score, sim=sim, cr_title=cr_title,
              attempts=prior_attempts + 1)
        return {"status": "pending_review", "doi": cr_doi, "sim": sim,
                "cr_title": cr_title}
    else:
        _mark(conn, ref_id, "no_match",
              doi=cr_doi, score=score, sim=sim, cr_title=cr_title,
              attempts=prior_attempts + 1)
        return {"status": "no_match", "sim": sim, "cr_title": cr_title}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def fetch_refs(limit: int | None, skip_done: set[str]) -> list[tuple[str, str, str]]:
    """Pull ExternalRef (id, title, author_year) from Neo4j, excluding
    those already processed by prior runs."""
    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    try:
        with ne.driver.session() as s:
            q = """
                MATCH (e:Source:ExternalRef)
                WHERE e.doi IS NULL AND size(e.title) >= 10
                RETURN e.id AS id, e.title AS title,
                       coalesce(e.author_year, '') AS author_year
                ORDER BY e.title
            """
            if limit:
                q += f" LIMIT {int(limit)}"
            refs = [(r["id"], r["title"], r["author_year"])
                    for r in s.run(q).data()]
    finally:
        ne.close()
    return [r for r in refs if r[0] not in skip_done]


def process_one(ne, conn, ref, accept_thresh, ambig_thresh, dry_run):
    """Worker body: classify + write Neo4j on 'done'."""
    ref_id, title, author_year = ref
    result = classify(title, author_year, conn, ref_id,
                      accept_thresh, ambig_thresh)
    if result["status"] == "done" and not dry_run:
        # Write Neo4j: SET doi + crossref_title on ExternalRef; run
        # _link_same_work_as_on_ref to MERGE SAME_WORK_AS with any Paper
        try:
            with ne.driver.session() as s:
                s.run(
                    """MATCH (e:Source:ExternalRef {id: $rid})
                       SET e.doi = $doi,
                           e.crossref_title = $ct,
                           e.crossref_score = $score,
                           e.crossref_matched_at = timestamp()""",
                    rid=ref_id, doi=result["doi"], ct=result["cr_title"],
                    score=result["score"],
                )
                # Link any Paper sharing this DOI to the ref
                s.run(
                    """MATCH (e:Source:ExternalRef {id: $rid})
                       WHERE e.doi IS NOT NULL
                       MATCH (p:Source:Paper {doi: e.doi})
                       MERGE (e)-[r:SAME_WORK_AS]->(p)
                         ON CREATE SET r.kind_sources = ['doi'],
                                       r.kind = 'preprint-published-ref',
                                       r.created_at = timestamp()
                         ON MATCH  SET r.kind_sources = apoc.coll.toSet(
                                         coalesce(r.kind_sources, []) + ['doi'])""",
                    rid=ref_id,
                )
            _mark(conn, ref_id, "done",
                  doi=result["doi"], score=result["score"],
                  sim=result["sim"], cr_title=result["cr_title"],
                  attempts=_get_attempts(conn, ref_id) + 1)
        except Exception as e:
            _mark(conn, ref_id, "neo4j_err",
                  attempts=_get_attempts(conn, ref_id) + 1,
                  last_err=f"{type(e).__name__}: {e}")
            result["status"] = "neo4j_err"
    return result


def run(args) -> None:
    global _limiter
    _limiter = RateLimiter(rate_per_sec=10)
    conn = _init_progress_db()

    # Determine skip set
    if args.resume:
        with _db_lock:
            skip_statuses = ("done", "permanent_fail")
            if not args.retry_nomatch:
                skip_statuses += ("no_match",)
            q = ("SELECT ref_id FROM crossref_progress WHERE status IN ("
                 + ",".join("?" * len(skip_statuses)) + ")")
            skip_done = {r[0] for r in conn.execute(q, skip_statuses).fetchall()}
    else:
        skip_done = set()

    refs = fetch_refs(args.limit, skip_done)
    logger.info("ExternalRefs to process: %d (resume-skip=%d)",
                len(refs), len(skip_done))

    if not refs:
        logger.info("nothing to do")
        return

    if args.dry_run:
        logger.info("DRY-RUN — first 20 sample classifications (no Neo4j write):")

    # Shared Neo4j writer for all workers
    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )

    stats = {"done": 0, "no_match": 0, "pending_review": 0, "retry": 0,
             "permanent_fail": 0, "neo4j_err": 0}
    samples = []
    t0 = time.time()

    try:
        if args.workers <= 1:
            for i, ref in enumerate(refs, 1):
                res = process_one(ne, conn, ref,
                                  args.jaccard_accept, args.jaccard_ambig,
                                  args.dry_run)
                stats[res["status"]] = stats.get(res["status"], 0) + 1
                if args.dry_run and len(samples) < 20:
                    samples.append((ref[0], ref[1], res))
                if i % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info("%d/%d @ %.0fs stats=%s",
                                i, len(refs), elapsed, stats)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(
                    process_one, ne, conn, r,
                    args.jaccard_accept, args.jaccard_ambig, args.dry_run): r
                    for r in refs}
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    stats[res["status"]] = stats.get(res["status"], 0) + 1
                    if args.dry_run and len(samples) < 20:
                        samples.append((futs[fut][0], futs[fut][1], res))
                    if i % 100 == 0:
                        elapsed = time.time() - t0
                        logger.info("%d/%d @ %.0fs stats=%s",
                                    i, len(refs), elapsed, stats)
    finally:
        ne.close()
        conn.close()

    elapsed = time.time() - t0
    logger.info("DONE in %.0fs. stats=%s", elapsed, stats)

    if args.dry_run:
        print("\n=== DRY-RUN sample classifications ===")
        for rid, title, res in samples:
            print(f"\n  ref_id={rid}")
            print(f"  orig title: {title[:90]}")
            if "cr_title" in res:
                print(f"  cr   title: {res.get('cr_title','-')[:90]}")
            print(f"  status={res['status']} sim={res.get('sim','-')}")


# ---------------------------------------------------------------------------
# Pending-review triage CLI (Finding 6 — 904 rows in SQLite otherwise stranded)
# ---------------------------------------------------------------------------
def cmd_list_pending(args) -> None:
    """Print all pending_review rows with ref title + candidate CrossRef DOI + sim."""
    conn = _init_progress_db()
    rows = conn.execute(
        """SELECT ref_id, crossref_doi, crossref_score, title_sim, crossref_title
           FROM crossref_progress WHERE status='pending_review'
           ORDER BY title_sim DESC"""
    ).fetchall()
    conn.close()
    if not rows:
        print("(no pending_review rows)")
        return
    # Pull original ref titles from Neo4j for comparison
    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    try:
        with ne.driver.session() as s:
            id_to_title = {r["id"]: r["title"] for r in s.run(
                "MATCH (e:ExternalRef) WHERE e.id IN $ids RETURN e.id AS id, e.title AS title",
                ids=[r[0] for r in rows]
            ).data()}
    finally:
        ne.close()
    print(f"=== pending_review ({len(rows)} rows; sorted by title_sim DESC) ===")
    for ref_id, cr_doi, cr_score, sim, cr_title in rows:
        orig = (id_to_title.get(ref_id) or "?")[:100]
        cr   = (cr_title or "")[:100]
        print(f"\n  {ref_id}  sim={sim:.2f} cr_score={cr_score:.0f}")
        print(f"    ORIG: {orig}")
        print(f"    CR  : {cr}")
        print(f"    DOI : {cr_doi}")
    print(f"\nReview then: python {sys.argv[0]} --approve <ref_id>   or   --reject <ref_id>")


def cmd_approve(ref_id: str) -> None:
    """Commit the pending CrossRef DOI to Neo4j + build SAME_WORK_AS + mark done."""
    conn = _init_progress_db()
    row = conn.execute(
        "SELECT crossref_doi, crossref_title FROM crossref_progress "
        "WHERE ref_id=? AND status='pending_review'",
        (ref_id,),
    ).fetchone()
    if not row:
        print(f"(ref_id {ref_id} not in pending_review — nothing to do)")
        conn.close()
        return
    cr_doi, cr_title = row
    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    try:
        with ne.driver.session() as s:
            s.run(
                "MATCH (e:ExternalRef {id: $rid}) SET e.doi = $doi, e.crossref_title = $ct",
                rid=ref_id, doi=cr_doi, ct=cr_title,
            )
        ne._link_same_work_as(ref_id)
    finally:
        ne.close()
    _mark(conn, ref_id, "done",
          doi=cr_doi,
          attempts=_get_attempts(conn, ref_id) + 1,
          last_err="approved_manually")
    conn.close()
    print(f"approved {ref_id} → doi={cr_doi}")


def cmd_reject(ref_id: str) -> None:
    """Mark pending_review as no_match (no DOI set; ref stays candidate-free)."""
    conn = _init_progress_db()
    row = conn.execute(
        "SELECT status FROM crossref_progress WHERE ref_id=?", (ref_id,),
    ).fetchone()
    if not row or row[0] != "pending_review":
        print(f"(ref_id {ref_id} not pending_review; current status={row[0] if row else 'missing'})")
        conn.close()
        return
    _mark(conn, ref_id, "no_match",
          attempts=_get_attempts(conn, ref_id) + 1,
          last_err="rejected_manually")
    conn.close()
    print(f"rejected {ref_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N refs (smoke/calibration)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip refs previously marked done / no_match / permanent_fail")
    ap.add_argument("--retry-nomatch", action="store_true",
                    help="Retry refs previously marked no_match (e.g. after threshold change)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Classify but don't write Neo4j or SET status=done rows")
    ap.add_argument("--jaccard-accept", type=float, default=0.5,
                    help="Title similarity threshold for auto-accept (default 0.5)")
    ap.add_argument("--jaccard-ambig", type=float, default=0.25,
                    help="Title similarity threshold for pending_review (default 0.25)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent workers (max 3 per CrossRef polite pool limit)")
    # Pending-review triage commands (mutually exclusive with the main run)
    ap.add_argument("--list-pending", action="store_true",
                    help="Print all pending_review rows (no enrichment run)")
    ap.add_argument("--approve", metavar="REF_ID",
                    help="Accept pending candidate: write DOI + SAME_WORK_AS, mark done")
    ap.add_argument("--reject", metavar="REF_ID",
                    help="Reject pending candidate: mark no_match (no Neo4j change)")
    args = ap.parse_args()

    # Triage commands short-circuit; they don't invoke the main enrichment loop.
    if args.list_pending:
        return cmd_list_pending(args)
    if args.approve:
        return cmd_approve(args.approve)
    if args.reject:
        return cmd_reject(args.reject)

    if args.workers > 3:
        logger.warning("clamping workers to 3 (CrossRef X-Concurrency-Limit)")
        args.workers = 3
    run(args)


if __name__ == "__main__":
    main()

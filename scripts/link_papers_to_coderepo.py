#!/usr/bin/env python3
"""T4: scan papers.md_text for GitHub URLs, build :Source:CodeRepo nodes
+ (Paper)-[:IMPLEMENTED_BY]->(CodeRepo) edges.

Spec: plans/zotero-kg-t4-t8-spec.md (v2, subagent-reviewed).

Usage:
    python scripts/link_papers_to_coderepo.py [--dry-run] [--limit N]
                                               [--resume]

Idempotent via Cypher MERGE. Progress persisted at
~/.cache/zotero-mcp/t4_progress.sqlite.

Key design points (subagent findings #2, #3, #4, #6):
  - PRE-PASS: read existing :CodeRepo.url values → map to their IDs so
    T4 reuses T3's 12 Zotero-key-id software-item nodes instead of
    forking a parallel gh:<owner>/<repo> node.
  - GitHub reserved owners blacklist (orgs/apps/users/sponsors etc.)
    + owner-name validity regex ([a-z0-9][a-z0-9-]{0,38}).
  - Regex trailing-strip: .git/.md suffixes, sentence punctuation.
  - Edge source_tags MERGE-outside-pattern so future T4b extra/relation
    sources append to list instead of creating parallel edges.
"""
from __future__ import annotations

import argparse
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

logger = logging.getLogger("t4_link")
logging.basicConfig(
    level=logging.INFO, force=True,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

PROGRESS_DB = Path(os.getenv(
    "T4_PROGRESS_DB",
    os.path.expanduser("~/.cache/zotero-mcp/t4_progress.sqlite"),
))
KG_DB = Path(os.getenv(
    "ZOTERO_KG_DB",
    os.path.expanduser("~/.cache/zotero-mcp/kg.sqlite"),
))


# ---------------------------------------------------------------------------
# Regex + blacklist (subagent #3, #4)
# ---------------------------------------------------------------------------
GITHUB_URL_RE = re.compile(
    r'https?://github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
)
# GitHub username rule: 1-39 chars, [a-z0-9-], can't start OR end with `-`.
OWNER_VALID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?$")
GITHUB_RESERVED_OWNERS = frozenset({
    # product pages
    "solutions", "resources", "security", "features", "enterprise",
    "pricing", "events", "marketplace", "trending", "topics",
    "collections", "about", "team", "customer-stories", "premium-support",
    "contact", "legal", "privacy", "terms", "advisories",
    # account / routing reserved words
    "orgs", "apps", "sponsors", "users", "watching", "stars",
    "settings", "notifications", "codespaces", "account", "assets",
    "readme", "login", "signup", "site", "pulls", "issues",
    "wiki", "tree", "blob", "discussions", "new",
})


def clean_repo(r: str) -> str | None:
    """Strip trailing punctuation + common suffixes (subagent #4)."""
    r = re.sub(r'\.(git|md)$', '', r, flags=re.IGNORECASE)
    r = r.rstrip('.,);]')
    if not r or r.startswith('.') or r.startswith('-'):
        return None
    return r


def normalize_url_key(u: str) -> str:
    """Canonical url form used to join T3 existing nodes with T4 new ones."""
    return u.lower().rstrip("/").removesuffix(".git")


def extract_github_repos(md: str) -> set[tuple[str, str]]:
    """Extract unique (owner, repo) tuples from markdown. Applies blacklist
    + owner validity + trailing strip."""
    out: set[tuple[str, str]] = set()
    for m in GITHUB_URL_RE.finditer(md):
        owner = m.group(1).lower()
        repo = clean_repo(m.group(2).lower())
        if not repo:
            continue
        if owner in GITHUB_RESERVED_OWNERS:
            continue
        if not OWNER_VALID_RE.match(owner):
            continue
        out.add((owner, repo))
    return out


# ---------------------------------------------------------------------------
# Progress SQLite
# ---------------------------------------------------------------------------
def _init_progress_db() -> sqlite3.Connection:
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROGRESS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS t4_paper_progress (
            paper_id      TEXT PRIMARY KEY,
            n_repos       INTEGER NOT NULL,
            status        TEXT NOT NULL,       -- done / no_repos / neo4j_err
            updated_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS t4_repo_catalog (
            repo_id       TEXT PRIMARY KEY,     -- gh:owner/repo or Zotero key
            owner         TEXT,
            repo          TEXT,
            url           TEXT,
            origin        TEXT NOT NULL,        -- 'existing' / 'new'
            first_seen    REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Neo4j operations
# ---------------------------------------------------------------------------
def load_existing_coderepo_urls(ne: Neo4jWriter) -> dict[str, str]:
    """Subagent #2 fix: avoid double-node. Pre-pass to build
    ``{url_key: existing_id}`` from any :CodeRepo with url set (includes
    the 12 T3 Zotero-key-id software items + any prior T4 run).
    """
    mapping: dict[str, str] = {}
    with ne.driver.session() as s:
        rows = s.run("""
            MATCH (c:Source:CodeRepo)
            WHERE c.url IS NOT NULL
            RETURN c.url AS url, c.id AS id
        """).data()
    for r in rows:
        mapping[normalize_url_key(r["url"])] = r["id"]
    logger.info("pre-pass: %d existing :CodeRepo nodes indexed by url", len(mapping))
    return mapping


def merge_coderepo_and_edge(ne: Neo4jWriter, paper_id: str,
                              repo_id: str, owner: str, repo: str,
                              url: str, is_new_node: bool) -> None:
    """MERGE CodeRepo node (if new) + the Paper->CodeRepo edge.

    Edge source_tags uses MERGE-outside-pattern per subagent #6: same
    (paper, repo) pair always gets single edge; re-runs or future
    extra/relation sources APPEND to ``source_tags`` list.
    """
    with ne.driver.session() as s:
        if is_new_node:
            s.run("""
                MERGE (c:Source:CodeRepo {id: $rid})
                  ON CREATE SET c.url = $url,
                                c.owner = $owner,
                                c.repo = $repo,
                                c.discovered_via = 'md_scan',
                                c.created_at = timestamp()
            """, rid=repo_id, url=url, owner=owner, repo=repo)
        s.run("""
            MATCH (p:Source {id: $pid})
            MATCH (c:Source:CodeRepo {id: $rid})
            MERGE (p)-[r:IMPLEMENTED_BY]->(c)
              ON CREATE SET r.source_tags = ['md_scan'],
                            r.created_at = timestamp()
              ON MATCH  SET r.source_tags = apoc.coll.toSet(
                              coalesce(r.source_tags, []) + ['md_scan'])
        """, pid=paper_id, rid=repo_id)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run(args) -> None:
    conn = _init_progress_db()
    if args.resume:
        done = {r[0] for r in conn.execute(
            "SELECT paper_id FROM t4_paper_progress WHERE status='done'")}
    else:
        done = set()

    # Pull all papers with github.com in md_text
    kg = sqlite3.connect(str(KG_DB))
    q = ("SELECT paper_id, md_text FROM papers "
         "WHERE md_text LIKE '%github.com/%'")
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    rows = list(kg.execute(q))
    kg.close()
    rows = [r for r in rows if r[0] not in done]
    logger.info("%d papers to process (after resume-skip=%d)", len(rows), len(done))

    ne = None
    existing_url_to_id: dict[str, str] = {}
    if not args.dry_run:
        ne = Neo4jWriter(
            os.environ["NEO4J_ZOTERO_URI"],
            os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
            os.environ["NEO4J_ZOTERO_PASSWORD"],
        )
        existing_url_to_id = load_existing_coderepo_urls(ne)

    stats = {"done": 0, "no_repos": 0, "neo4j_err": 0,
             "edges_created": 0, "new_coderepo_nodes": 0,
             "reused_existing_nodes": 0}
    t0 = time.time()
    try:
        for pid, md in rows:
            repos = extract_github_repos(md)
            if not repos:
                conn.execute(
                    "INSERT OR REPLACE INTO t4_paper_progress "
                    "(paper_id, n_repos, status, updated_at) VALUES (?,?,?,?)",
                    (pid, 0, "no_repos", time.time()))
                conn.commit()
                stats["no_repos"] += 1
                continue

            if args.dry_run:
                stats["edges_created"] += len(repos)
                stats["done"] += 1
                if stats["done"] <= 20:
                    print(f"\n  {pid}: {len(repos)} repos")
                    for o, r in list(repos)[:3]:
                        print(f"    https://github.com/{o}/{r}")
                continue

            try:
                for owner, repo in repos:
                    url = f"https://github.com/{owner}/{repo}"
                    url_k = normalize_url_key(url)
                    if url_k in existing_url_to_id:
                        # reuse existing node
                        repo_id = existing_url_to_id[url_k]
                        stats["reused_existing_nodes"] += 1
                        merge_coderepo_and_edge(
                            ne, pid, repo_id, owner, repo, url,
                            is_new_node=False)
                    else:
                        repo_id = f"gh:{owner}/{repo}"
                        merge_coderepo_and_edge(
                            ne, pid, repo_id, owner, repo, url,
                            is_new_node=True)
                        existing_url_to_id[url_k] = repo_id
                        stats["new_coderepo_nodes"] += 1
                        conn.execute(
                            "INSERT OR REPLACE INTO t4_repo_catalog "
                            "(repo_id, owner, repo, url, origin, first_seen) "
                            "VALUES (?,?,?,?,?,?)",
                            (repo_id, owner, repo, url, "new", time.time()))
                    stats["edges_created"] += 1
                conn.execute(
                    "INSERT OR REPLACE INTO t4_paper_progress "
                    "(paper_id, n_repos, status, updated_at) VALUES (?,?,?,?)",
                    (pid, len(repos), "done", time.time()))
                conn.commit()
                stats["done"] += 1
            except Exception as e:
                conn.execute(
                    "INSERT OR REPLACE INTO t4_paper_progress "
                    "(paper_id, n_repos, status, updated_at) VALUES (?,?,?,?)",
                    (pid, len(repos), "neo4j_err", time.time()))
                conn.commit()
                stats["neo4j_err"] += 1
                logger.warning("neo4j err on %s: %s", pid, e)

            if stats["done"] % 50 == 0 and stats["done"]:
                logger.info("progress: %d done, stats=%s", stats["done"], stats)
    finally:
        if ne:
            ne.close()
        conn.close()

    logger.info("DONE in %.0fs.  stats=%s", time.time() - t0, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N papers (smoke / calibration)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip papers already status='done' in progress DB")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan, no Neo4j writes")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

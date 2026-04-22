"""
Storage writers for the academic KG pipeline (M3 ingest).

Three stores, kept behind small focused classes so each can be swapped/tested:
  - ``SQLiteStore``:  raw markdown + extracted JSON + cost log, provenance
    so we never re-pay for LLM extraction on re-ingest.
  - ``QdrantWriter``:  section-aware chunks with dense (Qwen v4) + BM25
    sparse vectors, upserted into ``zotero_library`` collection.
  - ``Neo4jWriter``:   MERGEs ``:Source`` / ``:Method`` / ``:Concept`` /
    ``:Dataset`` / ``:Author`` nodes and their relations.

All three are idempotent per ``paper_id`` — safe to re-run on the same paper.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .extractor import ExtractedPaper

logger = logging.getLogger(__name__)

# ======================================================================
# SQLiteStore — provenance + resume
# ======================================================================
class SQLiteStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS papers (
      paper_id TEXT PRIMARY KEY,
      item_type TEXT,
      title TEXT,
      year INTEGER,
      md_chars INTEGER,
      md_text TEXT,
      extracted_json TEXT,
      ingested_at REAL,
      mineru_secs REAL,
      llm_secs REAL,
      llm_input_tokens INTEGER,
      llm_output_tokens INTEGER
    );
    CREATE TABLE IF NOT EXISTS failures (
      paper_id TEXT,
      stage TEXT,
      err TEXT,
      ts REAL
    );
    CREATE INDEX IF NOT EXISTS idx_failures_paper ON failures(paper_id);
    CREATE TABLE IF NOT EXISTS figures (
      paper_id     TEXT    NOT NULL,
      figure_idx   INTEGER NOT NULL,
      mineru_name  TEXT    NOT NULL,
      content_sha  TEXT    NOT NULL,
      caption      TEXT,
      bytes_len    INTEGER NOT NULL,
      width        INTEGER,
      height       INTEGER,
      mime         TEXT    DEFAULT 'image/jpeg',
      webdav_path  TEXT,
      downscaled   INTEGER NOT NULL DEFAULT 0,
      created_at   REAL    NOT NULL,
      PRIMARY KEY (paper_id, figure_idx),
      UNIQUE (paper_id, mineru_name),
      FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    );
    CREATE INDEX IF NOT EXISTS idx_figures_content_sha ON figures(content_sha);
    CREATE INDEX IF NOT EXISTS idx_figures_pending_upload
      ON figures(paper_id) WHERE webdav_path IS NULL;
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the ThreadPoolExecutor workers can share the
        # connection; writes serialized via self._lock, so we stay consistent.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        # Idempotent: add columns introduced after initial schema
        self._ensure_column("papers", "extract_provider", "TEXT")
        self.conn.commit()
        self._lock = threading.Lock()

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        """ALTER TABLE ADD COLUMN if missing (no-op otherwise). SQLite < 3.35
        lacks IF NOT EXISTS for ADD COLUMN, so guard with pragma_table_info."""
        cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self):
        with self._lock:
            self.conn.close()

    def already_done(self, paper_id: str) -> bool:
        with self._lock:
            c = self.conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (paper_id,))
            return c.fetchone() is not None

    def save_paper(
        self,
        paper: ExtractedPaper,
        *,
        item_type: str,
        md_text: str,
        mineru_secs: float = 0.0,
        llm_secs: float = 0.0,
    ):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, item_type, title, year, md_chars, md_text,
                    extracted_json, ingested_at, mineru_secs, llm_secs,
                    llm_input_tokens, llm_output_tokens)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    paper.paper_id, item_type, paper.title, paper.year,
                    len(md_text), md_text, paper.model_dump_json(),
                    time.time(), mineru_secs, llm_secs, 0, 0,
                ),
            )
            self.conn.commit()

    def record_failure(self, paper_id: str, stage: str, err: str):
        with self._lock:
            self.conn.execute(
                "INSERT INTO failures(paper_id,stage,err,ts) VALUES (?,?,?,?)",
                (paper_id, stage, err[:2000], time.time()),
            )
            self.conn.commit()

    def save_paper_with_figures(
        self,
        paper: ExtractedPaper,
        *,
        item_type: str,
        md_text: str,
        figures: list[dict],
        extract_provider: str,
        mineru_secs: float = 0.0,
        llm_secs: float = 0.0,
    ) -> None:
        """Atomic write: figures + papers in one transaction.

        ``figures`` entries carry: mineru_name, content_sha, caption, bytes_len,
        width, height, mime, webdav_path (may be None if WebDAV upload skipped
        on oversize after recompression), downscaled.

        If anything raises, the ``with self.conn`` block rolls back — no half-
        written state. This is the ONLY write path T0's ingest should use when
        figures are being captured; ``save_paper`` stays for legacy / figure-
        less flows.
        """
        now = time.time()
        with self._lock:
            with self.conn:  # auto-commit on success, rollback on exception
                # Replace any prior figures for this paper (retry-safe).
                self.conn.execute(
                    "DELETE FROM figures WHERE paper_id=?", (paper.paper_id,))
                for i, f in enumerate(figures):
                    self.conn.execute(
                        """INSERT INTO figures
                           (paper_id, figure_idx, mineru_name, content_sha,
                            caption, bytes_len, width, height, mime,
                            webdav_path, downscaled, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            paper.paper_id, i,
                            f["mineru_name"], f["content_sha"],
                            f.get("caption"), f["bytes_len"],
                            f.get("width"), f.get("height"),
                            f.get("mime", "image/jpeg"),
                            f.get("webdav_path"),
                            1 if f.get("downscaled") else 0,
                            now,
                        ),
                    )
                self.conn.execute(
                    """INSERT OR REPLACE INTO papers
                       (paper_id, item_type, title, year, md_chars, md_text,
                        extracted_json, ingested_at, mineru_secs, llm_secs,
                        llm_input_tokens, llm_output_tokens, extract_provider)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        paper.paper_id, item_type, paper.title, paper.year,
                        len(md_text), md_text, paper.model_dump_json(),
                        now, mineru_secs, llm_secs, 0, 0,
                        extract_provider,
                    ),
                )

    def pending_webdav_figures(self, paper_id: str | None = None) -> list[dict]:
        """Return figures missing WebDAV upload (for later backfill)."""
        sql = ("SELECT paper_id, figure_idx, mineru_name FROM figures "
               "WHERE webdav_path IS NULL")
        params: tuple = ()
        if paper_id is not None:
            sql += " AND paper_id=?"
            params = (paper_id,)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [{"paper_id": r[0], "figure_idx": r[1], "mineru_name": r[2]}
                for r in rows]

    def mark_figure_uploaded(self, paper_id: str, mineru_name: str,
                              webdav_path: str) -> None:
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "UPDATE figures SET webdav_path=? "
                    "WHERE paper_id=? AND mineru_name=?",
                    (webdav_path, paper_id, mineru_name),
                )


# ======================================================================
# Chunking + embedding helpers
# ======================================================================
_H2_SPLIT = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Public — Match MinerU figure refs: ``![](images/<64-hex>.<ext>)``.
# Shared source of truth for both QdrantWriter.upsert_paper (chunks
# ``payload.figure_refs``) and ingest._build_figures (caption capture).
# ``mn`` group captures the **full filename including extension** — this
# is what SQLite ``figures.mineru_name`` stores and what the WebDAV path
# appends under ``<figures_root>/<paper_id>/``. Keeping the extension
# avoids guessing MIME downstream and lets us support mixed .jpg/.png.
IMG_REF_RE = re.compile(
    r'!\[[^\]]*\]\(images/(?P<mn>[a-f0-9]{64}\.\w{2,4})\)'
)
# Legacy alias retained for readability in QdrantWriter below.
_CHUNK_IMG_REF_RE = IMG_REF_RE


def section_chunks(markdown: str, *, max_chars: int = 6000) -> list[dict[str, Any]]:
    """Split markdown into section-aware chunks.

    Each chunk carries ``{section_title, chunk_idx, text}``. Long sections are
    further windowed by ``max_chars`` (rough ~2000 tokens).
    """
    lines = markdown.splitlines()
    sections: list[tuple[str, list[str]]] = [("", [])]
    for ln in lines:
        m = _H2_SPLIT.match(ln)
        if m and m.group(1) in ("#", "##"):
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(ln)

    chunks: list[dict[str, Any]] = []
    idx = 0
    for title, body in sections:
        text = "\n".join(body).strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append({"section_title": title, "chunk_idx": idx, "text": text})
            idx += 1
            continue
        # window
        step = max_chars - 500
        for start in range(0, len(text), step):
            end = start + max_chars
            chunks.append({
                "section_title": title,
                "chunk_idx": idx,
                "text": text[start:end],
            })
            idx += 1
    return chunks


# ======================================================================
# QdrantWriter — dense + BM25 sparse upsert
# ======================================================================
class QdrantWriter:
    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        dashscope_api_key: str,
        embed_model: str | None = None,
        embed_dim: int | None = None,
        embed_base_url: str | None = None,
    ):
        # qdrant-client's internal `requests`/`httpx` pool had repeated mid-
        # stream `IncompleteRead`/`Connection refused` against the VPS 1.17.1
        # instance (raw HTTP PUT w/ same payload worked fine — so it's the
        # lib). Use raw urllib directly for upsert to sidestep it. Client kept
        # around only for non-hot paths.
        from qdrant_client import QdrantClient
        self.qc = QdrantClient(host=host, port=port, timeout=60)
        self._qdrant_base = f"http://{host}:{port}"
        self.collection = collection
        self.dashscope_key = dashscope_api_key
        # env-driven defaults so the client stays portable (new provider /
        # self-hosted endpoint / different embedding model only changes env).
        self.embed_model = embed_model or os.getenv(
            "EMBED_MODEL", "text-embedding-v4")
        self.embed_dim = embed_dim or int(os.getenv("EMBED_DIM", "2048"))
        self.embed_base_url = (embed_base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        # lazy BM25
        self._bm25 = None

    def _bm25_model(self):
        if self._bm25 is None:
            from fastembed import SparseTextEmbedding
            self._bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._bm25

    def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        """DashScope text-embedding-v4; batches up to 10."""
        import urllib.request
        import urllib.error
        out: list[list[float]] = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            payload = json.dumps({
                "model": self.embed_model,
                "input": batch,
                "dimensions": self.embed_dim,
            }).encode()
            req = urllib.request.Request(
                f"{self.embed_base_url}/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.dashscope_key}",
                    "Content-Type": "application/json",
                },
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        resp = json.loads(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            for d in resp["data"]:
                out.append(d["embedding"])
        return out

    def upsert_paper(
        self,
        paper: ExtractedPaper,
        *,
        item_type: str,
        markdown: str,
    ) -> int:
        """Chunk markdown, embed (dense + sparse), upsert to Qdrant. Returns chunk count.

        Each chunk's payload carries a ``figure_refs`` list of MinerU
        filenames (``<64hex>.jpg``) that appear as ``![](images/<mn>)``
        inside the chunk text, so multimodal retrieval can cross-join to
        SQLite ``figures`` / WebDAV ``/dav/zotero-kg-figures/``. Empty list
        when no figure refs present — caller reads with
        ``payload.get("figure_refs", [])`` for old-chunk compatibility.
        """
        chunks = section_chunks(markdown)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        dense = self._embed_dense(texts)
        sparse_embeds = list(self._bm25_model().embed(texts))

        points = []
        for c, d_vec, s_vec in zip(chunks, dense, sparse_embeds):
            pid = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{paper.paper_id}#{c['chunk_idx']}",
            ))
            # Collect MinerU figure names referenced in this chunk. dedup
            # but preserve order for downstream consumers.
            seen: set[str] = set()
            figure_refs: list[str] = []
            for m in _CHUNK_IMG_REF_RE.finditer(c["text"]):
                name = m.group("mn")
                if name not in seen:
                    seen.add(name)
                    figure_refs.append(name)
            points.append(dict(
                id=pid,
                vector={
                    "dense": list(d_vec),
                    "bm25": {
                        "indices": s_vec.indices.tolist(),
                        "values":  s_vec.values.tolist(),
                    },
                },
                payload={
                    "paper_id": paper.paper_id,
                    "item_type": item_type,
                    "title": paper.title,
                    "year": paper.year or 0,
                    "section_title": c["section_title"],
                    "chunk_idx": c["chunk_idx"],
                    "text": c["text"],
                    "figure_refs": figure_refs,
                },
            ))
        # Raw HTTP upsert in batches of 50 with wait=false. qdrant-client 1.12
        # against Qdrant server 1.17.1 was truncating responses mid-stream
        # (IncompleteRead), raw HTTP from the same process works fine — so
        # we bypass the lib for the hot write path. Exp backoff on transient
        # errors; raise on final failure so ingest can record it for later
        # --retry-failed.
        import urllib.request, urllib.error
        url = f"{self._qdrant_base}/collections/{self.collection}/points?wait=false"

        def _put(batch):
            body = json.dumps({"points": batch}).encode()
            last_exc: Exception | None = None
            for attempt, delay in enumerate([0, 2, 8, 30], start=1):
                if delay:
                    time.sleep(delay)
                try:
                    req = urllib.request.Request(url, data=body, method="PUT",
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = json.loads(r.read())
                    if data.get("status") != "ok":
                        raise RuntimeError(f"qdrant reported status={data.get('status')}")
                    if attempt > 1:
                        logger.info("qdrant upsert %s batch(%d) succeeded on attempt %d",
                                    paper.paper_id, len(batch), attempt)
                    return
                except Exception as e:
                    last_exc = e
                    logger.warning("qdrant upsert %s batch(%d) attempt %d failed: %s",
                                   paper.paper_id, len(batch), attempt, e)
            assert last_exc is not None
            raise last_exc

        for i in range(0, len(points), 50):
            _put(points[i:i+50])
        return len(points)


# ======================================================================
# Neo4jWriter — MERGE nodes + relations
# ======================================================================
# ======================================================================
# T3 · itemType → Neo4j label hierarchy
# ======================================================================
# Wiki §4.2 enumerates allowed :Source sub-labels:
#     :Paper | :Book | :Thesis | :Webpage | :CodeRepo | :ExternalRef
# For v1 we only branch to :Paper, :Webpage, :CodeRepo. :Book/:Thesis kept
# as :Paper to avoid re-labeling the already-ingested 4276 nodes.
ITEM_TYPE_TO_LABELS: dict[str, tuple[str, str]] = {
    # paper family — unchanged behavior
    "journalArticle":     ("Source", "Paper"),
    "preprint":           ("Source", "Paper"),
    "conferencePaper":    ("Source", "Paper"),
    "thesis":             ("Source", "Paper"),
    "bookSection":        ("Source", "Paper"),
    "book":               ("Source", "Paper"),
    "report":             ("Source", "Paper"),
    # T3 new
    "webpage":            ("Source", "Webpage"),
    "blogPost":           ("Source", "Webpage"),
    "encyclopediaArticle": ("Source", "Webpage"),
    "forumPost":          ("Source", "Webpage"),
    "computerProgram":    ("Source", "CodeRepo"),
    "software":           ("Source", "CodeRepo"),
}


def _label_hierarchy(item_type: str) -> tuple[str, str]:
    """Return (``Source``, <sub_label>) for a Zotero itemType.

    Unknown types (including accidentally routed ``attachment`` / ``note``
    or any new itemType added to Zotero) → logged warning + returned as
    ``('Source', 'Unknown')`` so the node is easy to find and fix, not
    silently merged into ``:Paper``.
    """
    out = ITEM_TYPE_TO_LABELS.get(item_type)
    if out is None:
        logger.warning("unknown itemType %r → tagging :Source:Unknown", item_type)
        return ("Source", "Unknown")
    return out


# Closed set for f-string label injection — every sub-label below must be
# a valid Cypher identifier (letters only, no special chars). Guards against
# accidental label expansion breaking Cypher.
_ALLOWED_SUB_LABELS = frozenset({"Paper", "Webpage", "CodeRepo", "Unknown"})


class Neo4jWriter:
    # T1 schema indexes — created once per process. Idempotent via
    # ``CREATE INDEX ... IF NOT EXISTS``; safe to re-run.
    #
    # Rationale (see plans/zotero-kg-t1-spec.md §2.4):
    #   source_doi / source_arxiv_id  — make SAME_WORK_AS scan do index seek
    #     rather than scan-all-Source. In Neo4j 5 a range index on :Source
    #     honors the label even when the node is multi-labelled
    #     :Source:Paper / :Source:ExternalRef.
    #   paper_id — drives the MATCH (:Paper {id:$pid}) entry point; we only
    #     have 4276 Papers now but the cost is trivial and pays off for
    #     backfill-script bulk ops.
    _INDEXES_CYPHER = (
        "CREATE INDEX source_doi      IF NOT EXISTS FOR (s:Source) ON (s.doi)",
        "CREATE INDEX source_arxiv_id IF NOT EXISTS FOR (s:Source) ON (s.arxiv_id)",
        "CREATE INDEX paper_id        IF NOT EXISTS FOR (p:Paper)  ON (p.id)",
    )

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Best-effort index creation. Neo4j sometimes 503s on DDL right
        after container start; we log and move on rather than block ingest
        startup. Backfill script can explicitly rerun via _ensure_indexes()."""
        try:
            with self.driver.session() as s:
                for stmt in self._INDEXES_CYPHER:
                    s.run(stmt).consume()
        except Exception as e:
            logger.warning("neo4j index ensure failed (will retry lazily): %s", e)

    def close(self):
        self.driver.close()

    @staticmethod
    def _canon(name: str) -> str:
        """Canonical form for entity dedup."""
        s = re.sub(r"[^\w\s]", " ", name.lower()).strip()
        return re.sub(r"\s+", " ", s)

    def write_paper(
        self,
        paper: ExtractedPaper,
        *,
        item_type: str,
        authors: list[dict] | None = None,
        ids: dict | None = None,
    ):
        """MERGE Source + Method/Concept/Dataset/Author/Venue nodes and edges.

        Retries on neo4j transient errors (ServiceUnavailable /
        SessionExpired / TransientError) with exp backoff. Each retry
        opens a fresh session so connection-pool eviction recovers
        gracefully. Neo4j's MERGE is idempotent so re-running the full
        block is safe.

        ``ids``: optional ``{"doi": str|None, "arxiv_id": str|None}``. When
        provided, the values are SET on the Paper node and a separate pass
        (`_link_same_work_as`) MERGEs ``SAME_WORK_AS`` edges to any other
        ``:Source`` sharing the same identifiers. Split off ``_write_paper_once``
        so retry of the main node/rel writes doesn't re-scan the entire
        identifier graph on every transient error.
        """
        import time as _t
        try:
            from neo4j.exceptions import (
                ServiceUnavailable, SessionExpired, TransientError)
            _retry_excs = (ServiceUnavailable, SessionExpired, TransientError)
        except ImportError:
            _retry_excs = ()

        delay = 1.0
        for attempt in range(1, 4):
            try:
                self._write_paper_once(
                    paper, item_type=item_type, authors=authors, ids=ids)
                break
            except _retry_excs as e:
                if attempt >= 3:
                    raise
                logger.warning("neo4j write_paper attempt %d/3 transient (%s), sleep %.1fs",
                               attempt, type(e).__name__, delay)
                _t.sleep(delay)
                delay = min(delay * 2, 15)

        # Link SAME_WORK_AS after main write succeeded. Standalone so the
        # retry loop above doesn't re-run this scan on every transient fail.
        if ids and (ids.get("doi") or ids.get("arxiv_id")):
            try:
                self._link_same_work_as(paper.paper_id)
            except Exception as e:
                # don't fail the whole paper ingest over a SAME_WORK_AS scan —
                # record but let caller continue
                logger.warning("SAME_WORK_AS link for %s failed: %s",
                               paper.paper_id, e)

    def _link_same_work_as(self, paper_id: str) -> dict:
        """Scan for existing Sources sharing this paper's DOI / arxiv_id and
        MERGE ``SAME_WORK_AS`` edges. Returns counts per kind.

        Safe to call standalone from the backfill script (T1-b) after bulk
        identifier SET. Uses two separate MERGE passes (DOI then arxiv_id)
        so both paths contribute a tag to ``r.kind_sources`` even when
        the edge already exists (``ON MATCH`` appends).
        """
        # T3 fix: start from :Source (not :Paper) so Wikipedia / blog items
        # that happen to carry a DOI in item.data.DOI ALSO get linked to
        # the corresponding Paper node via SAME_WORK_AS. The DOI uniqueness
        # invariant carries through — there's no concern that we'd
        # accidentally link an unrelated Source.
        counts = {"doi_links": 0, "arxiv_links": 0}
        with self.driver.session() as s:
            # DOI-driven linking
            res = s.run(
                """
                MATCH (p:Source {id: $pid}) WHERE p.doi IS NOT NULL
                WITH p
                MATCH (other:Source {doi: p.doi}) WHERE other.id <> p.id
                MERGE (p)-[r:SAME_WORK_AS]->(other)
                  ON CREATE SET r.kind_sources = ['doi'],
                                r.kind = CASE
                                  WHEN 'ExternalRef' IN labels(other) THEN 'preprint-published-ref'
                                  ELSE 'preprint-published'
                                END,
                                r.created_at = timestamp()
                  ON MATCH  SET r.kind_sources = apoc.coll.toSet(
                                  coalesce(r.kind_sources, []) + ['doi'])
                RETURN count(r) AS n
                """,
                pid=paper_id,
            )
            counts["doi_links"] = res.single()["n"]

            # arxiv_id-driven linking (independent MERGE; ON MATCH appends tag)
            res = s.run(
                """
                MATCH (p:Source {id: $pid}) WHERE p.arxiv_id IS NOT NULL
                WITH p
                MATCH (other:Source {arxiv_id: p.arxiv_id}) WHERE other.id <> p.id
                MERGE (p)-[r:SAME_WORK_AS]->(other)
                  ON CREATE SET r.kind_sources = ['arxiv'],
                                r.kind = 'arxiv-alias',
                                r.created_at = timestamp()
                  ON MATCH  SET r.kind_sources = apoc.coll.toSet(
                                  coalesce(r.kind_sources, []) + ['arxiv'])
                RETURN count(r) AS n
                """,
                pid=paper_id,
            )
            counts["arxiv_links"] = res.single()["n"]
        return counts

    def _write_paper_once(
        self,
        paper: ExtractedPaper,
        *,
        item_type: str,
        authors: list[dict] | None = None,
        ids: dict | None = None,
    ):
        doi = (ids or {}).get("doi")
        arxiv_id = (ids or {}).get("arxiv_id")
        # T3: route to :Paper / :Webpage / :CodeRepo / :Unknown by itemType.
        # Sub-label is from closed set ``_ALLOWED_SUB_LABELS``; f-string
        # interpolation is injection-safe so long as the set stays closed.
        _root, sub = _label_hierarchy(item_type)
        if sub not in _ALLOWED_SUB_LABELS:
            # Defensive: _label_hierarchy shouldn't produce anything outside
            # the allowed set, but guard against future additions breaking
            # Cypher. Degrade to Unknown.
            logger.error("unexpected sub-label %r (allowed=%s); "
                         "using :Unknown", sub, sorted(_ALLOWED_SUB_LABELS))
            sub = "Unknown"
        with self.driver.session() as s:
            s.run(
                f"""MERGE (p:Source:{sub} {{id: $pid}})
                    SET p.title = $title,
                        p.year = $year,
                        p.tldr = $tldr,
                        p.problem = $problem,
                        p.item_type = $item_type,
                        p.contribution_type = $ctype,
                        p.doi = $doi,
                        p.arxiv_id = $arxiv_id""",
                pid=paper.paper_id, title=paper.title, year=paper.year,
                tldr=paper.tldr, problem=paper.problem[:500],
                item_type=item_type, ctype=paper.contribution_type,
                doi=doi, arxiv_id=arxiv_id,
            )
            # venue
            if paper.venue:
                s.run("""MERGE (v:Venue {name: $v})
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[:PUBLISHED_AT]->(v)""",
                      pid=paper.paper_id, v=paper.venue)
            # authors
            for i, a in enumerate(authors or []):
                name = (a.get("lastName") or a.get("name") or "").strip()
                if not name:
                    continue
                s.run("""MERGE (au:Author {canonical_name: $cn})
                         SET au.name = coalesce(au.name, $name)
                         MERGE (p:Source {id: $pid})
                         MERGE (au)-[r:AUTHORED]->(p)
                         SET r.order = $ord""",
                      cn=self._canon(name), name=name,
                      pid=paper.paper_id, ord=i)
            # methods
            for m in paper.methods_used:
                s.run("""MERGE (m:Method {canonical_name: $cn})
                         SET m.name = coalesce(m.name, $n)
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[:USES]->(m)""",
                      cn=self._canon(m), n=m, pid=paper.paper_id)
            for m in paper.methods_proposed:
                s.run("""MERGE (m:Method {canonical_name: $cn})
                         SET m.name = coalesce(m.name, $n)
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[:PROPOSES]->(m)""",
                      cn=self._canon(m), n=m, pid=paper.paper_id)
            for mi in paper.methods_improved:
                s.run("""MERGE (m:Method {canonical_name: $cn})
                         SET m.name = coalesce(m.name, $n)
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[r:IMPROVES]->(m)
                         SET r.aspect = $asp""",
                      cn=self._canon(mi.base), n=mi.base,
                      pid=paper.paper_id, asp=mi.aspect)
            # datasets
            for d in paper.datasets:
                s.run("""MERGE (d:Dataset {canonical_name: $cn})
                         SET d.name = coalesce(d.name, $n)
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[:EVALUATES_ON]->(d)""",
                      cn=self._canon(d), n=d, pid=paper.paper_id)
            # concepts
            for c in paper.concepts:
                s.run("""MERGE (c:Concept {canonical_name: $cn})
                         SET c.name = coalesce(c.name, $n)
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[:ABOUT]->(c)""",
                      cn=self._canon(c), n=c, pid=paper.paper_id)
            # citations — targets go as :Source:ExternalRef until resolved
            for ref in paper.references:
                ref_id = f"ext:{self._canon(ref.cited_title)[:80]}"
                s.run("""MERGE (t:Source {id: $rid})
                         ON CREATE SET t:ExternalRef,
                                       t.title = $title,
                                       t.author_year = $ay
                         MERGE (p:Source {id: $pid})
                         MERGE (p)-[r:CITES]->(t)
                         SET r.role = $role,
                             r.context_quote = $quote""",
                      rid=ref_id, title=ref.cited_title[:300],
                      ay=ref.cited_author_year, pid=paper.paper_id,
                      role=ref.role, quote=ref.context_quote)


# ======================================================================
# One-shot factory
# ======================================================================
def make_writers(
    *,
    sqlite_path: str | Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_collection: str = "zotero_library",
    dashscope_api_key: str | None = None,
) -> tuple[SQLiteStore, QdrantWriter, Neo4jWriter]:
    dashscope_key = dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        raise RuntimeError("DASHSCOPE_API_KEY required for embeddings")
    sql = SQLiteStore(sqlite_path)
    q = QdrantWriter(qdrant_host, qdrant_port, qdrant_collection, dashscope_key)
    n = Neo4jWriter(neo4j_uri, neo4j_user, neo4j_password)
    return sql, q, n

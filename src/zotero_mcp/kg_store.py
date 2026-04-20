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
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the ThreadPoolExecutor workers can share the
        # connection; writes serialized via self._lock, so we stay consistent.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self._lock = threading.Lock()

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


# ======================================================================
# Chunking + embedding helpers
# ======================================================================
_H2_SPLIT = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


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
        from qdrant_client import QdrantClient
        self.qc = QdrantClient(host=host, port=port, timeout=60)
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
        """Chunk markdown, embed (dense + sparse), upsert to Qdrant. Returns chunk count."""
        from qdrant_client.http.models import PointStruct, SparseVector

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
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": d_vec,
                    "bm25": SparseVector(
                        indices=s_vec.indices.tolist(),
                        values=s_vec.values.tolist(),
                    ),
                },
                payload={
                    "paper_id": paper.paper_id,
                    "item_type": item_type,
                    "title": paper.title,
                    "year": paper.year or 0,
                    "section_title": c["section_title"],
                    "chunk_idx": c["chunk_idx"],
                    "text": c["text"],
                },
            ))
        self.qc.upsert(collection_name=self.collection, points=points)
        return len(points)


# ======================================================================
# Neo4jWriter — MERGE nodes + relations
# ======================================================================
class Neo4jWriter:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

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
    ):
        """MERGE Source + Method/Concept/Dataset/Author/Venue nodes and edges."""
        with self.driver.session() as s:
            # Source node (always Paper for now — book/thesis subtypes come later)
            s.run(
                """MERGE (p:Source:Paper {id: $pid})
                   SET p.title = $title,
                       p.year = $year,
                       p.tldr = $tldr,
                       p.problem = $problem,
                       p.item_type = $item_type,
                       p.contribution_type = $ctype""",
                pid=paper.paper_id, title=paper.title, year=paper.year,
                tldr=paper.tldr, problem=paper.problem[:500],
                item_type=item_type, ctype=paper.contribution_type,
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

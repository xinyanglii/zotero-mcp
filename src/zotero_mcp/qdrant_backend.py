"""
Qdrant remote backend for zotero-mcp semantic search.

Drop-in replacement for ``chroma_client.ChromaClient`` that stores vectors
in a remote Qdrant server (e.g. reachable over tailnet). Public method
surface matches ``ChromaClient`` so ``semantic_search.py`` can use either.

Also defines ``QwenEmbeddingFunction`` — an OpenAI-compatible embedder
pointed at DashScope (``text-embedding-v4``, 2048 dims by default).

NOTE: This module intentionally does NOT import chromadb. It is loaded only
when ``backend.type == "qdrant"`` in the config.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "qdrant-client is required for the Qdrant backend. "
        "Install it with: pip install 'zotero-mcp-server[qdrant]'"
    ) from e


# ---------------------------------------------------------------------------
# Embedding function (OpenAI-compatible, for DashScope Qwen v4)
# ---------------------------------------------------------------------------
class QwenEmbeddingFunction:
    """OpenAI-compatible embedding function for DashScope Qwen v4.

    Exposes the same small API the rest of zotero-mcp expects on an
    embedding function: ``__call__(list[str]) -> list[list[float]]``,
    ``embed_query(str) -> list[float]``, ``truncate(str, int) -> str``,
    and ``max_input_tokens``.
    """

    # Qwen3-Embedding-v4 supports up to 8192 input tokens.
    max_input_tokens = 8000

    # DashScope caps batch size at 10 inputs per request for text-embedding-v4.
    # (Confirmed by DashScope docs 2025; exceeding returns 400.)
    MAX_BATCH = 10

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        dimensions: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required for QwenEmbeddingFunction "
                "(set env var or pass api_key)"
            )

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for Qwen embeddings"
            ) from e
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def name() -> str:
        return "qwen-dashscope"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "base_url": self.base_url,
        }

    def __call__(self, input: list[str]) -> list[list[float]]:
        texts = list(input)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batch = texts[start : start + self.MAX_BATCH]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.__call__([text])[0]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Char-based conservative truncation (~3 chars/token for Qwen BPE)."""
        max_chars = max_tokens * 3
        if len(text) > max_chars:
            text = text[:max_chars]
        return text


# ---------------------------------------------------------------------------
# Chroma-compatible wrapper around QdrantClient
# ---------------------------------------------------------------------------
def _doc_id_to_point_id(doc_id: str) -> str:
    """Qdrant point IDs must be uint or UUID. Deterministically derive a
    UUID from the caller's free-form string ID (e.g. Zotero item key)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"zotero-mcp://{doc_id}"))


class _QdrantCollectionShim:
    """Thin shim mimicking ``chromadb.Collection`` methods actually used
    by ``cli.py db-inspect`` (``get(limit=, include=[...])`` and
    ``count()``). Keep surface minimal — only what existing code touches.
    """

    def __init__(self, parent: "ZoteroQdrantClient"):
        self._parent = parent

    def count(self) -> int:
        return self._parent._count()

    def get(
        self,
        ids: list[str] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        include = include or []
        return self._parent._get(ids=ids, limit=limit, include=include)


class ZoteroQdrantClient:
    """Qdrant-backed drop-in replacement for ``ChromaClient``.

    Public method surface mirrors ``chroma_client.ChromaClient`` so the
    rest of the codebase stays unchanged.
    """

    def __init__(
        self,
        collection_name: str = "zotero_library",
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        embedding_model: str = "qwen",
        embedding_config: dict[str, Any] | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}

        client_kwargs: dict[str, Any] = {"prefer_grpc": prefer_grpc}
        if url:
            client_kwargs["url"] = url
        else:
            client_kwargs["host"] = host
            client_kwargs["port"] = port
        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = QdrantClient(**client_kwargs)
        self.embedding_function = self._create_embedding_function()
        self._ensure_collection()

        # Chroma-compat attribute used by ``cli.py db-inspect``.
        self.collection = _QdrantCollectionShim(self)

        # Used by db-status display only.
        self.persist_directory = f"qdrant://{url or f'{host}:{port}'}/{collection_name}"

    # -- embedder --------------------------------------------------------
    def _create_embedding_function(self):
        if self.embedding_model in ("qwen", "qwen-dashscope", "dashscope"):
            return QwenEmbeddingFunction(
                model_name=self.embedding_config.get("model_name", "text-embedding-v4"),
                dimensions=self.embedding_config.get("dimensions", 2048),
                api_key=self.embedding_config.get("api_key"),
                base_url=self.embedding_config.get("base_url")
                or self.embedding_config.get("api_base"),
            )
        # For other providers, lazily import from chroma_client so we
        # don't force chromadb as a hard dep when only Qdrant+Qwen is used.
        if self.embedding_model == "openai":
            from .chroma_client import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(
                model_name=self.embedding_config.get("model_name", "text-embedding-3-small"),
                api_key=self.embedding_config.get("api_key"),
                base_url=self.embedding_config.get("base_url"),
            )
        if self.embedding_model == "gemini":
            from .chroma_client import GeminiEmbeddingFunction
            return GeminiEmbeddingFunction(
                model_name=self.embedding_config.get("model_name", "gemini-embedding-001"),
                api_key=self.embedding_config.get("api_key"),
                base_url=self.embedding_config.get("base_url"),
            )
        # HuggingFace fallback
        from .chroma_client import HuggingFaceEmbeddingFunction
        model_name = self.embedding_config.get("model_name", self.embedding_model)
        return HuggingFaceEmbeddingFunction(model_name=model_name)

    @property
    def embedding_max_tokens(self) -> int:
        return getattr(self.embedding_function, "max_input_tokens", 8000)

    def truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = self.embedding_max_tokens
        if hasattr(self.embedding_function, "truncate"):
            return self.embedding_function.truncate(text, max_tokens)
        max_chars = max_tokens * 3
        return text[:max_chars] if len(text) > max_chars else text

    # -- collection lifecycle -------------------------------------------
    def _vector_size(self) -> int:
        ef = self.embedding_function
        if hasattr(ef, "dimensions"):
            return int(ef.dimensions)
        # Probe: embed a dummy string to read the dim.
        probe = ef(["dim-probe"])
        return len(probe[0])

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        size = self._vector_size()
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=size, distance=qmodels.Distance.COSINE
            ),
        )
        logger.info(
            f"Created Qdrant collection '{self.collection_name}' with vector size {size}"
        )

    # -- write path ------------------------------------------------------
    def _build_points(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> list[qmodels.PointStruct]:
        vectors = self.embedding_function(documents)
        points: list[qmodels.PointStruct] = []
        for doc, meta, doc_id, vec in zip(documents, metadatas, ids, vectors):
            payload = dict(meta or {})
            payload["_doc_id"] = doc_id
            payload["_document"] = doc
            points.append(
                qmodels.PointStruct(
                    id=_doc_id_to_point_id(doc_id),
                    vector=vec,
                    payload=payload,
                )
            )
        return points

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        points = self._build_points(documents, metadatas, ids)
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info(f"Added {len(points)} documents to Qdrant collection")

    def upsert_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        # Qdrant upsert is idempotent on point id.
        self.add_documents(documents, metadatas, ids)

    def delete_documents(self, ids: list[str]) -> None:
        if not ids:
            return
        point_ids = [_doc_id_to_point_id(d) for d in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
            wait=True,
        )
        logger.info(f"Deleted {len(ids)} documents from Qdrant collection")

    # -- read path -------------------------------------------------------
    def search(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,  # ignored (Qdrant has no doc-text filter)
    ) -> dict[str, Any]:
        if where_document:
            logger.warning("Qdrant backend ignores where_document filter (unsupported)")

        qfilter = _chroma_where_to_qdrant(where) if where else None

        # Embed queries (use embed_query for per-provider query-time tuning).
        query_vectors: list[list[float]] = []
        for qt in query_texts:
            if hasattr(self.embedding_function, "embed_query"):
                qv = self.embedding_function.embed_query(qt)
            else:
                qv = self.embedding_function([qt])[0]
            if hasattr(qv, "tolist"):
                qv = qv.tolist()
            query_vectors.append(qv)

        # Build Chroma-shaped result dict.
        ids_out: list[list[str]] = []
        docs_out: list[list[str]] = []
        metas_out: list[list[dict[str, Any]]] = []
        dists_out: list[list[float]] = []
        for qv in query_vectors:
            # qdrant-client ≥1.10 replaced ``.search()`` with ``.query_points()``
            # (old name removed in ≥1.13). Use the new API.
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qv,
                query_filter=qfilter,
                limit=n_results,
                with_payload=True,
            )
            hits = resp.points
            ids_out.append([h.payload.get("_doc_id", str(h.id)) for h in hits])
            docs_out.append([h.payload.get("_document", "") for h in hits])
            metas_out.append([_strip_internal(h.payload) for h in hits])
            # Chroma returns "distances" where lower is better. Qdrant cosine
            # score is similarity (higher is better). Convert to distance.
            dists_out.append([1.0 - float(h.score) for h in hits])

        logger.info(f"Qdrant semantic search returned {len(ids_out[0]) if ids_out else 0} results")
        return {
            "ids": ids_out,
            "documents": docs_out,
            "metadatas": metas_out,
            "distances": dists_out,
        }

    def get_collection_info(self) -> dict[str, Any]:
        try:
            return {
                "name": self.collection_name,
                "count": self._count(),
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting Qdrant collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e),
            }

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass  # may not exist yet
        self._ensure_collection()
        logger.info(f"Reset Qdrant collection '{self.collection_name}'")

    def document_exists(self, doc_id: str) -> bool:
        pid = _doc_id_to_point_id(doc_id)
        try:
            recs = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[pid],
                with_payload=False,
                with_vectors=False,
            )
            return len(recs) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        pid = _doc_id_to_point_id(doc_id)
        try:
            recs = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[pid],
                with_payload=True,
                with_vectors=False,
            )
            if recs:
                return _strip_internal(recs[0].payload)
            return None
        except Exception:
            return None

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        pid_to_doc = {_doc_id_to_point_id(d): d for d in ids}
        try:
            recs = self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(pid_to_doc.keys()),
                with_payload=False,
                with_vectors=False,
            )
            return {pid_to_doc[str(r.id)] for r in recs if str(r.id) in pid_to_doc}
        except Exception:
            return set()

    # -- internals used by _QdrantCollectionShim ------------------------
    def _count(self) -> int:
        try:
            return int(self.client.count(self.collection_name, exact=True).count)
        except Exception as e:
            logger.error(f"Qdrant count failed: {e}")
            return 0

    def _get(
        self,
        ids: list[str] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mimic ``chromadb.Collection.get``. Only supports the calls
        present in the zotero-mcp codebase (db-inspect, semantic_search)."""
        include = include or []
        want_meta = "metadatas" in include
        want_doc = "documents" in include

        if ids is not None:
            pid_to_doc = {_doc_id_to_point_id(d): d for d in ids}
            recs = self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(pid_to_doc.keys()),
                with_payload=True,
                with_vectors=False,
            )
            out_ids: list[str] = []
            out_metas: list[dict[str, Any]] = []
            out_docs: list[str] = []
            for r in recs:
                doc_id = pid_to_doc.get(str(r.id)) or r.payload.get("_doc_id", str(r.id))
                out_ids.append(doc_id)
                if want_meta:
                    out_metas.append(_strip_internal(r.payload))
                if want_doc:
                    out_docs.append(r.payload.get("_document", ""))
            result: dict[str, Any] = {"ids": out_ids}
            if want_meta:
                result["metadatas"] = out_metas
            if want_doc:
                result["documents"] = out_docs
            return result

        # Scroll path (limit-based).
        limit_val = limit or 100
        points, _next = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit_val,
            with_payload=True,
            with_vectors=False,
        )
        out_ids = []
        out_metas = []
        out_docs = []
        for p in points:
            out_ids.append(p.payload.get("_doc_id", str(p.id)))
            if want_meta:
                out_metas.append(_strip_internal(p.payload))
            if want_doc:
                out_docs.append(p.payload.get("_document", ""))
        result = {"ids": out_ids}
        if want_meta:
            result["metadatas"] = out_metas
        if want_doc:
            result["documents"] = out_docs
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_INTERNAL_PAYLOAD_KEYS = {"_doc_id", "_document"}


def _strip_internal(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {k: v for k, v in payload.items() if k not in _INTERNAL_PAYLOAD_KEYS}


def _chroma_where_to_qdrant(where: dict[str, Any]) -> qmodels.Filter | None:
    """Best-effort translation of a subset of Chroma's ``where`` dialect
    into a Qdrant ``Filter``.

    Supports:
      - ``{field: value}`` → match
      - ``{field: {"$eq": value}}`` → match
      - ``{field: {"$ne": value}}`` → must_not match
      - ``{field: {"$in": [v,...]}}`` → any-of
      - ``{"$and": [clause, clause, ...]}`` → must
      - ``{"$or":  [clause, clause, ...]}`` → should
    Other operators fall back to a no-op filter with a warning.
    """
    if not where:
        return None

    def _clause_to_conditions(clause: dict[str, Any]) -> tuple[list, list]:
        must: list = []
        must_not: list = []
        for key, val in clause.items():
            if key in ("$and", "$or"):
                continue  # handled at top level
            if isinstance(val, dict):
                for op, v in val.items():
                    if op == "$eq":
                        must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=v)))
                    elif op == "$ne":
                        must_not.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=v)))
                    elif op == "$in":
                        must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(v))))
                    else:
                        logger.warning(f"Qdrant filter translation: unsupported op '{op}' on '{key}'")
            else:
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=val)))
        return must, must_not

    if "$and" in where:
        must: list = []
        must_not: list = []
        for sub in where["$and"]:
            m, mn = _clause_to_conditions(sub)
            must.extend(m)
            must_not.extend(mn)
        return qmodels.Filter(must=must or None, must_not=must_not or None)

    if "$or" in where:
        should: list = []
        for sub in where["$or"]:
            m, _mn = _clause_to_conditions(sub)
            should.extend(m)
        return qmodels.Filter(should=should or None)

    must, must_not = _clause_to_conditions(where)
    return qmodels.Filter(must=must or None, must_not=must_not or None)


def create_qdrant_client(config: dict[str, Any]) -> ZoteroQdrantClient:
    """Build a ZoteroQdrantClient from a parsed config dict (see
    ``backend_factory.create_backend_client``)."""
    backend_cfg = (config.get("backend") or {}).get("qdrant") or {}
    collection_name = backend_cfg.get("collection", "zotero_library")
    host = backend_cfg.get("host", "localhost")
    port = int(backend_cfg.get("port", 6333))
    url = backend_cfg.get("url")
    api_key = backend_cfg.get("api_key") or os.getenv("QDRANT_API_KEY")
    prefer_grpc = bool(backend_cfg.get("prefer_grpc", False))

    embedding_cfg = config.get("embedding") or config.get("semantic_search", {}).get("embedding_config") or {}
    provider = embedding_cfg.get("provider") or config.get("semantic_search", {}).get("embedding_model") or "qwen"

    # Normalize: openai provider with dashscope base_url is basically qwen.
    if provider == "openai" and "dashscope" in (embedding_cfg.get("api_base") or embedding_cfg.get("base_url") or ""):
        provider = "qwen"

    return ZoteroQdrantClient(
        collection_name=collection_name,
        host=host,
        port=port,
        url=url,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        embedding_model=provider,
        embedding_config=embedding_cfg,
    )

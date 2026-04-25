"""
Backend factory: chooses between ChromaDB (local PersistentClient, default)
and Qdrant (remote HttpClient) based on configuration.

Selection rules (first match wins):
  1. Env ``ZOTERO_MCP_BACKEND`` in {"chroma", "qdrant"} → force that backend
  2. config.json has ``backend.type == "qdrant"`` → Qdrant
  3. fallback → ChromaDB (preserves pre-existing behavior)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning(f"Error loading config from {config_path}: {e}")
        return {}


def _resolve_backend(config: dict[str, Any]) -> str:
    override = os.getenv("ZOTERO_MCP_BACKEND", "").strip().lower()
    if override in {"chroma", "qdrant"}:
        return override
    backend_cfg = config.get("backend") or {}
    btype = (backend_cfg.get("type") or "").strip().lower()
    if btype in {"chroma", "qdrant"}:
        return btype
    return "chroma"


def create_backend_client(config_path: str | None = None):
    """Build the semantic-search client (Chroma or Qdrant) per config."""
    config = _load_config(config_path)
    backend = _resolve_backend(config)

    if backend == "qdrant":
        from zotero_kg.qdrant_backend import create_qdrant_client
        logger.info("Using Qdrant remote backend for semantic search")
        return create_qdrant_client(config)

    from .chroma_client import create_chroma_client
    logger.info("Using ChromaDB local backend for semantic search")
    return create_chroma_client(config_path)

"""
Graph-walk MCP tool: ``zotero_related``.

This is the one genuinely new capability the KG pipeline gives us beyond
what ``zotero_semantic_search`` already exposes. Given a paper's Zotero
key, return its neighbors in the Neo4j graph (CITES / ABOUT / USES /
SAME_WORK_AS / ...) so Claude can assemble baseline chains, 'related
work' lists, 'papers that share concept X', etc. without re-querying
Qdrant for each hop.

Enable by setting ``NEO4J_ZOTERO_URI`` + ``NEO4J_ZOTERO_USER`` +
``NEO4J_ZOTERO_PASSWORD`` env vars. When the creds are missing, the
tool returns a short message explaining how to configure it — it does
NOT fail the server. This keeps the package usable for users who
haven't set up the KG yet.
"""
from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import Context

from zotero_mcp._app import mcp


DEFAULT_EDGE_TYPES = ["CITES", "ABOUT", "USES", "PROPOSES", "IMPROVES",
                      "EVALUATES_ON", "SAME_WORK_AS"]


def _neo4j_configured() -> bool:
    return bool(os.environ.get("NEO4J_ZOTERO_URI")
                and os.environ.get("NEO4J_ZOTERO_USER")
                and os.environ.get("NEO4J_ZOTERO_PASSWORD"))


def _driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        os.environ["NEO4J_ZOTERO_URI"],
        auth=(os.environ["NEO4J_ZOTERO_USER"], os.environ["NEO4J_ZOTERO_PASSWORD"]),
    )


@mcp.tool(
    name="zotero_related",
    description=(
        "Find papers related to a given Zotero item via the knowledge graph "
        "(CITES, ABOUT, USES, SAME_WORK_AS etc). "
        "Use this for queries like 'what are the baseline/prior-work papers "
        "for X', 'which papers use method Y', 'papers that share concept Z'. "
        "Complements zotero_semantic_search (which does text similarity). "
        "Requires NEO4J_ZOTERO_* env vars; returns graceful error if the "
        "KG isn't set up yet."
    ),
)
def zotero_related(
    paper_id: str,
    hops: int = 1,
    edge_types: list[str] | None = None,
    limit: int = 20,
    *,
    ctx: Context,
) -> str:
    """Return neighbors of ``paper_id`` in the Zotero KG.

    Args:
        paper_id: Zotero item key of the source paper.
        hops: 1 for direct neighbors, 2 for their neighbors too.
              Clamped to [1, 3] to bound cost.
        edge_types: Which relationship types to traverse. Defaults to all
              commonly useful ones. Pass ["CITES"] for a strict citation
              walk, ["ABOUT"] for concept-shared papers, etc.
        limit: Max papers returned across all hops.

    Returns:
        JSON string with ``{"source": {...}, "neighbors": [...]}``.
    """
    if not _neo4j_configured():
        return json.dumps({
            "error": "Neo4j not configured",
            "hint": "Set NEO4J_ZOTERO_URI / NEO4J_ZOTERO_USER / "
                    "NEO4J_ZOTERO_PASSWORD in ~/.claude/.env.local. "
                    "See plans/zotero-graphrag-plan.md §0.1.",
        })

    try:
        from neo4j import GraphDatabase  # noqa: F401 (imports driver lazily)
    except ImportError:
        return json.dumps({
            "error": "neo4j driver missing",
            "hint": "pip install neo4j",
        })

    hops = max(1, min(3, int(hops)))
    types = edge_types or DEFAULT_EDGE_TYPES
    # Cypher needs a pipe-joined type list for variable-length match
    rel_pattern = "|".join(types)
    ctx.info(f"graph walk: {paper_id} hops={hops} types={rel_pattern} limit={limit}")

    # Note: ``size(r)`` (list length) not ``length(r)`` (which is Path-only).
    # Variable-length match returns a list of relationships, not a path.
    cypher = f"""
    MATCH (src:Source {{id: $pid}})
    CALL {{
        WITH src
        MATCH (src)-[r:{rel_pattern}*1..{hops}]-(neighbor:Source)
        WHERE neighbor.id <> src.id
        RETURN neighbor, r, size(r) AS depth
        LIMIT $limit
    }}
    RETURN src {{.id, .title, .year, .tldr, .contribution_type, .item_type}} AS source,
           collect(DISTINCT {{
               id: neighbor.id,
               title: neighbor.title,
               year: neighbor.year,
               tldr: neighbor.tldr,
               contribution_type: neighbor.contribution_type,
               item_type: neighbor.item_type,
               external: 'ExternalRef' IN labels(neighbor),
               depth: depth,
               edge_types: [rel IN r | type(rel)],
               first_edge_role: [rel IN r WHERE type(rel) = 'CITES' | rel.role][0]
           }}) AS neighbors
    """

    try:
        with _driver().session() as session:
            result = session.run(cypher, pid=paper_id, limit=limit).single()
    except Exception as e:
        ctx.error(f"Neo4j query failed: {e}")
        return json.dumps({"error": "query_failed", "detail": str(e)[:300]})

    if result is None:
        return json.dumps({
            "error": "paper_not_in_kg",
            "paper_id": paper_id,
            "hint": "The paper may not be ingested yet. Check with "
                    "zotero_semantic_search or run the ingest pipeline.",
        })

    payload: dict[str, Any] = {
        "source": result["source"],
        "neighbors": result["neighbors"] or [],
        "query": {"hops": hops, "edge_types": types, "limit": limit},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

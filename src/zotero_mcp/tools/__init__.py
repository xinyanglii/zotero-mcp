"""Tool modules — importing this package registers all tools with the MCP app."""

from zotero_mcp.tools import (  # noqa: F401
    annotations,
    connectors,
    retrieval,
    search,
    write,
)

# Optional: Scite enrichment (requires ``pip install zotero-mcp-server[scite]``)
try:
    from zotero_mcp.tools import scite as scite  # noqa: F401
except ImportError:
    pass

# Optional: Knowledge-graph walk tool (requires ``neo4j`` package + NEO4J_ZOTERO_*
# env vars; tool registers unconditionally and reports a graceful error at
# call time when the config is absent, so the import itself only fails when
# the ``neo4j`` driver isn't installed at all).
try:
    from zotero_mcp.tools import graph as graph  # noqa: F401
except ImportError:
    pass

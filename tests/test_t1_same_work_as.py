"""T1 integration tests for SAME_WORK_AS edge building.

Uses the live Neo4j instance (cheaper + matches prod than testcontainers)
isolated via a unique ``T1TEST_*`` paper-id prefix. Each test creates 2-3
test nodes, runs the Cypher under test, asserts, then cleans up.

Skipped if NEO4J_ZOTERO_URI is not set. Does NOT require Docker.

Covers spec §2.3 and §8.2:
  - A: two Papers same DOI  → 1 edge, kind=preprint-published, kind_sources=['doi']
  - B: two Papers same DOI + same arxiv_id → 1 edge, kind_sources contains both
  - C: two Papers only same arxiv_id → 1 edge, kind=arxiv-alias
  - D: Paper + ExternalRef same DOI → 1 edge, kind=preprint-published-ref
  - E: idempotency — running _link_same_work_as twice doesn't duplicate edges
"""
from __future__ import annotations

import os
import uuid

import pytest

from zotero_mcp.kg_store import Neo4jWriter


_NEO4J_AVAILABLE = all(
    os.environ.get(k)
    for k in ("NEO4J_ZOTERO_URI", "NEO4J_ZOTERO_PASSWORD")
)
pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE,
    reason="Neo4j env (NEO4J_ZOTERO_URI/USER/PASSWORD) not configured",
)


@pytest.fixture(scope="module")
def ne() -> Neo4jWriter:
    w = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    yield w
    w.close()


@pytest.fixture
def test_prefix() -> str:
    """Unique prefix per-test; used to build isolated test node ids. Every
    fixture teardown removes nodes + edges matching this prefix."""
    return f"T1TEST_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup(ne, test_prefix):
    """Teardown: delete all nodes whose id starts with test_prefix."""
    yield
    with ne.driver.session() as s:
        s.run(
            """MATCH (n:Source) WHERE n.id STARTS WITH $pref
               DETACH DELETE n""",
            pref=test_prefix,
        ).consume()


def _create_paper(ne, pid, *, title="", doi=None, arxiv_id=None):
    with ne.driver.session() as s:
        s.run(
            """MERGE (p:Source:Paper {id: $pid})
               SET p.title = $title, p.doi = $doi, p.arxiv_id = $arxiv_id,
                   p.contribution_type = 'method'""",
            pid=pid, title=title, doi=doi, arxiv_id=arxiv_id,
        ).consume()


def _create_external_ref(ne, rid, *, title="", doi=None):
    with ne.driver.session() as s:
        s.run(
            """MERGE (e:Source:ExternalRef {id: $rid})
               SET e.title = $title, e.doi = $doi""",
            rid=rid, title=title, doi=doi,
        ).consume()


def _edges_of(ne, pid):
    """Return all SAME_WORK_AS outgoing edges from pid with properties."""
    with ne.driver.session() as s:
        data = s.run(
            """MATCH (p:Source {id: $pid})-[r:SAME_WORK_AS]->(other:Source)
               RETURN other.id AS other_id, other.doi AS other_doi,
                      other.arxiv_id AS other_arxiv_id,
                      r.kind AS kind, r.kind_sources AS kind_sources""",
            pid=pid,
        ).data()
    return data


# ---------------------------------------------------------------------------
# Test A — two Papers, same DOI
# ---------------------------------------------------------------------------
def test_same_doi_two_papers(ne, test_prefix, cleanup):
    pid1 = f"{test_prefix}_A1"
    pid2 = f"{test_prefix}_A2"
    _create_paper(ne, pid1, title="Deep X", doi="10.xxxx/test-a")
    _create_paper(ne, pid2, title="Deep X (preprint)", doi="10.xxxx/test-a")

    counts = ne._link_same_work_as(pid1)
    assert counts["doi_links"] == 1
    assert counts["arxiv_links"] == 0

    edges = _edges_of(ne, pid1)
    assert len(edges) == 1
    e = edges[0]
    assert e["other_id"] == pid2
    assert e["kind"] == "preprint-published"
    assert e["kind_sources"] == ["doi"]


# ---------------------------------------------------------------------------
# Test B — two Papers, same DOI + same arxiv_id → kind_sources should accumulate
# ---------------------------------------------------------------------------
def test_same_doi_and_arxiv_accumulates_kind_sources(ne, test_prefix, cleanup):
    pid1 = f"{test_prefix}_B1"
    pid2 = f"{test_prefix}_B2"
    _create_paper(ne, pid1, doi="10.xxxx/test-b", arxiv_id="2601.00001")
    _create_paper(ne, pid2, doi="10.xxxx/test-b", arxiv_id="2601.00001")

    counts = ne._link_same_work_as(pid1)
    # Both passes find the same other node → second MERGE hits ON MATCH + appends
    assert counts["doi_links"] == 1
    assert counts["arxiv_links"] == 1

    edges = _edges_of(ne, pid1)
    assert len(edges) == 1
    e = edges[0]
    # kind stays as first-write value (preprint-published from DOI pass)
    assert e["kind"] == "preprint-published"
    # kind_sources should contain BOTH 'doi' and 'arxiv' (dedup by apoc.coll.toSet)
    assert set(e["kind_sources"]) == {"doi", "arxiv"}


# ---------------------------------------------------------------------------
# Test C — two Papers, only same arxiv_id
# ---------------------------------------------------------------------------
def test_only_arxiv_same_kind_is_arxiv_alias(ne, test_prefix, cleanup):
    pid1 = f"{test_prefix}_C1"
    pid2 = f"{test_prefix}_C2"
    _create_paper(ne, pid1, arxiv_id="2601.00002")
    _create_paper(ne, pid2, arxiv_id="2601.00002")

    counts = ne._link_same_work_as(pid1)
    assert counts["doi_links"] == 0
    assert counts["arxiv_links"] == 1

    edges = _edges_of(ne, pid1)
    assert len(edges) == 1
    assert edges[0]["kind"] == "arxiv-alias"
    assert edges[0]["kind_sources"] == ["arxiv"]


# ---------------------------------------------------------------------------
# Test D — Paper + ExternalRef, same DOI
# ---------------------------------------------------------------------------
def test_paper_and_external_ref_same_doi(ne, test_prefix, cleanup):
    pid = f"{test_prefix}_D1"
    rid = f"{test_prefix}_Dref"
    _create_paper(ne, pid, doi="10.xxxx/test-d")
    _create_external_ref(ne, rid, doi="10.xxxx/test-d")

    counts = ne._link_same_work_as(pid)
    assert counts["doi_links"] == 1

    edges = _edges_of(ne, pid)
    assert len(edges) == 1
    assert edges[0]["other_id"] == rid
    assert edges[0]["kind"] == "preprint-published-ref"
    assert edges[0]["kind_sources"] == ["doi"]


# ---------------------------------------------------------------------------
# Test E — idempotency: re-running _link_same_work_as doesn't duplicate
# ---------------------------------------------------------------------------
def test_idempotent_rerun(ne, test_prefix, cleanup):
    pid1 = f"{test_prefix}_E1"
    pid2 = f"{test_prefix}_E2"
    _create_paper(ne, pid1, doi="10.xxxx/test-e", arxiv_id="2601.00003")
    _create_paper(ne, pid2, doi="10.xxxx/test-e", arxiv_id="2601.00003")

    # First run
    ne._link_same_work_as(pid1)
    edges_1 = _edges_of(ne, pid1)
    assert len(edges_1) == 1

    # Second run — MERGE should not create duplicates
    ne._link_same_work_as(pid1)
    edges_2 = _edges_of(ne, pid1)
    assert len(edges_2) == 1
    # kind_sources remains deduped to {'doi', 'arxiv'}
    assert set(edges_2[0]["kind_sources"]) == {"doi", "arxiv"}

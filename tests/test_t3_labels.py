"""T3 integration tests — Neo4j label routing by itemType.

Uses the live Neo4j instance with T3TEST_* prefix isolation (same pattern
as test_t1_same_work_as.py). Cleans up on teardown. No Docker required.
"""
from __future__ import annotations

import os
import uuid

import pytest

from zotero_mcp.extractor import ExtractedPaper
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
    return f"T3TEST_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup(ne, test_prefix):
    """Teardown: delete all :Source nodes whose id starts with test_prefix,
    and any dangling auth/method/concept nodes we created for them."""
    yield
    with ne.driver.session() as s:
        s.run(
            """MATCH (n:Source) WHERE n.id STARTS WITH $pref
               DETACH DELETE n""",
            pref=test_prefix,
        ).consume()


def _fake_paper(pid: str, *, title="T3 test") -> ExtractedPaper:
    return ExtractedPaper(paper_id=pid, title=title)


def _labels_of(ne, pid: str) -> list[str]:
    with ne.driver.session() as s:
        rec = s.run(
            "MATCH (n:Source {id: $pid}) RETURN labels(n) AS ls",
            pid=pid,
        ).single()
    return sorted(rec["ls"]) if rec else []


# ---------------------------------------------------------------------------
# Single-type routing
# ---------------------------------------------------------------------------
def test_webpage_gets_webpage_label(ne, test_prefix, cleanup):
    pid = f"{test_prefix}_WEB"
    ne.write_paper(_fake_paper(pid), item_type="webpage")
    assert _labels_of(ne, pid) == ["Source", "Webpage"]


def test_computer_program_gets_coderepo_label(ne, test_prefix, cleanup):
    pid = f"{test_prefix}_CODE"
    ne.write_paper(_fake_paper(pid), item_type="computerProgram")
    assert _labels_of(ne, pid) == ["CodeRepo", "Source"]


def test_blog_post_gets_webpage_label(ne, test_prefix, cleanup):
    pid = f"{test_prefix}_BLOG"
    ne.write_paper(_fake_paper(pid), item_type="blogPost")
    assert _labels_of(ne, pid) == ["Source", "Webpage"]


def test_journal_article_regression_still_paper(ne, test_prefix, cleanup):
    """Regression: paper types still get :Source:Paper (no T0/T1 breakage)."""
    pid = f"{test_prefix}_PAPER"
    ne.write_paper(_fake_paper(pid), item_type="journalArticle")
    assert _labels_of(ne, pid) == ["Paper", "Source"]


# ---------------------------------------------------------------------------
# Coexistence — two different sub-labels under same test prefix don't pollute
# ---------------------------------------------------------------------------
def test_webpage_and_coderepo_coexist_without_polluting(ne, test_prefix, cleanup):
    """Writing a :Webpage and a :CodeRepo in the same session (or same key
    prefix) must keep their label hierarchies independent."""
    w_pid = f"{test_prefix}_W"
    c_pid = f"{test_prefix}_C"
    ne.write_paper(_fake_paper(w_pid), item_type="webpage")
    ne.write_paper(_fake_paper(c_pid), item_type="software")

    assert _labels_of(ne, w_pid) == ["Source", "Webpage"]
    assert _labels_of(ne, c_pid) == ["CodeRepo", "Source"]

    # Cross-verify: Webpage scan must NOT include CodeRepo node
    with ne.driver.session() as s:
        webpage_rows = s.run(
            """MATCH (n:Source:Webpage) WHERE n.id STARTS WITH $pref
               RETURN n.id AS id""", pref=test_prefix).data()
        coderepo_rows = s.run(
            """MATCH (n:Source:CodeRepo) WHERE n.id STARTS WITH $pref
               RETURN n.id AS id""", pref=test_prefix).data()
    webpage_ids = {r["id"] for r in webpage_rows}
    coderepo_ids = {r["id"] for r in coderepo_rows}
    assert w_pid in webpage_ids and c_pid not in webpage_ids
    assert c_pid in coderepo_ids and w_pid not in coderepo_ids


# ---------------------------------------------------------------------------
# Unknown type → :Source:Unknown fallback (defensive)
# ---------------------------------------------------------------------------
def test_unknown_itemtype_tags_unknown(ne, test_prefix, cleanup):
    """Future / misrouted itemTypes get :Source:Unknown instead of silently
    polluting :Paper. Warning is logged (see test_t3_nonpaper for caplog)."""
    pid = f"{test_prefix}_UNK"
    ne.write_paper(_fake_paper(pid), item_type="audioRecording")
    assert _labels_of(ne, pid) == ["Source", "Unknown"]

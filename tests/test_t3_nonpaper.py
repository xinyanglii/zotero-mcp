"""T3 unit tests — label mapping + URL fetch helper + prompt hint.

Integration (live Neo4j) tests for label routing live in
``test_t3_labels.py``. These are pure-Python, no network / no Neo4j.
"""
from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch

import pytest

from zotero_mcp.kg_store import (
    ITEM_TYPE_TO_LABELS,
    _ALLOWED_SUB_LABELS,
    _label_hierarchy,
)
from zotero_mcp.ingest import (
    T3_NON_PAPER_TYPES,
    fetch_markdown_via_markitdown,
)
from zotero_mcp.extractor import (
    NON_PAPER_HINT,
    _NON_PAPER_ITEM_TYPES,
)


# ---------------------------------------------------------------------------
# ITEM_TYPE_TO_LABELS completeness + _label_hierarchy
# ---------------------------------------------------------------------------
def test_item_type_to_labels_covers_all_t3_types():
    """Every T3_NON_PAPER_TYPES entry must route somewhere."""
    for t in T3_NON_PAPER_TYPES:
        assert t in ITEM_TYPE_TO_LABELS, f"{t} missing from ITEM_TYPE_TO_LABELS"


def test_item_type_to_labels_paper_family_unchanged():
    """Existing paper types still map to :Source:Paper."""
    for t in ("journalArticle", "preprint", "conferencePaper",
              "thesis", "bookSection", "book", "report"):
        assert ITEM_TYPE_TO_LABELS[t] == ("Source", "Paper")


def test_item_type_to_labels_new_webpage_family():
    for t in ("webpage", "blogPost", "encyclopediaArticle", "forumPost"):
        assert ITEM_TYPE_TO_LABELS[t] == ("Source", "Webpage")


def test_item_type_to_labels_new_coderepo_family():
    for t in ("computerProgram", "software"):
        assert ITEM_TYPE_TO_LABELS[t] == ("Source", "CodeRepo")


def test_label_hierarchy_unknown_returns_unknown(caplog):
    """Unknown itemType → :Source:Unknown + warning logged."""
    import logging
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.kg_store"):
        out = _label_hierarchy("attachment")
    assert out == ("Source", "Unknown")
    assert any("unknown itemType" in r.message for r in caplog.records)


def test_allowed_sub_labels_contains_expected():
    """Any label we can ever emit must be in the closed set (Cypher injection guard)."""
    emitted = {ITEM_TYPE_TO_LABELS[t][1] for t in ITEM_TYPE_TO_LABELS}
    emitted.add("Unknown")   # from _label_hierarchy fallback
    assert emitted <= _ALLOWED_SUB_LABELS, (
        f"sub-labels {emitted - _ALLOWED_SUB_LABELS} not in _ALLOWED_SUB_LABELS")


# ---------------------------------------------------------------------------
# fetch_markdown_via_markitdown
# ---------------------------------------------------------------------------
def test_fetch_empty_url_returns_empty():
    assert fetch_markdown_via_markitdown("") == ""
    assert fetch_markdown_via_markitdown(None) == ""  # type: ignore


def test_fetch_whitespace_only_url_returns_empty():
    """Whitespace URL must NOT crash urllib.request.Request — returns ""."""
    assert fetch_markdown_via_markitdown("   ") == ""
    assert fetch_markdown_via_markitdown("\n\t ") == ""


def test_fetch_malformed_url_returns_empty():
    """Bad URL format (no scheme etc) → return empty, don't crash caller."""
    assert fetch_markdown_via_markitdown("not-a-url-at-all") == ""


class _FakeResponse:
    def __init__(self, body: bytes, ctype: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": ctype}
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._body


def test_fetch_success_returns_markdown(monkeypatch):
    """Happy path: urlopen returns HTML bytes → markitdown parses → md string."""
    html = b"<html><body><h1>Test Title</h1><p>Para one.</p></body></html>"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResponse(html),
    )
    md = fetch_markdown_via_markitdown("https://example.com/x")
    assert "Test Title" in md
    assert "Para one" in md


def test_fetch_403_returns_empty(monkeypatch):
    """HTTP 403 (Zhihu scenario) → empty string, no exception bubbled up."""
    def _urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://zhihu/x", 403, "Forbidden", {}, io.BytesIO(b""))
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    assert fetch_markdown_via_markitdown("https://zhuanlan.zhihu.com/p/1") == ""


def test_fetch_timeout_returns_empty(monkeypatch):
    def _urlopen(req, timeout=None):
        raise TimeoutError("slow upstream")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    assert fetch_markdown_via_markitdown("https://slow.example/x") == ""


def test_fetch_urlerror_returns_empty(monkeypatch):
    def _urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    assert fetch_markdown_via_markitdown("http://localhost:1/x") == ""


def test_fetch_uses_browser_ua(monkeypatch):
    """Confirm we send the browser UA (Wikipedia / SE need it)."""
    seen = {}
    def _urlopen(req, timeout=None):
        # req is a urllib.request.Request
        seen["ua"] = req.headers.get("User-agent", "")  # urllib normalizes capitalization
        return _FakeResponse(b"<html></html>")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    fetch_markdown_via_markitdown("https://example.com")
    assert "Mozilla" in seen["ua"], f"expected browser UA, got {seen['ua']!r}"


# ---------------------------------------------------------------------------
# NON_PAPER_HINT prompt injection via extract_structured (no Neo4j / no LLM)
# ---------------------------------------------------------------------------
def test_non_paper_item_types_consistent():
    """extractor._NON_PAPER_ITEM_TYPES should align with ingest's T3_NON_PAPER_TYPES
    (otherwise hints and whitelist drift apart silently)."""
    assert _NON_PAPER_ITEM_TYPES == T3_NON_PAPER_TYPES


def test_non_paper_hint_prepended_on_webpage(monkeypatch):
    """extract_structured with item_type_hint='webpage' includes NON_PAPER_HINT
    in the user message seen by the LLM call."""
    from zotero_mcp import extractor

    captured = {}
    def _fake_run(user_msg, figures=None):
        captured["user_msg"] = user_msg
        captured["figures"] = figures
        # Return a minimal valid JSON to short-circuit retries
        return ('{"paper_id":"X","title":"T","contribution_type":"method",'
                '"tldr":"","problem":"","methods_used":[],"methods_proposed":[],'
                '"methods_improved":[],"datasets":[],"concepts":[],'
                '"references":[]}',
                "mock-provider")
    monkeypatch.setattr(extractor, "_run_extract_call", _fake_run)
    paper, provider = extractor.extract_structured(
        "some content", title="T", paper_id="X",
        item_type_hint="webpage",
    )
    assert paper is not None
    assert provider == "mock-provider"
    assert NON_PAPER_HINT in captured["user_msg"]


def test_non_paper_hint_not_prepended_on_paper(monkeypatch):
    """Paper itemType → no hint appended (legacy behavior preserved)."""
    from zotero_mcp import extractor

    captured = {}
    def _fake_run(user_msg, figures=None):
        captured["user_msg"] = user_msg
        return ('{"paper_id":"X","title":"T","contribution_type":"method",'
                '"tldr":"","problem":"","methods_used":[],"methods_proposed":[],'
                '"methods_improved":[],"datasets":[],"concepts":[],'
                '"references":[]}',
                "mock")
    monkeypatch.setattr(extractor, "_run_extract_call", _fake_run)
    extractor.extract_structured(
        "paper content", title="T", paper_id="X",
        item_type_hint="journalArticle",
    )
    assert NON_PAPER_HINT not in captured["user_msg"]


def test_non_paper_hint_not_prepended_when_hint_is_none(monkeypatch):
    from zotero_mcp import extractor

    captured = {}
    def _fake_run(user_msg, figures=None):
        captured["user_msg"] = user_msg
        return ('{"paper_id":"X","title":"T","contribution_type":"method",'
                '"tldr":"","problem":"","methods_used":[],"methods_proposed":[],'
                '"methods_improved":[],"datasets":[],"concepts":[],'
                '"references":[]}',
                "mock")
    monkeypatch.setattr(extractor, "_run_extract_call", _fake_run)
    extractor.extract_structured("x", title="T", paper_id="X")
    assert NON_PAPER_HINT not in captured["user_msg"]

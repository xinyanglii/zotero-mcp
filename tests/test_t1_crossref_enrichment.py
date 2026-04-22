"""T1-c unit tests for crossref_enrich_refs.

Covers spec §4 failure state machine + §8 Finding #14 (mocked CrossRef
responses for each failure class). No Neo4j / no real CrossRef — pure
python + mocked urlopen.
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

# Import target module directly — it lives under scripts/, not src/
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import crossref_enrich_refs as cer  # noqa: E402


# ---------------------------------------------------------------------------
# title_similar — subset rescue + short-title fallback (Finding #6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,expected_min", [
    # identical
    ("Deep Learning", "Deep Learning", 1.0),
    # subset rescue: shorter's tokens ⊆ longer's
    ("Methods of Information Geometry",
     "Methods of Information Geometry to model complex shapes", 1.0),
    ("AlexNet",
     "AlexNet: ImageNet Classification with Deep Convolutional Neural Networks",
     1.0),
    # short title (< 4 tokens) token Jaccard
    ("BERT model", "BERT training", 0.33),  # 1/3
    # normal 3-shingle
    ("Information Geometry and Its Applications",
     "Computational Information Geometry and its Applications", 0.30),  # lower
])
def test_title_similar_behaviors(a, b, expected_min):
    sim = cer.title_similar(a, b)
    # For exact / subset cases, expected_min is exact; otherwise at least
    if expected_min == 1.0:
        assert sim == 1.0
    else:
        assert sim >= expected_min - 0.05, (a, b, sim)


def test_title_similar_unrelated():
    sim = cer.title_similar("Deep Learning Survey",
                             "Adaptive Control of Legged Robots")
    assert sim < 0.3


def test_title_similar_empty():
    assert cer.title_similar("", "abc") == 0.0
    assert cer.title_similar("abc", "") == 0.0


# ---------------------------------------------------------------------------
# _extract_first_author_lastname
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ay,expected", [
    ("Smith et al. 2020", "Smith"),
    ("Rao 1945", "Rao"),
    ("Amari, Shun-ichi 2016", "Amari"),
    ("Goodfellow, Bengio, Courville 2016", "Goodfellow"),
    ("", None),
    ("12345", None),  # only digits
])
def test_extract_first_author(ay, expected):
    assert cer._extract_first_author_lastname(ay) == expected


# ---------------------------------------------------------------------------
# classify — failure state machine (mocked crossref_query)
# ---------------------------------------------------------------------------
@pytest.fixture
def progress_db():
    """Fresh SQLite progress DB for each test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    # Point the module's constant at the tmp file
    orig_path = cer.PROGRESS_DB
    cer.PROGRESS_DB = Path(tmp.name)
    conn = cer._init_progress_db()
    # Ensure rate limiter exists (normally initialized in run())
    cer._limiter = cer.RateLimiter(rate_per_sec=100)  # fast for tests
    yield conn
    conn.close()
    cer.PROGRESS_DB = orig_path
    Path(tmp.name).unlink(missing_ok=True)


def _mock_items(*items):
    """Return a factory that crossref_query will "fetch"."""
    def _fake_query(title, author_year="", timeout=30, rows=3):
        return list(items)
    return _fake_query


def test_classify_accept_writes_done(progress_db):
    items = [{"DOI": "10.1234/TEST.ABCD",
              "title": ["Methods of Information Geometry"],
              "score": 25.5}]
    with patch.object(cer, "crossref_query", _mock_items(*items)):
        res = cer.classify(
            "Methods of Information Geometry", "Amari 2016",
            progress_db, "testref1", 0.5, 0.25,
        )
    assert res["status"] == "done"
    assert res["doi"] == "10.1234/test.abcd"  # normalized
    assert res["sim"] == 1.0


def test_classify_pending_review_gray_zone(progress_db):
    # Neither subset of the other + triggers short-title path (<4 tokens on
    # one side) → token Jaccard = 2/4 = 0.5... use 5-word vs 3-word pair so
    # one side is <4 AND neither is subset AND token-Jaccard lands in 0.25-0.5.
    # A = {the, statistical, learning, theory, book} (5 tokens)
    # B = {computational, learning, theory}         (3 tokens)
    # common = {learning, theory} = 2; union = 6; token-Jaccard = 2/6 = 0.33
    items = [{"DOI": "10.1234/X",
              "title": ["Computational Learning Theory"],
              "score": 15.0}]
    with patch.object(cer, "crossref_query", _mock_items(*items)):
        res = cer.classify(
            "The Statistical Learning Theory Book",
            "Vapnik 1998",
            progress_db, "testref2", 0.5, 0.25,
        )
    assert res["status"] == "pending_review"
    row = progress_db.execute(
        "SELECT status, title_sim FROM crossref_progress WHERE ref_id='testref2'"
    ).fetchone()
    assert row[0] == "pending_review"
    assert 0.25 <= row[1] < 0.5, f"sim={row[1]} not in gray zone"


def test_classify_no_match_below_ambig(progress_db):
    items = [{"DOI": "10.1234/X",
              "title": ["Adaptive Robot Control"],
              "score": 3.0}]
    with patch.object(cer, "crossref_query", _mock_items(*items)):
        res = cer.classify("Deep Learning for Physics", "Smith 2020",
                           progress_db, "testref3", 0.5, 0.25)
    assert res["status"] == "no_match"


def test_classify_empty_items(progress_db):
    with patch.object(cer, "crossref_query", _mock_items()):
        res = cer.classify("Obscure Title", "", progress_db, "testref4",
                           0.5, 0.25)
    assert res["status"] == "no_match"


def test_classify_http_429_retry_then_permanent_fail(progress_db):
    """First 2 HTTP 429 → status='retry', third attempt → 'permanent_fail'."""
    def _fake_429(title, author_year="", timeout=30, rows=3):
        raise urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, io.BytesIO(b""))

    with patch.object(cer, "crossref_query", _fake_429):
        res1 = cer.classify("T", "", progress_db, "tr5", 0.5, 0.25)
        assert res1["status"] == "retry"
        res2 = cer.classify("T", "", progress_db, "tr5", 0.5, 0.25)
        assert res2["status"] == "retry"
        res3 = cer.classify("T", "", progress_db, "tr5", 0.5, 0.25)
        assert res3["status"] == "permanent_fail"


def test_classify_timeout_goes_to_retry(progress_db):
    def _fake_timeout(title, author_year="", timeout=30, rows=3):
        raise TimeoutError("upstream slow")

    with patch.object(cer, "crossref_query", _fake_timeout):
        res = cer.classify("T", "", progress_db, "tr6", 0.5, 0.25)
    assert res["status"] == "retry"


def test_classify_urlerror_goes_to_retry(progress_db):
    def _fake_urlerror(title, author_year="", timeout=30, rows=3):
        raise urllib.error.URLError("connection refused")

    with patch.object(cer, "crossref_query", _fake_urlerror):
        res = cer.classify("T", "", progress_db, "tr7", 0.5, 0.25)
    assert res["status"] == "retry"


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
def test_rate_limiter_blocks_over_rate():
    import time as _time
    rl = cer.RateLimiter(rate_per_sec=5)
    t0 = _time.time()
    for _ in range(10):
        rl.acquire()
    # 10 tokens at 5/sec requires at least ~1s total wall
    elapsed = _time.time() - t0
    assert elapsed >= 0.9, f"elapsed {elapsed:.2f}s (expected >=0.9s)"

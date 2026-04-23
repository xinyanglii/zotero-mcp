"""Tests for scripts/canonicalize_entities.py::get_cached_judge.

Covers the T2 LLM judge cache lookup introduced 2026-04-23 to fix the
rerun regression where previously-rejected pairs (same_entity=0, not in
t2_edge_log) were re-LLM-judged every run.

Design constraints verified:
- pair_key-only lookup (no embed_model coupling — judgments are stable
  across embedding model changes)
- ``judge_err:`` rows must force retry (transient LLM failures are not
  real verdicts)
- return shape matches ``llm_judge`` return contract:
  ``{"same_entity": bool, "reason": str, "model": str}``
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import canonicalize_entities as t2  # noqa: E402


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE t2_llm_judge (
            pair_key     TEXT PRIMARY KEY,
            label_a      TEXT NOT NULL,
            label_b      TEXT NOT NULL,
            cosine       REAL,
            same_entity  INTEGER,
            reason       TEXT,
            model        TEXT,
            judged_at    REAL
        );
        """
    )
    return c


def _seed(conn, pk, same, reason, model="kimi-for-coding"):
    conn.execute(
        "INSERT INTO t2_llm_judge "
        "(pair_key, label_a, label_b, cosine, same_entity, reason, model, judged_at) "
        "VALUES (?, 'Method', 'Method', 0.87, ?, ?, ?, 0.0)",
        (pk, int(same), reason, model),
    )
    conn.commit()


def test_cache_miss_returns_none(conn):
    assert t2.get_cached_judge(conn, "A::B") is None


def test_cache_hit_same_entity_true(conn):
    _seed(conn, "A::B", same=True, reason="identical terms")
    v = t2.get_cached_judge(conn, "A::B")
    assert v == {
        "same_entity": True,
        "reason": "identical terms",
        "model": "kimi-for-coding",
    }


def test_cache_hit_same_entity_false_is_used(conn):
    """Confirmed-negative verdicts must be cache-served (the original bug)."""
    _seed(conn, "X::Y", same=False, reason="different domains")
    v = t2.get_cached_judge(conn, "X::Y")
    assert v is not None
    assert v["same_entity"] is False
    assert v["reason"] == "different domains"


def test_judge_err_row_forces_retry(conn):
    """Transient LLM call failures are stored as same_entity=0 + reason
    ``judge_err:...`` by ``llm_judge``'s exception handler. Those MUST be
    retried, not cache-served."""
    _seed(conn, "E::F", same=False,
          reason="judge_err:URLError:Connection refused")
    assert t2.get_cached_judge(conn, "E::F") is None


def test_judge_err_prefix_exact_match(conn):
    """Only the exact prefix ``judge_err:`` triggers retry. A row whose
    reason mentions 'judge_err' elsewhere is a real verdict."""
    _seed(conn, "G::H", same=False,
          reason="rejected; their doc says 'judge_err:429' but different domain")
    v = t2.get_cached_judge(conn, "G::H")
    assert v is not None
    assert v["same_entity"] is False


def test_null_reason_handled(conn):
    conn.execute(
        "INSERT INTO t2_llm_judge "
        "(pair_key, label_a, label_b, cosine, same_entity, reason, model, judged_at) "
        "VALUES ('N::M', 'Method', 'Method', 0.9, 1, NULL, 'kimi-for-coding', 0.0)"
    )
    conn.commit()
    v = t2.get_cached_judge(conn, "N::M")
    assert v is not None
    assert v["reason"] == ""
    assert v["same_entity"] is True


def test_return_shape_matches_llm_judge(conn):
    """``get_cached_judge`` return shape is a drop-in replacement for
    ``llm_judge`` return value. Keys must match exactly."""
    _seed(conn, "A::B", same=True, reason="r")
    v = t2.get_cached_judge(conn, "A::B")
    assert set(v.keys()) == {"same_entity", "reason", "model"}
    assert isinstance(v["same_entity"], bool)
    assert isinstance(v["reason"], str)
    assert isinstance(v["model"], str)


def test_cache_ignores_embed_model_variation(conn):
    """pair_key is the only lookup key — no coupling to embed_model. Verdict
    stored under an old embed_model run remains valid after embedder swap."""
    # Seed with one pair_key, query with same pair_key
    _seed(conn, "P::Q", same=True, reason="aliases", model="kimi-for-coding")
    assert t2.get_cached_judge(conn, "P::Q") is not None

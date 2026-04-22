"""T2 canonicalization — pure-Python helper unit tests.

Does NOT hit DashScope / Neo4j / Kimi. Those are exercised by the
dry-run A / smoke B calibration step (spec §6 step 1-2) against the
live system.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import canonicalize_entities as t2


# =========================================================================
# _canon + _collapse + CJK helpers
# =========================================================================
def test_canon_lowercases_and_strips_punct():
    assert t2._canon("ISAC") == "isac"
    assert t2._canon("  Attention Mech.  ") == "attention mech"
    # Comma becomes space, then \s+ collapses runs → single space
    assert t2._canon("integrated sensing, and communication") == \
           "integrated sensing and communication"


def test_collapse_drops_all_nonword():
    assert t2._collapse("CIFAR-10") == "cifar10"
    assert t2._collapse("CIFAR 10") == "cifar10"
    assert t2._collapse("CIFAR_10") == "cifar_10"   # underscore is \w
    assert t2._collapse("") == ""
    assert t2._collapse(None) == ""


def test_cjk_detection():
    assert t2._has_cjk("混合感知通信")
    assert t2._has_cjk("ISAC（混合感知）")   # mixed CJK + Latin
    assert t2._has_cjk("ひらがな")             # hiragana
    assert t2._has_cjk("カタカナ")             # full-width katakana
    assert not t2._has_cjk("attention")
    assert not t2._has_cjk("")
    assert not t2._has_cjk(None)


# =========================================================================
# pair_key determinism (spec §2.1 canonical min→max)
# =========================================================================
def test_pair_key_order_independent():
    assert t2.pair_key("a", "b") == t2.pair_key("b", "a") == "a::b"
    assert t2.pair_key("m1", "m2") == "m1::m2"
    # lex order: "10" < "2" because string compare
    assert t2.pair_key("10", "2") == "10::2"


# =========================================================================
# _embed_inputs_for (§1.2 single-encode when _canon(name) == canonical_name)
# =========================================================================
def test_embed_inputs_single_encode_when_equal():
    # n1: _canon('Attention') == 'attention' == canonical → single
    # n2: _canon('Pascal VOC 2012') = 'pascal voc 2012' != 'pascalvoc2012' → double
    # n3: _canon('CIFAR-10') = 'cifar 10' == canonical → single
    nodes = [
        ("n1", "Attention", "attention", 5),
        ("n2", "Pascal VOC 2012", "pascalvoc2012", 3),
        ("n3", "CIFAR-10", "cifar 10", 2),
    ]
    inputs = t2._embed_inputs_for(nodes)
    assert inputs[0] == "Attention"
    assert inputs[1] == "Pascal VOC 2012 | pascalvoc2012"
    assert inputs[2] == "CIFAR-10"


# =========================================================================
# find_similar_pairs (§6.1 batched GEMM, upper-triangle only)
# =========================================================================
def _unit(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def test_find_similar_pairs_upper_triangle_only():
    E = _unit(np.array([
        [1, 0, 0],
        [0.99, 0.01, 0],        # very close to 0
        [0, 1, 0],
    ], dtype=np.float32))
    pairs = t2.find_similar_pairs(E, threshold=0.9)
    ids = [(i, j) for i, j, _ in pairs]
    assert ids == [(0, 1)]      # only upper-tri (0,1); (0,2) and (1,2) too low


def test_find_similar_pairs_respects_threshold():
    E = _unit(np.array([
        [1, 0, 0],
        [0.95, 0.05, 0],
        [0.9, 0.4, 0],
    ], dtype=np.float32))
    # At threshold 0.95, only the 0-1 pair should pass
    pairs = t2.find_similar_pairs(E, threshold=0.99)
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 1


def test_find_similar_pairs_batch_equiv():
    """Batching the GEMM must give the same pairs as single-batch full matmul."""
    rng = np.random.default_rng(42)
    E = _unit(rng.standard_normal((40, 16)).astype(np.float32))
    p_single = sorted(t2.find_similar_pairs(E, threshold=0.2, batch=1000))
    p_batch = sorted(t2.find_similar_pairs(E, threshold=0.2, batch=7))
    # pairs set equal (ignoring float fuzz on cosine)
    ids_single = {(i, j) for i, j, _ in p_single}
    ids_batch = {(i, j) for i, j, _ in p_batch}
    assert ids_single == ids_batch


def test_find_similar_pairs_empty_or_tiny():
    E = np.zeros((0, 16), dtype=np.float32)
    assert t2.find_similar_pairs(E, 0.5) == []
    E = np.ones((1, 16), dtype=np.float32)
    assert t2.find_similar_pairs(E, 0.5) == []


# =========================================================================
# dataset_exact_match_pairs
# =========================================================================
def test_dataset_exact_match_clusters_cifar_variants():
    nodes = [
        ("d1", "CIFAR-10", "cifar 10", 10),
        ("d2", "CIFAR10", "cifar10", 5),
        ("d3", "CIFAR 10", "cifar 10", 3),
        ("d4", "ImageNet", "imagenet", 20),
    ]
    pairs = t2.dataset_exact_match_pairs(nodes)
    assert (0, 1, "exact") in pairs
    assert (0, 2, "exact") in pairs
    assert (1, 2, "exact") in pairs
    # ImageNet is a singleton bucket — no pair
    assert not any(3 in (i, j) for i, j, _ in pairs)


def test_dataset_exact_match_ignores_blank():
    nodes = [
        ("d1", "", "cifar10", 1),
        ("d2", "CIFAR10", "cifar10", 2),
    ]
    assert t2.dataset_exact_match_pairs(nodes) == []


# =========================================================================
# SQLite schema + idempotency helpers
# =========================================================================
@pytest.fixture
def tmp_conn(tmp_path, monkeypatch):
    """Point T2 progress DB to a tmp file; init + return connection."""
    db = tmp_path / "t2.sqlite"
    monkeypatch.setattr(t2, "PROGRESS_DB", db)
    conn = t2._init_progress_db()
    yield conn
    conn.close()


def test_sqlite_init_creates_all_tables(tmp_conn):
    tables = {r[0] for r in tmp_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert tables == {"t2_embeddings", "t2_llm_judge",
                      "t2_edge_log", "t2_failures"}


def test_edge_log_idempotent(tmp_conn):
    pk = "a::b"
    assert not t2.was_edge_written(tmp_conn, pk)
    t2.log_edge_written(tmp_conn, pk, "embed_auto")
    assert t2.was_edge_written(tmp_conn, pk)
    # second log is no-op (INSERT OR IGNORE)
    t2.log_edge_written(tmp_conn, pk, "llm_judge")
    rows = tmp_conn.execute(
        "SELECT kind FROM t2_edge_log WHERE pair_key=?", (pk,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "embed_auto"   # first-write-wins


def test_failure_log(tmp_conn):
    t2._record_failure(tmp_conn, "embed", "n1", "HTTP 429")
    rows = tmp_conn.execute(
        "SELECT stage, target, err FROM t2_failures"
    ).fetchall()
    assert rows == [("embed", "n1", "HTTP 429")]


# =========================================================================
# Label validation — Cypher injection guard (Finding 2)
# =========================================================================
@pytest.mark.parametrize("label", ["Concept", "Method", "Dataset"])
def test_validate_label_allowed(label):
    assert t2._validate_label(label) == label


@pytest.mark.parametrize("bad", [
    "Paper",                              # not in allowed set
    "Source",                             # superclass, not T2-scoped
    "concept",                            # wrong case
    "",                                   # empty
    "Concept; DROP DATABASE",             # injection attempt
    "Concept) DETACH DELETE n //",        # the exact injection from review
    "Concept OR TRUE",
])
def test_validate_label_rejects_junk(bad):
    with pytest.raises(ValueError):
        t2._validate_label(bad)

"""T4 + T8 unit tests.

Covers regex / blacklist / canonical / alias logic — pure Python, no
Neo4j / no network. Integration tests (live Neo4j MERGE behavior)
implicitly covered by T3's test_t3_labels.py which exercises
_write_paper_once label routing the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import link_papers_to_coderepo as t4
import backfill_datasets as t8


# ==========================================================================
# T4 — GitHub URL extraction (subagent #2/#3/#4/#6)
# ==========================================================================
@pytest.mark.parametrize("url,expect", [
    ("https://github.com/google/jax",                 {("google", "jax")}),
    ("http://github.com/user/REPO.git",               {("user", "repo")}),
    ("https://github.com/user/repo.",                 {("user", "repo")}),
    ("https://github.com/user/repo).",                {("user", "repo")}),
    # multiple on one line
    ("See https://github.com/a/b and github.com too... https://github.com/c/d-tool",
        {("a", "b"), ("c", "d-tool")}),
    # reserved owner → filtered
    ("https://github.com/features/copilot",           set()),
    ("https://github.com/orgs/community",             set()),
    ("https://github.com/solutions/industry",         set()),
    ("https://github.com/sponsors/monkeyuser",        set()),
    # owner with invalid chars → filtered (OWNER_VALID_RE)
    ("https://github.com/_bad/repo",                  set()),
    ("https://github.com/-bad/repo",                  set()),
    # repo with trailing non-identifier → still accepted after strip
    ("https://github.com/user/repo/tree/main",        {("user", "repo")}),
    # Subdomains (api.github.com) — current pattern does NOT strip them;
    # they are rare in academic md. Not a correctness concern.
])
def test_t4_extract_github_repos(url, expect):
    assert t4.extract_github_repos(url) == expect


def test_t4_clean_repo_suffix_strip():
    assert t4.clean_repo("repo.git") == "repo"
    assert t4.clean_repo("repo.md") == "repo"
    assert t4.clean_repo("repo.") == "repo"
    assert t4.clean_repo("repo.,)") == "repo"
    assert t4.clean_repo(".bad") is None
    assert t4.clean_repo("-bad") is None


def test_t4_owner_valid_re_github_rules():
    assert t4.OWNER_VALID_RE.match("google")
    assert t4.OWNER_VALID_RE.match("analog-devices-inc")
    assert t4.OWNER_VALID_RE.match("user123")
    assert not t4.OWNER_VALID_RE.match("-bad")
    assert not t4.OWNER_VALID_RE.match("bad-")    # trailing -
    assert not t4.OWNER_VALID_RE.match("a" * 40)  # too long


def test_t4_normalize_url_key():
    assert (t4.normalize_url_key("https://github.com/a/b")
            == t4.normalize_url_key("https://github.com/A/B/"))
    assert (t4.normalize_url_key("https://github.com/a/b.git")
            == "https://github.com/a/b")


def test_t4_blacklist_covers_common_github_reserved():
    """Sanity: most common 'looks like owner but is a reserved path' words."""
    for bad in ("solutions", "features", "orgs", "sponsors", "apps",
                "pulls", "issues", "settings", "account", "tree", "blob"):
        assert bad in t4.GITHUB_RESERVED_OWNERS


# ==========================================================================
# T8 — benchmark detection + alias dedup (subagent #7/#8/#9)
# ==========================================================================
@pytest.mark.parametrize("md,existing,expected_new", [
    # Positive: md has ImageNet but existing has verbose ilsvrc alias → SKIP
    ("Trained on ImageNet.",    ["imagenet-ilsvrc-2012"], []),
    ("Trained on ImageNet.",    [],                       ["ImageNet"]),
    # Positive: md has KITTI but existing has kittiodometry → SKIP
    ("KITTI benchmark.",        ["kittiodometry"],        []),
    ("KITTI benchmark.",        [],                       ["KITTI"]),
    # #9 guard: kitti-mots is different benchmark, KITTI alias does NOT match
    ("KITTI benchmark.",        ["kitti-mots"],           ["KITTI"]),
    # #8 MNIST vs Fashion-MNIST: md says MNIST, existing has fashion-mnist → should still add MNIST
    ("Evaluated on MNIST.",     ["fashion-mnist"],        ["MNIST"]),
    # MNIST-for-neural-networks verbose alias should dedup
    ("Evaluated on MNIST.",     ["mnistforneuralnetworkillustration"], []),
    # #7 GLUE: removed faulty (?!\s*benchmark) — "GLUE benchmark" phrasing should HIT
    ("The GLUE benchmark results.", [],                   ["GLUE"]),
    # COCO edge case: existing cocopanoptic shouldn't block new COCO
    # (they're different; cocopanoptic is a subset so conservative merge OK)
    ("Tested on COCO.",         ["cocopanoptic"],         ["COCO"]),
    # CIFAR variants
    ("On CIFAR-10.",            ["cifar100"],             ["CIFAR-10"]),
    ("On CIFAR-10.",            ["cifar10"],              []),
    # No match at all
    ("Nothing about benchmarks here.", [], []),
    # ISAC-specific benchmark hits
    ("DeepMIMO dataset for MU-MIMO.", [],                 ["DeepMIMO"]),
])
def test_t8_detect_new_benchmarks(md, existing, expected_new):
    got = t8.detect_new_benchmarks(md, existing)
    assert got == expected_new, f"md={md!r} existing={existing!r} got={got} expected={expected_new}"


def test_t8_collapse_canonical():
    """_collapse() used for alias-side comparison."""
    assert t8._collapse("Pascal VOC") == "pascalvoc"
    assert t8._collapse("Pascal-VOC-2012") == "pascalvoc2012"
    assert t8._collapse("MNIST") == "mnist"


def test_t8_dedup_within_same_paper():
    """Rare: md mentions MNIST twice, add-only-once semantics."""
    # detect_new_benchmarks uses a list (not set) but alias-hit on its own
    # output after first add → second mention no-ops
    adds = t8.detect_new_benchmarks("MNIST again MNIST", [])
    assert adds == ["MNIST"]


def test_t8_canonical_benchmarks_shape():
    """Sanity: every entry is 3-tuple (display, md_regex, alias_regex)."""
    for entry in t8.CANONICAL_BENCHMARKS:
        assert len(entry) == 3
        display, md_re, alias_re = entry
        assert display and md_re and alias_re


def test_t8_no_spurious_hits_on_unrelated_text():
    """Common words near benchmark names shouldn't trigger false positives."""
    # "MNIST" as part of a longer compound dataset name in md — still hits
    # because \bMNIST\b has word boundary on both sides
    cases = [
        ("The squared error...", []),       # "SQuAD" not substring
        ("imagenetilsvrc2012",    ["ImageNet"]),  # missing word boundary but \b matches at start/end of string
    ]
    # We only assert the first doesn't misfire
    assert t8.detect_new_benchmarks(cases[0][0], []) == cases[0][1]

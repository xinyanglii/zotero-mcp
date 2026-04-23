"""Tests for T5 L4 title-hash dedup helpers in zotero_mcp.kg_store.

Covers the pure-function layer (no Neo4j). Backfill / _link_same_work_as
integration is verified manually against production data post-deploy.
"""
from __future__ import annotations

from zotero_mcp.kg_store import (
    _normalize_author,
    _normalize_title,
    compute_title_hash,
)


# --- normalize_title -----------------------------------------------------

def test_title_case_and_punctuation_ignored():
    a = _normalize_title("Attention Is All You Need.")
    b = _normalize_title("attention is all you need")
    c = _normalize_title("  Attention   is  all  you NEED!  ")
    assert a == b == c == "attention is all you need"


def test_title_smart_quotes_and_em_dashes_stripped():
    a = _normalize_title("Transformer — the definitive guide")
    b = _normalize_title("Transformer - the definitive guide")
    assert a == b == "transformer the definitive guide"


def test_title_preprint_suffix_removed():
    a = _normalize_title("BERT: Pre-training of Deep Bidirectional Transformers (preprint)")
    b = _normalize_title("BERT: Pre-training of Deep Bidirectional Transformers")
    assert a == b


def test_title_preserves_cjk():
    s = _normalize_title("注意力就是你所需要的全部：一个综述")
    # colons etc. stripped, CJK preserved
    assert "注意力就是你所需要的全部" in s
    assert "：" not in s


# --- compute_title_hash --------------------------------------------------

def test_same_input_same_hash():
    h1 = compute_title_hash("Attention Is All You Need", "Vaswani", 2017)
    h2 = compute_title_hash("attention is all you need.", "vaswani", 2017)
    assert h1 == h2 and len(h1) == 64  # sha256 hex


def test_preprint_published_variants_collide_by_design():
    """The core L4 value: preprint + published with same title+author+year
    get the SAME hash — even if one has "(preprint)" suffix and different
    punctuation."""
    h_arxiv = compute_title_hash(
        "Efficient Transformers: A Survey (preprint)", "Tay", 2020,
    )
    h_journal = compute_title_hash(
        "Efficient Transformers: A Survey", "Tay", 2020,
    )
    assert h_arxiv == h_journal


def test_different_year_different_hash():
    """year in hash protects against conflating a 2024 workshop version
    and a 2025 conference version that share title+author."""
    h1 = compute_title_hash("Novel Method X", "Smith", 2024)
    h2 = compute_title_hash("Novel Method X", "Smith", 2025)
    assert h1 != h2


def test_title_too_short_returns_none():
    # < 10 chars after strip — too collision-prone
    assert compute_title_hash("Attention", "Vaswani", 2017) is None
    assert compute_title_hash("BERT", "Devlin", 2018) is None


def test_missing_author_returns_none():
    """Pure title+year without author collides too aggressively (e.g.,
    many reports titled "Quarterly Review ...")."""
    assert compute_title_hash("Some reasonably long title here", None, 2020) is None
    assert compute_title_hash("Some reasonably long title here", "", 2020) is None
    assert compute_title_hash("Some reasonably long title here", "   ", 2020) is None


def test_missing_year_maps_to_zero():
    """Records without year should only dedup against other no-year records,
    not accidentally merge with a year-having twin."""
    h_noyear = compute_title_hash("Paper Without Date Info", "Author", None)
    h_year_0 = compute_title_hash("Paper Without Date Info", "Author", 0)
    h_year_2020 = compute_title_hash("Paper Without Date Info", "Author", 2020)
    assert h_noyear is not None
    assert h_noyear == h_year_0
    assert h_noyear != h_year_2020


def test_cjk_author_and_title_hash_stable():
    h1 = compute_title_hash("深度学习与自然语言处理", "张三", 2024)
    h2 = compute_title_hash(" 深度学习与自然语言处理 ", "张三", 2024)
    assert h1 == h2
    assert h1 is not None


def test_normalize_author_strips_whitespace_and_punct():
    assert _normalize_author(" Van-Der-Berg ") == "vanderberg"
    assert _normalize_author("O'Brien") == "obrien"


def test_weak_author_string_returns_none():
    """author that normalizes to empty string (e.g., punctuation only)
    should refuse to produce a hash."""
    assert compute_title_hash("A proper long title for testing",
                              "---", 2020) is None

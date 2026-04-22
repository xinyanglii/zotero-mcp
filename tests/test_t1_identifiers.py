"""Unit tests for T1 DOI / arXiv parsing helpers.

Covers spec §1 (normalization) + §2.1 (_parse_paper_ids). Runs without
any external dependency (Neo4j / Zotero / CrossRef).
"""
from __future__ import annotations

import pytest

from zotero_mcp.ingest import (
    normalize_doi,
    normalize_arxiv,
    _parse_paper_ids,
)


# ---------------------------------------------------------------------------
# normalize_doi
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    # canonical
    ("10.1109/TWC.2023.3274571", "10.1109/twc.2023.3274571"),
    # URL prefix variants
    ("https://doi.org/10.1109/TWC.2023.3274571", "10.1109/twc.2023.3274571"),
    ("http://dx.doi.org/10.1109/ISWCS.2018.8491078", "10.1109/iswcs.2018.8491078"),
    ("doi:10.1017/CBO9780511569920.003", "10.1017/cbo9780511569920.003"),
    ("DOI: 10.48550/arXiv.2501.18799", "10.48550/arxiv.2501.18799"),
    # whitespace + case
    ("  10.23919/JCC.2023.03.003  ", "10.23919/jcc.2023.03.003"),
    # invalid
    ("", None),
    (None, None),
    ("not a doi", None),
    ("10.foo", None),  # no slash
    ("invalid/10.1109/abc", None),  # doesn't start with 10.
    (12345, None),    # non-string
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


# ---------------------------------------------------------------------------
# normalize_arxiv
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    # common Zotero extra patterns (from real library sampling)
    ("arXiv:2501.18799 [eess]", "2501.18799"),
    ("arXiv:2311.17630 [cs, eess]", "2311.17630"),
    ("arXiv:2503.06629 [cs]", "2503.06629"),
    ("arXiv:1904.05835 [cs]", "1904.05835"),
    ("arXiv:2505.13461 [cs]", "2505.13461"),
    # with version suffix
    ("arXiv:2501.18799v2", "2501.18799"),
    ("arXiv:2501.18799v12 [eess]", "2501.18799"),
    # multi-line extra (citations prefix + arxiv line)
    ("0 citations (Semantic Scholar/arXiv) [2023-08-28]\narXiv:2302.08444 [eess]",
     "2302.08444"),
    # DOI alias form (10.48550/arXiv prefix)
    ("10.48550/arXiv.2501.18799", "2501.18799"),
    ("10.48550/arxiv.1410.5846", "1410.5846"),
    # bare id
    ("2501.18799", "2501.18799"),
    # case insensitivity
    ("ARXIV:2501.18799", "2501.18799"),
    # 4-digit suffix (older ids)
    ("arXiv:1410.5846", "1410.5846"),
    # invalid
    ("", None),
    (None, None),
    ("just prose", None),
    ("arXiv: bad-id", None),
    ("Conference Name: foo", None),
    (12345, None),
])
def test_normalize_arxiv(raw, expected):
    assert normalize_arxiv(raw) == expected


# ---------------------------------------------------------------------------
# _parse_paper_ids — fixture from real Zotero item.data shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("item_data,expected", [
    # Case 1: plain journalArticle with DOI, no arXiv
    (
        {"DOI": "10.1109/TWC.2023.3274571",
         "extra": "0 citations (Semantic Scholar/DOI) [2023-05-20]\n"
                   "Conference Name: IEEE Transactions on Wireless Communications"},
        {"doi": "10.1109/twc.2023.3274571", "arxiv_id": None},
    ),
    # Case 2: preprint where DOI is arxiv alias AND extra has arXiv line
    (
        {"DOI": "10.48550/arXiv.2501.18799",
         "extra": "arXiv:2501.18799 [eess]"},
        {"doi": "10.48550/arxiv.2501.18799", "arxiv_id": "2501.18799"},
    ),
    # Case 3: preprint with DOI alias but no extra arxiv line — still mirror
    (
        {"DOI": "10.48550/arXiv.2302.08444",
         "extra": ""},
        {"doi": "10.48550/arxiv.2302.08444", "arxiv_id": "2302.08444"},
    ),
    # Case 4: extra has arXiv, no DOI field
    (
        {"DOI": "",
         "extra": "arXiv:2503.06629 [cs]"},
        {"doi": None, "arxiv_id": "2503.06629"},
    ),
    # Case 5: totally empty — should return all None
    (
        {"DOI": "", "extra": ""},
        {"doi": None, "arxiv_id": None},
    ),
    # Case 6: missing fields entirely
    (
        {},
        {"doi": None, "arxiv_id": None},
    ),
])
def test_parse_paper_ids(item_data, expected):
    assert _parse_paper_ids(item_data) == expected

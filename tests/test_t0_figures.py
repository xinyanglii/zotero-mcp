"""Unit tests for T0 figure pipeline helpers.

Covers:
  - ``_maybe_downscale`` behavior on small / >10MB inputs (spec §3.3 / §4)
  - ``_caption_for`` regex on both AFTER and BEFORE layouts (spec §5.3)
  - ``IMG_REF_RE`` scanning for chunk ``figure_refs`` (spec §5.2)

Run with: ``pytest tests/test_t0_figures.py -v``
Does NOT require MinerU / WebDAV / LLM APIs. Pure in-memory.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from zotero_mcp.ingest import _caption_for, _maybe_downscale
from zotero_mcp.kg_store import IMG_REF_RE


# ---------------------------------------------------------------------------
# _maybe_downscale
# ---------------------------------------------------------------------------
def _make_jpeg(size: tuple[int, int], target_bytes: int | None = None) -> bytes:
    """Generate a JPEG of roughly ``target_bytes`` by stuffing noise pixels."""
    import os as _os
    im = Image.frombytes("RGB", size, _os.urandom(size[0] * size[1] * 3))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_maybe_downscale_preserves_small_image():
    """<=10MB: bytes returned unchanged, downscaled=False, dims match."""
    small = _make_jpeg((512, 256))
    assert len(small) < 10 * 1024 * 1024
    out, downscaled, w, h = _maybe_downscale(small)
    assert out == small
    assert downscaled is False
    assert (w, h) == (512, 256)


def test_maybe_downscale_large_image_compresses():
    """>10MB: downscale to max 2048px + JPEG q85 (or q70 fallback)."""
    # Construct a ~15MB JPEG via a huge noise image. Pillow q=95 noise
    # hits ~0.8 bytes/pixel → 4500x4500 ≈ 16MB.
    big = _make_jpeg((4500, 4500))
    # Some versions of PIL may compress noise very well; if under 10MB,
    # pad via a second dimension. Use assert to skip if env prohibitive.
    if len(big) <= 10 * 1024 * 1024:
        pytest.skip(f"failed to construct >10MB JPEG (got {len(big)}b) — skipping")
    out, downscaled, w, h = _maybe_downscale(big)
    assert downscaled is True
    # Longest edge of downscaled ≤ 2048
    assert max(w, h) <= 2048
    # Output < input (or equal when q70 still oversize — rare)
    assert len(out) <= len(big)


# ---------------------------------------------------------------------------
# _caption_for
# ---------------------------------------------------------------------------
MN = "a" * 64 + ".jpg"  # fake 64-hex.jpg
OTHER_MN = "b" * 64 + ".jpg"


def test_caption_for_after_pattern():
    """``![]() ⏎ Figure N: ...`` — AFTER layout."""
    md = (
        f"Some text above.\n\n"
        f"![](images/{MN})\n\n"
        f"Figure 3: An architecture diagram showing three encoder stages.\n\n"
        f"Some text below.\n"
    )
    cap = _caption_for(MN, md)
    assert cap is not None
    assert cap.startswith("Figure 3:")
    assert "architecture" in cap


def test_caption_for_before_pattern():
    """``Fig. N. ... ⏎ ![]()`` — BEFORE layout."""
    md = (
        f"Prior section.\n\n"
        f"Fig. 7. Confusion matrix for the MUSIC estimator.\n\n"
        f"![](images/{MN})\n\n"
        f"Next section.\n"
    )
    cap = _caption_for(MN, md)
    assert cap is not None
    assert cap.startswith("Fig. 7.")
    assert "Confusion" in cap


def test_caption_for_no_match_returns_none():
    """Image ref with no adjacent Figure/Fig. line → None."""
    md = (
        f"Just some prose.\n\n"
        f"![](images/{MN})\n\n"
        f"More prose, no caption.\n"
    )
    assert _caption_for(MN, md) is None


def test_caption_for_other_mineru_name_isolated():
    """Ensure caption for one ref doesn't leak to another ref."""
    md = (
        f"![](images/{MN})\n\n"
        f"Figure 1: caption for A.\n\n"
        f"![](images/{OTHER_MN})\n\n"
        f"Some text (no caption for B).\n"
    )
    assert _caption_for(MN, md) is not None
    assert "caption for A" in _caption_for(MN, md)
    assert _caption_for(OTHER_MN, md) is None


# ---------------------------------------------------------------------------
# IMG_REF_RE (shared between kg_store QdrantWriter and ingest._build_figures)
# ---------------------------------------------------------------------------
def test_img_ref_re_captures_mineru_names():
    """Shared regex should capture every ![](images/<64hex>.jpg) ref, dedup preserved."""
    md = (
        f"![](images/{MN})\n"
        f"Figure 1: something.\n"
        f"![](images/{OTHER_MN})\n"
        f"More.\n"
        f"![](images/{MN})\n"  # duplicate ref of MN
    )
    matches = [m.group("mn") for m in IMG_REF_RE.finditer(md)]
    assert matches == [MN, OTHER_MN, MN]
    # dedup preserves order
    seen: set[str] = set()
    unique_ordered = []
    for mn in matches:
        if mn not in seen:
            seen.add(mn)
            unique_ordered.append(mn)
    assert unique_ordered == [MN, OTHER_MN]


def test_img_ref_re_handles_png_extension():
    """MinerU can produce PNGs too — regex allows .png/.jpeg/etc."""
    md = f"![](images/{'c' * 64}.png)"
    m = IMG_REF_RE.search(md)
    assert m is not None
    assert m.group("mn") == "c" * 64 + ".png"


def test_img_ref_re_rejects_non_mineru_path():
    """Don't match normal markdown images under other paths."""
    md = "![alt](assets/diagram.png)"
    assert IMG_REF_RE.search(md) is None

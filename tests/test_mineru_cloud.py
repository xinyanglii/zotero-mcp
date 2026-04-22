"""Unit + integration tests for MinerU Cloud API Tier 2 fallback.

Unit tests use mocked requests (no network). Integration test (marked
``@pytest.mark.integration``) hits real mineru.net — skipped by default;
run with ``pytest -m integration`` + ``MINERU_CLOUD_TOKEN`` set.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp.mineru_parser import (
    convert_pdf_mineru_cloud_with_images,
    convert_pdf_mineru_with_images,
)


def _build_result_zip(md: str = "# Test\n\nHello",
                      imgs: dict[str, bytes] | None = None) -> bytes:
    """Build a MinerU result zip (full.md + images/*)."""
    imgs = imgs or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("full.md", md)
        zf.writestr("layout.json", "{}")  # mimic real zip
        for name, data in imgs.items():
            zf.writestr(f"images/{name}", data)
    return buf.getvalue()


@pytest.fixture
def tmp_pdf(tmp_path):
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4\nmock content\n%%EOF")
    return p


# ---------------------------------------------------------------------------
# convert_pdf_mineru_cloud_with_images — mocked full flow
# ---------------------------------------------------------------------------
def test_cloud_full_flow_happy_path(tmp_pdf, monkeypatch):
    """End-to-end mock: batch→PUT→poll→zip → returns (md, images)."""
    zip_bytes = _build_result_zip(
        md="# Paper Title\n\n![](images/abc.jpg)\n",
        imgs={"abc.jpg": b"\xff\xd8\xff\xe0jpg"},
    )

    def fake_post(url, json=None, **kw):
        assert url.endswith("/file-urls/batch")
        assert kw["headers"]["Authorization"] == "Bearer tok123"
        r = MagicMock(); r.status_code = 200
        r.json.return_value = {
            "code": 0, "data": {
                "batch_id": "B1",
                "file_urls": ["https://oss.example/presigned"],
            }}
        r.raise_for_status = MagicMock()
        return r

    put_calls = []
    def fake_put(url, data=None, **kw):
        put_calls.append(url)
        r = MagicMock(); r.status_code = 200
        r.raise_for_status = MagicMock()
        return r

    poll_counter = {"n": 0}
    def fake_get(url, headers=None, **kw):
        r = MagicMock(); r.status_code = 200
        r.raise_for_status = MagicMock()
        if "extract-results/batch" in url:
            poll_counter["n"] += 1
            # First two polls: running; third: done
            if poll_counter["n"] < 3:
                r.json.return_value = {"data": {"extract_result": [
                    {"state": "running", "extract_progress": {"extracted_pages": 5}}]}}
            else:
                r.json.return_value = {"data": {"extract_result": [
                    {"state": "done", "full_zip_url": "https://cdn/result.zip"}]}}
        elif "cdn" in url:
            r.content = zip_bytes
        else:
            raise AssertionError(f"unexpected GET {url}")
        return r

    with patch("requests.post", side_effect=fake_post), \
         patch("requests.put", side_effect=fake_put), \
         patch("requests.get", side_effect=fake_get):
        md, images = convert_pdf_mineru_cloud_with_images(
            tmp_pdf, token="tok123", poll_interval=0,  # no sleep in tests
        )
    assert "Paper Title" in md
    assert "abc.jpg" in images
    assert images["abc.jpg"] == b"\xff\xd8\xff\xe0jpg"
    assert len(put_calls) == 1
    assert "presigned" in put_calls[0]


def test_cloud_missing_token(tmp_pdf, monkeypatch):
    monkeypatch.delenv("MINERU_CLOUD_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MINERU_CLOUD_TOKEN"):
        convert_pdf_mineru_cloud_with_images(tmp_pdf)


def test_cloud_batch_request_reject(tmp_pdf):
    """API returns code!=0 → RuntimeError."""
    bad_resp = MagicMock(status_code=200)
    bad_resp.json.return_value = {"code": 4001, "msg": "auth expired"}
    bad_resp.raise_for_status = MagicMock()
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(RuntimeError, match="rejected"):
            convert_pdf_mineru_cloud_with_images(tmp_pdf, token="tok")


def test_cloud_oss_upload_403_raises(tmp_pdf):
    """OSS PUT returns non-2xx → RuntimeError caller can catch."""
    post_resp = MagicMock(status_code=200)
    post_resp.json.return_value = {"code": 0, "data": {
        "batch_id": "B2", "file_urls": ["https://oss/u"]}}
    post_resp.raise_for_status = MagicMock()

    import requests as _r
    def _raise_put(url, data=None, **kw):
        # Simulate 403 from OSS
        resp = MagicMock(status_code=403)
        resp.raise_for_status.side_effect = _r.HTTPError("403")
        return resp

    with patch("requests.post", return_value=post_resp), \
         patch("requests.put", side_effect=_raise_put):
        with pytest.raises(RuntimeError, match="OSS upload"):
            convert_pdf_mineru_cloud_with_images(tmp_pdf, token="tok")


def test_cloud_failed_state_raises(tmp_pdf):
    """Poll returns state=failed → RuntimeError with err_msg."""
    post_resp = MagicMock(status_code=200)
    post_resp.json.return_value = {"code": 0, "data": {
        "batch_id": "B3", "file_urls": ["https://oss/u"]}}
    post_resp.raise_for_status = MagicMock()
    put_resp = MagicMock(status_code=200); put_resp.raise_for_status = MagicMock()
    get_resp = MagicMock(status_code=200); get_resp.raise_for_status = MagicMock()
    get_resp.json.return_value = {"data": {"extract_result": [
        {"state": "failed", "err_msg": "too many pages"}]}}
    with patch("requests.post", return_value=post_resp), \
         patch("requests.put", return_value=put_resp), \
         patch("requests.get", return_value=get_resp):
        with pytest.raises(RuntimeError, match="too many pages"):
            convert_pdf_mineru_cloud_with_images(tmp_pdf, token="tok",
                                                  poll_interval=0)


# ---------------------------------------------------------------------------
# convert_pdf_mineru_with_images — three-tier orchestration
# ---------------------------------------------------------------------------
def test_three_tier_self_hosted_wins_when_api_url_set(tmp_pdf, monkeypatch):
    """When MINERU_API_URL is set and succeeds, never calls cloud/CLI."""
    monkeypatch.setenv("MINERU_API_URL", "http://localhost:8765")
    monkeypatch.setenv("MINERU_CLOUD_TOKEN", "tok")
    cloud_called = {"n": 0}
    with patch("zotero_mcp.mineru_parser.convert_pdf_mineru_api_with_images",
                return_value=("# self-hosted\n", {"x.jpg": b"img"})), \
         patch("zotero_mcp.mineru_parser.convert_pdf_mineru_cloud_with_images",
                side_effect=lambda *a, **k: cloud_called.update(n=cloud_called["n"]+1)):
        md, imgs = convert_pdf_mineru_with_images(tmp_pdf)
    assert "self-hosted" in md
    assert cloud_called["n"] == 0


def test_three_tier_falls_through_to_cloud_when_self_hosted_fails(tmp_pdf, monkeypatch):
    """Self-hosted raises → cloud tier gets the chance."""
    monkeypatch.setenv("MINERU_API_URL", "http://localhost:8765")
    monkeypatch.setenv("MINERU_CLOUD_TOKEN", "tok")
    with patch("zotero_mcp.mineru_parser.convert_pdf_mineru_api_with_images",
                side_effect=RuntimeError("self-hosted dead")), \
         patch("zotero_mcp.mineru_parser.convert_pdf_mineru_cloud_with_images",
                return_value=("# cloud\n", {"y.jpg": b"cloudimg"})):
        md, imgs = convert_pdf_mineru_with_images(tmp_pdf)
    assert "cloud" in md
    assert "y.jpg" in imgs


def test_three_tier_skips_self_hosted_if_url_missing(tmp_pdf, monkeypatch):
    """No MINERU_API_URL → cloud is Tier 1."""
    monkeypatch.delenv("MINERU_API_URL", raising=False)
    monkeypatch.setenv("MINERU_CLOUD_TOKEN", "tok")
    sh_called = {"n": 0}
    with patch("zotero_mcp.mineru_parser.convert_pdf_mineru_api_with_images",
                side_effect=lambda *a, **k: sh_called.update(n=1)), \
         patch("zotero_mcp.mineru_parser.convert_pdf_mineru_cloud_with_images",
                return_value=("# cloud direct\n", {})):
        md, imgs = convert_pdf_mineru_with_images(tmp_pdf)
    assert sh_called["n"] == 0
    assert "cloud direct" in md


# ---------------------------------------------------------------------------
# Live integration — hits real mineru.net, slow, skipped by default
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MINERU_CLOUD_TOKEN"),
    reason="MINERU_CLOUD_TOKEN not set",
)
def test_cloud_live_integration():
    """Real end-to-end against mineru.net using a small sample PDF."""
    pdf = "/tmp/sample_76AYWLEU.pdf"
    if not os.path.exists(pdf):
        pytest.skip("sample PDF missing")
    md, images = convert_pdf_mineru_cloud_with_images(pdf)
    assert len(md) > 500, "md looks suspiciously short"
    # MinerU returns 64-hex filenames — verify at least one matches pattern
    import re
    hex64 = re.compile(r"^[a-f0-9]{64}\.(jpg|png|jpeg)$")
    sha_names = [n for n in images if hex64.match(n)]
    assert sha_names, f"expected SHA-named images, got {list(images)[:3]}"

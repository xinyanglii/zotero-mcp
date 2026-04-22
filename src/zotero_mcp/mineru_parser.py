"""
MinerU-based PDF parser with two modes:

1. **API mode** (preferred for batch work): POST to a long-running
   ``mineru-api`` server. Models stay hot → saves ~10s/paper cold-start.
   Enable via ``MINERU_API_URL`` env (e.g. ``http://127.0.0.1:8765``).

2. **CLI mode** (fallback, single-shot): shell out to ``mineru`` CLI. Safe
   for ad-hoc one-off parses but reloads models every call.

Both return the same markdown string. The ``convert_pdf_mineru`` entry point
auto-picks API mode when ``MINERU_API_URL`` is set and the service is alive.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Tier 1 circuit breaker state — process-local, reset on restart. If the
# self-hosted mineru-api daemon is dead (systemctl --user stop, container
# crash, etc.) every ingest_one call would eat 20-30s of HTTP timeout
# before falling through. ``MINERU_TIER1_FAIL_LIMIT`` consecutive failures
# flips a flag that skips Tier 1 for the rest of this process. Workaround
# if restarted incorrectly: bump env to higher number or fix the daemon.
_TIER1_FAIL_LIMIT = int(os.environ.get("MINERU_TIER1_FAIL_LIMIT", "3"))
_tier1_failures = 0


def _tier1_tripped() -> bool:
    return _tier1_failures >= _TIER1_FAIL_LIMIT


def _tier1_record_failure() -> None:
    global _tier1_failures
    _tier1_failures += 1
    if _tier1_failures == _TIER1_FAIL_LIMIT:
        logger.error(
            "mineru Tier 1 (self-hosted) circuit breaker tripped after %d "
            "consecutive failures — skipping for the rest of this process",
            _tier1_failures,
        )


def _mineru_bin() -> str:
    env_override = os.getenv("MINERU_BIN", "").strip()
    if env_override and Path(env_override).is_file():
        return env_override
    # PATH first (typical case: user is in the right env)
    bin_path = shutil.which("mineru")
    if bin_path:
        return bin_path
    # Fallback: look next to the current Python (conda/venv entrypoint)
    import sys
    candidate = Path(sys.executable).parent / "mineru"
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(
        "`mineru` binary not found. Install with `pip install mineru` "
        "and run `mineru-models-download -s huggingface -m all` once."
    )


def _find_output_md(out_dir: Path, pdf_stem: str) -> Path | None:
    """MinerU writes output under `<out_dir>/<pdf_stem>/<method>/<pdf_stem>.md`.

    We don't hardcode `method` — walk and take the first .md file that matches
    the expected naming. Fall back to any *.md file under the output dir.
    """
    preferred = [p for p in out_dir.rglob(f"{pdf_stem}.md")]
    if preferred:
        return sorted(preferred, key=lambda p: len(p.parts))[0]
    any_md = list(out_dir.rglob("*.md"))
    return any_md[0] if any_md else None


# ==========================================================================
# Multipart-form builder (no external deps) for POSTing to /file_parse
# ==========================================================================
def _multipart_body(
    fields: dict[str, str], files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    """Return (body_bytes, content_type) for an RFC7578 multipart/form-data POST."""
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(value.encode())
        buf.write(b"\r\n")
    for name, (fname, data, ctype) in files.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
        )
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def convert_pdf_mineru_api(
    pdf_path: str | Path,
    *,
    api_url: str | None = None,
    backend: str = "pipeline",
    lang: str = "ch",
    timeout: int = 600,
) -> str:
    """Convert via a running mineru-api server. Much faster for batch work.

    Legacy md-only entry — stays for backward compatibility with callers that
    don't care about images. For the image-aware path, see
    ``convert_pdf_mineru_api_with_images``.
    """
    md, _imgs = _call_mineru_api(
        pdf_path, api_url=api_url, backend=backend, lang=lang, timeout=timeout,
        return_images=False,
    )
    return md


def convert_pdf_mineru_api_with_images(
    pdf_path: str | Path,
    *,
    api_url: str | None = None,
    backend: str = "pipeline",
    lang: str = "ch",
    timeout: int = 600,
) -> tuple[str, dict[str, bytes]]:
    """Convert + return images dict.

    Returns ``(markdown, images)`` where ``images`` maps ``"<mineru_name>.jpg"``
    (matching the ``![](images/<mineru_name>.jpg)`` refs embedded in markdown)
    to raw image bytes (already base64-decoded from the API's data URIs).
    """
    return _call_mineru_api(
        pdf_path, api_url=api_url, backend=backend, lang=lang, timeout=timeout,
        return_images=True,
    )


def _call_mineru_api(
    pdf_path: str | Path,
    *,
    api_url: str | None,
    backend: str,
    lang: str,
    timeout: int,
    return_images: bool,
) -> tuple[str, dict[str, bytes]]:
    """Shared helper — POSTs to /file_parse and decodes response.

    When ``return_images=False`` the second tuple element is an empty dict.
    When ``return_images=True`` it is ``{"<mineru_name>.jpg": raw_bytes}``.
    Data URIs of the form ``data:image/<mime>;base64,<b64>`` are decoded into
    raw bytes here so the caller doesn't have to string-parse.
    """
    import base64
    api_url = (api_url or os.environ["MINERU_API_URL"]).rstrip("/")
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(str(pdf))
    pdf_bytes = pdf.read_bytes()

    body, ctype = _multipart_body(
        fields={
            "backend": backend,
            "parse_method": "auto",
            "lang_list": lang,       # mineru-api accepts a single string
            "return_md": "true",
            "return_middle_json": "false",
            "return_images": "true" if return_images else "false",
            "response_format_zip": "false",
        },
        files={"files": (pdf.name, pdf_bytes, "application/pdf")},
    )
    req = urllib.request.Request(
        f"{api_url}/file_parse", data=body, method="POST",
        headers={"Content-Type": ctype, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_err = e.read()[:1500].decode("utf-8", errors="replace")
        raise RuntimeError(f"mineru-api HTTP {e.code}: {body_err}") from e

    # The response wraps results per filename; pick the first md we find.
    results = data.get("results") or {}
    md = ""
    images_raw: dict[str, str] = {}
    for _fname, entry in results.items():
        md = entry.get("md_content") or entry.get("markdown") or entry.get("md") or ""
        if md:
            if return_images:
                images_raw = entry.get("images") or {}
            break
    if not md and isinstance(data.get("md_content"), str):
        # some versions nest differently
        md = data["md_content"]
    if not md:
        raise RuntimeError(f"mineru-api returned no markdown. Payload: {str(data)[:500]}")

    # Decode data URIs → raw bytes. MinerU's format is
    # ``data:image/<mime>;base64,<b64>``. We strip the header and b64decode.
    images: dict[str, bytes] = {}
    for name, data_uri in images_raw.items():
        if not isinstance(data_uri, str):
            logger.warning("mineru image %s not a string (type=%s), skipping",
                           name, type(data_uri).__name__)
            continue
        _, _, payload = data_uri.partition(",")
        if not payload:
            logger.warning("mineru image %s has no base64 payload, skipping", name)
            continue
        try:
            images[name] = base64.b64decode(payload)
        except Exception as e:
            logger.warning("mineru image %s base64 decode failed: %s", name, e)
            continue

    return md, images


def convert_pdf_mineru(
    pdf_path: str | Path,
    *,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "ch",
    timeout: int = 600,
) -> str:
    """Convert a PDF to Markdown. Prefers MINERU_API_URL when set; falls back to CLI.

    CLI keeps the same defaults (pipeline backend; auto parse method; Chinese
    language hint works for mixed zh/en papers).
    """
    md, _imgs = convert_pdf_mineru_with_images(
        pdf_path, backend=backend, method=method, lang=lang, timeout=timeout,
        want_images=False,
    )
    return md


def convert_pdf_mineru_cloud_with_images(
    pdf_path: str | Path,
    *,
    token: str | None = None,
    lang: str = "en",
    model_version: str = "vlm",
    poll_interval: int = 3,
    total_timeout: int = 600,
) -> tuple[str, dict[str, bytes]]:
    """Convert via MinerU Cloud API (https://mineru.net/api/v4).

    Five-step protocol (signed upload, async job, CDN result):
      1. POST /file-urls/batch  → get ``batch_id`` + Ali-OSS presigned PUT URL
      2. ``requests.put(url, data=bytes)`` — CRITICAL to use requests and
         NOT urllib: urllib auto-adds ``Content-Type`` which changes the
         OSS canonical-string and triggers ``SignatureDoesNotMatch``.
      3. Poll ``GET /extract-results/batch/{batch_id}`` every ``poll_interval``
         seconds until ``state`` ∈ {done, failed} or ``total_timeout`` hit.
      4. On ``done``: download ``full_zip_url`` (CDN at cdn-mineru.openxlab.org.cn).
      5. Extract ``full.md`` + ``images/<sha>.jpg`` to match the same output
         shape as ``convert_pdf_mineru_api_with_images`` (key = filename
         including extension, value = raw bytes; MinerU uses 64-hex content
         hash as filename, identical to self-hosted API).

    Returns ``(md, images_dict)``. Raises RuntimeError on auth/quota/upload
    failures so caller can fall through to next tier (CLI).

    Quota: 2000 pages/day at highest priority per MinerU account, then
    degraded (not hard-capped). Use this tier when self-hosted API is
    unavailable (laptop, no GPU, no 3-5GB model on disk).
    """
    import io as _io
    import requests
    import zipfile as _zipfile

    token = token or os.environ.get("MINERU_CLOUD_TOKEN", "")
    if not token:
        raise RuntimeError("MINERU_CLOUD_TOKEN env var not set")

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(str(pdf))
    pdf_bytes = pdf.read_bytes()

    hdr_auth = {"Authorization": f"Bearer {token}"}
    api_base = "https://mineru.net/api/v4"

    # Step 1: request signed upload URL
    try:
        resp = requests.post(
            f"{api_base}/file-urls/batch",
            json={
                "enable_formula": True,
                "enable_table": True,
                "language": lang,
                "files": [{"name": pdf.name, "is_ocr": False,
                            "data_id": pdf.stem}],
                "model_version": model_version,
            },
            headers=hdr_auth, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"mineru cloud: /file-urls/batch failed: {e}") from e

    if data.get("code") != 0:
        raise RuntimeError(f"mineru cloud: file-urls batch rejected: {data}")
    inner = data.get("data", {})
    batch_id = inner.get("batch_id")
    file_urls = inner.get("file_urls", [])
    if not batch_id or not file_urls:
        raise RuntimeError(f"mineru cloud: malformed response: {inner}")
    upload_url = file_urls[0]

    # Step 2: PUT bytes to OSS presigned URL (NO Content-Type header — urllib
    # auto-adds application/x-www-form-urlencoded which breaks OSS signature;
    # requests doesn't by default).
    try:
        put_resp = requests.put(upload_url, data=pdf_bytes, timeout=120)
        put_resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"mineru cloud: OSS upload failed: {e}") from e

    # Step 3: poll status. Budget reset here (Finding #1) — upload may eat
    # several minutes for large PDFs / weak networks, but ``total_timeout``
    # is meant for the server-side extraction time, not end-to-end.
    t_poll_start = time.time()
    result_url: str | None = None
    state = "pending"
    backoff = poll_interval    # grows on transient errors, reset on success
    while state not in ("done", "failed"):
        if time.time() - t_poll_start > total_timeout:
            raise RuntimeError(
                f"mineru cloud: poll timeout after {total_timeout}s "
                f"(last state={state})")
        time.sleep(backoff)
        try:
            poll_resp = requests.get(
                f"{api_base}/extract-results/batch/{batch_id}",
                headers=hdr_auth, timeout=30,
            )
            sc = poll_resp.status_code
            # Auth/resource-level errors are fatal — don't burn budget.
            if sc in (401, 403, 404):
                raise RuntimeError(
                    f"mineru cloud: poll HTTP {sc} (fatal — token/batch bad): "
                    f"{poll_resp.text[:200]}")
            poll_resp.raise_for_status()
            poll = poll_resp.json()
            backoff = poll_interval   # success → reset exp-backoff
        except RuntimeError:
            raise
        except Exception as e:
            # Transient (network flake, 502 gateway, timeout). Exp backoff
            # capped at 30s so a brief outage doesn't hammer + long outage
            # still respects total_timeout.
            backoff = min(backoff * 2, 30)
            logger.debug("mineru cloud: poll transient (%s), backoff=%ds", e, backoff)
            continue
        er = (poll.get("data", {}).get("extract_result") or [{}])[0]
        state = er.get("state", "unknown")
        if state == "failed":
            raise RuntimeError(
                f"mineru cloud: parse failed: {er.get('err_msg','')}")
        if state == "done":
            result_url = er.get("full_zip_url")
            break

    if not result_url:
        raise RuntimeError("mineru cloud: no full_zip_url in done state")

    # Step 4: download zip
    try:
        zr = requests.get(result_url, timeout=120)
        zr.raise_for_status()
        zbytes = zr.content
    except Exception as e:
        raise RuntimeError(f"mineru cloud: result zip download failed: {e}") from e

    # Step 5: extract md + images; narrow BadZipFile so log shows "corrupt"
    # distinct from other failures (Finding #5).
    md_text = ""
    images: dict[str, bytes] = {}
    try:
        with _zipfile.ZipFile(_io.BytesIO(zbytes)) as zf:
            for name in zf.namelist():
                if name == "full.md":
                    md_text = zf.read(name).decode("utf-8", errors="replace")
                elif name.startswith("images/") and not name.endswith("/"):
                    # Strip "images/" prefix → match self-hosted API's key
                    # format (filename with extension, no directory)
                    fname = name.split("/", 1)[1]
                    images[fname] = zf.read(name)
    except _zipfile.BadZipFile as e:
        raise RuntimeError(
            f"mineru cloud: corrupt zip ({len(zbytes)} bytes): {e}") from e

    # Finding #4: whitespace-only md isn't useful; treat as empty.
    if not md_text.strip():
        raise RuntimeError("mineru cloud: full.md empty/whitespace")
    return md_text, images


def convert_pdf_mineru_with_images(
    pdf_path: str | Path,
    *,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "ch",
    timeout: int = 600,
    want_images: bool = True,
) -> tuple[str, dict[str, bytes]]:
    """Convert + optionally return images. Three-tier fallback:

        Tier 1  self-hosted mineru-api (fastest ~2-26s, MINERU_API_URL set)
        Tier 2  MinerU Cloud API (~60-120s, laptop-friendly,
                MINERU_CLOUD_TOKEN set; 2000 pages/day high-priority quota)
        Tier 3  local CLI (~40-60s cold load, needs 3-5GB model)

    Each tier skipped silently when its trigger env var is missing. A tier
    that fires but errors falls through to the next. When ``want_images`` is
    False, the returned images dict is empty (parity with legacy md-only
    callers) — but Tier 2 always fetches images (cost is negligible
    compared to the ~100s total), and we just don't return them.
    """
    errors: list[str] = []

    # Tier 1: self-hosted mineru-api daemon. Circuit-break after
    # ``_TIER1_FAIL_LIMIT`` consecutive failures in this process — otherwise
    # a dead daemon + stale env var costs 20-30s HTTP timeout per paper,
    # which for a 4k-paper batch burns hours (Finding #6).
    if os.environ.get("MINERU_API_URL") and not _tier1_tripped():
        try:
            if want_images:
                return convert_pdf_mineru_api_with_images(
                    pdf_path, backend=backend, lang=lang, timeout=timeout,
                )
            return convert_pdf_mineru_api(
                pdf_path, backend=backend, lang=lang, timeout=timeout,
            ), {}
        except Exception as e:
            _tier1_record_failure()
            errors.append(f"mineru-api: {e}")
            logger.warning("mineru Tier 1 (self-hosted) failed: %s", e)

    # Tier 2: MinerU Cloud API. Language passes through (Finding #7 —
    # previous code forced ch→en, dropping the Chinese-document hint).
    if os.environ.get("MINERU_CLOUD_TOKEN"):
        try:
            md, images = convert_pdf_mineru_cloud_with_images(
                pdf_path, lang=lang, total_timeout=timeout,
                model_version=os.environ.get("MINERU_CLOUD_MODEL", "vlm"),
            )
            return (md, images if want_images else {})
        except Exception as e:
            errors.append(f"mineru-cloud: {e}")
            logger.warning("mineru Tier 2 (cloud) failed: %s", e)

    # Tier 3: local CLI
    logger.info("mineru Tier 3 (CLI fallback) — this is slow (model cold-load)")
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(str(pdf))

    with tempfile.TemporaryDirectory(prefix="mineru-") as td:
        out_dir = Path(td)
        cmd = [
            _mineru_bin(),
            "-p", str(pdf), "-o", str(out_dir),
            "-b", backend, "-m", method, "-l", lang,
        ]
        logger.info("running mineru CLI: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"mineru failed (exit {result.returncode}): "
                f"stderr={result.stderr[-2000:]}"
            )
        md_path = _find_output_md(out_dir, pdf.stem)
        if md_path is None:
            raise RuntimeError(
                f"mineru produced no markdown under {out_dir}. "
                f"Stdout tail: {result.stdout[-500:]}"
            )
        md = md_path.read_text(encoding="utf-8", errors="replace")
        images: dict[str, bytes] = {}
        if want_images:
            # CLI lays out `<out_dir>/.../images/*.jpg`. Read bytes out before
            # TemporaryDirectory context exits and nukes everything.
            for img_path in out_dir.rglob("images/*"):
                if img_path.is_file():
                    images[img_path.name] = img_path.read_bytes()
        return md, images

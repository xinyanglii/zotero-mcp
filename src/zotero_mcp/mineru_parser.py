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
import urllib.error
import urllib.request
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


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


def convert_pdf_mineru_with_images(
    pdf_path: str | Path,
    *,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "ch",
    timeout: int = 600,
    want_images: bool = True,
) -> tuple[str, dict[str, bytes]]:
    """Convert + optionally return images.

    API mode is preferred; on failure falls back to CLI. When ``want_images``
    is False the returned images dict is empty (parity with legacy md-only
    callers). The CLI fallback scans ``<out_dir>/.../images/*.jpg`` within the
    ``TemporaryDirectory`` context so bytes are read out before the dir is
    auto-deleted on context exit.
    """
    if os.environ.get("MINERU_API_URL"):
        try:
            if want_images:
                return convert_pdf_mineru_api_with_images(
                    pdf_path, backend=backend, lang=lang, timeout=timeout,
                )
            return convert_pdf_mineru_api(
                pdf_path, backend=backend, lang=lang, timeout=timeout,
            ), {}
        except Exception as e:
            logger.warning("mineru-api failed, falling back to CLI: %s", e)

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

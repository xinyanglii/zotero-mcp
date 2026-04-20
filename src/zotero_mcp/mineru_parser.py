"""
MinerU-based PDF parser.

MinerU 3.1.0 (上海 AI Lab OpenDataLab) offers LaTeX formula recognition,
accurate table extraction, multi-column layout, figure captions, and mixed
CJK/English handling. This is the preferred parser for academic PDFs.

We shell out to the `mineru` CLI rather than calling internal APIs — the CLI
surface is stable and the internal `mineru.backend.*` modules are not a
publicly documented contract.

Usage:
    from zotero_mcp.mineru_parser import convert_pdf_mineru
    md = convert_pdf_mineru("/path/to/paper.pdf")
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
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


def convert_pdf_mineru(
    pdf_path: str | Path,
    *,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "ch",
    timeout: int = 600,
) -> str:
    """Convert a PDF to Markdown via MinerU.

    Args:
        pdf_path: Path to the input PDF.
        backend: MinerU backend. Default `hybrid-auto-engine` is the most
            accurate local option in 3.1.0.
        method: pdf parsing method; auto/txt/ocr.
        lang: OCR language hint; `ch` handles Chinese+English reasonably.
        timeout: Subprocess timeout in seconds (default 10 min per paper).

    Returns:
        The full markdown text.

    Raises:
        FileNotFoundError: input PDF missing.
        RuntimeError: MinerU binary missing or produced no markdown.
        subprocess.TimeoutExpired: conversion took too long.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(str(pdf))

    with tempfile.TemporaryDirectory(prefix="mineru-") as td:
        out_dir = Path(td)
        cmd = [
            _mineru_bin(),
            "-p", str(pdf),
            "-o", str(out_dir),
            "-b", backend,
            "-m", method,
            "-l", lang,
        ]
        logger.info("running mineru: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
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
        return md_path.read_text(encoding="utf-8", errors="replace")

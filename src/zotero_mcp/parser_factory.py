"""
Parser factory: picks the right Markdown converter for a given file.

Routing rules (first match wins):
  1. Env `ZOTERO_MCP_PARSER` in {"mineru", "markitdown"} → force it
  2. File extension `.pdf` and MinerU is available → MinerU (academic PDFs)
  3. Fallback → markitdown (handles docx/xlsx/pptx/webpage snapshots etc.)

The single entry point `convert_to_markdown_smart(path)` preserves the existing
contract (returns a markdown string) so `client.convert_to_markdown` can
delegate transparently.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _mineru_available() -> bool:
    # Mirror mineru_parser._mineru_bin so PATH-less subprocesses (nohup/systemd)
    # still find the binary next to the current python interpreter.
    override = os.getenv("MINERU_BIN", "").strip()
    if override and Path(override).is_file():
        return True
    if shutil.which("mineru"):
        return True
    import sys
    return (Path(sys.executable).parent / "mineru").is_file()


def _select_parser(path: Path) -> str:
    override = os.getenv("ZOTERO_MCP_PARSER", "").strip().lower()
    if override in {"mineru", "markitdown"}:
        return override
    if path.suffix.lower() == ".pdf" and _mineru_available():
        return "mineru"
    return "markitdown"


def convert_to_markdown_smart(file_path: str | Path) -> str:
    """Convert any supported file to markdown, routing to the best parser.

    Legacy md-only entry. For figure-aware conversion see
    ``convert_to_markdown_smart_with_images``.
    """
    md, _imgs = convert_to_markdown_smart_with_images(file_path, want_images=False)
    return md


def convert_to_markdown_smart_with_images(
    file_path: str | Path, *, want_images: bool = True,
) -> tuple[str, dict[str, bytes]]:
    """Convert + optionally return figure images.

    Returns ``(markdown, images)`` where ``images`` maps
    ``"<mineru_name>.jpg"`` → raw bytes. Non-PDF (markitdown) path returns an
    empty images dict regardless of ``want_images`` — markitdown doesn't
    produce MinerU-style figure refs so there's nothing to capture.
    """
    path = Path(file_path)
    parser = _select_parser(path)

    if parser == "mineru":
        try:
            from .mineru_parser import convert_pdf_mineru_with_images
            logger.info("parser=mineru file=%s (with_images=%s)", path.name, want_images)
            return convert_pdf_mineru_with_images(path, want_images=want_images)
        except Exception as e:
            logger.warning("MinerU failed on %s: %s — falling back to markitdown", path.name, e)

    # markitdown path (also the fallback). No images to return.
    try:
        from markitdown import MarkItDown
        logger.info("parser=markitdown file=%s", path.name)
        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content, {}
    except Exception as e:
        return f"Error converting file to markdown: {e}", {}

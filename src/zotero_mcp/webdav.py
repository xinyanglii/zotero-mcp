"""
Zotero WebDAV helpers.

Zotero's built-in WebDAV sync uses a specific format: each attachment is
stored as ``<attachmentKey>.zip`` (zipped file) + ``<attachmentKey>.prop``
(XML metadata) under the user's WebDAV root (typically
``https://dav.jianguoyun.com/dav/zotero/``).

This module provides small, dependency-free helpers for:

- Downloading an attachment file from WebDAV given its Zotero key
- Uploading a file (PDF / markdown / anything) as a Zotero-compatible
  attachment pair

Enable by setting ``ZOTERO_WEBDAV_USER`` + ``ZOTERO_WEBDAV_PASS`` +
(optional) ``ZOTERO_WEBDAV_URL`` env vars. ``ZOTERO_WEBDAV_URL`` defaults
to ``https://dav.jianguoyun.com/dav/zotero`` (trailing slash optional,
trimmed).

The ``create_zotero_webdav_attachment`` function combines Zotero item
creation + WebDAV upload + md5/mtime patch in one call — use it from
``tools/_helpers.py:_download_and_attach_pdf`` and from the ingest pipeline
when auto-attaching extracted markdown.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def webdav_enabled() -> bool:
    """True when the env has WebDAV credentials configured."""
    return bool(os.environ.get("ZOTERO_WEBDAV_USER")
                and os.environ.get("ZOTERO_WEBDAV_PASS"))


def _webdav_root() -> str:
    return os.environ.get("ZOTERO_WEBDAV_URL",
                          "https://dav.jianguoyun.com/dav/zotero").rstrip("/")


def _auth_header() -> str:
    user = os.environ["ZOTERO_WEBDAV_USER"]
    passwd = os.environ["ZOTERO_WEBDAV_PASS"]
    return "Basic " + base64.b64encode(f"{user}:{passwd}".encode()).decode()


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def fetch_attachment_bytes(attachment_key: str, timeout: int = 60) -> bytes | None:
    """Download ``<key>.zip`` from WebDAV and return the inner file's bytes.

    Returns ``None`` on 404 / missing / bad zip.
    """
    url = f"{_webdav_root()}/{attachment_key}.zip"
    req = urllib.request.Request(url, headers={"Authorization": _auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("WebDAV miss %s (404)", attachment_key)
            return None
        raise
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # Skip directory entries; Zotero's sync zip contains a single
            # file but a hostile or malformed zip could include traversal
            # paths or dir entries that would raise IsADirectoryError here.
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                return zf.read(name)
            return None
    except zipfile.BadZipFile:
        logger.warning("WebDAV %s returned non-zip content", attachment_key)
        return None


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
def upload_attachment_bytes(
    attachment_key: str,
    filename: str,
    file_bytes: bytes,
    *,
    timeout: int = 120,
) -> tuple[str, int]:
    """Upload ``<key>.zip`` + ``<key>.prop`` following Zotero sync protocol.

    Args:
        attachment_key: The Zotero attachment item key (not the parent paper's!).
        filename: Filename inside the zip (must match the attachment's
            ``filename`` field so Zotero desktop lines them up).
        file_bytes: Raw bytes of the file to attach.

    Returns:
        ``(md5_hex, mtime_ms)`` — caller should PATCH these onto the Zotero
        attachment item so the desktop client recognizes the upload.
    """
    root = _webdav_root()
    md5_hex = hashlib.md5(file_bytes).hexdigest()
    mtime_ms = int(time.time() * 1000)

    # zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, file_bytes)
    zip_bytes = buf.getvalue()

    # .prop XML
    prop_xml = (
        f'<properties version="1">\n'
        f'<mtime>{mtime_ms}</mtime>\n'
        f'<hash>{md5_hex}</hash>\n'
        f'</properties>\n'
    ).encode()

    headers = {"Authorization": _auth_header()}
    uploaded: list[str] = []
    try:
        for suffix, body in ((".zip", zip_bytes), (".prop", prop_xml)):
            req = urllib.request.Request(
                f"{root}/{attachment_key}{suffix}",
                data=body, method="PUT", headers=headers,
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read()
            uploaded.append(suffix)
    except Exception:
        # Partial upload cleanup: if .zip succeeded but .prop failed, delete
        # the orphan .zip so Zotero desktop doesn't see a broken attachment.
        for suffix in uploaded:
            try:
                req = urllib.request.Request(
                    f"{root}/{attachment_key}{suffix}",
                    method="DELETE", headers=headers,
                )
                urllib.request.urlopen(req, timeout=timeout).read()
            except Exception as cleanup_err:
                logger.debug("orphan cleanup of %s%s failed: %s",
                             attachment_key, suffix, cleanup_err)
        raise
    return md5_hex, mtime_ms


# ---------------------------------------------------------------------------
# combined: create Zotero attachment item + WebDAV upload + md5/mtime patch
# ---------------------------------------------------------------------------
def create_zotero_webdav_attachment(
    zot,
    *,
    parent_key: str,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    title: str | None = None,
    extra_tags: list[str] | None = None,
) -> str | None:
    """Create an imported_file attachment in Zotero + upload via WebDAV.

    Args:
        zot: A ``pyzotero.zotero.Zotero`` write client.
        parent_key: The parent paper/item's Zotero key.
        file_bytes: Raw bytes to attach.
        filename: Filename to register (e.g. ``"paper.pdf"`` or ``"X.md"``).
        content_type: MIME type (``application/pdf`` / ``text/markdown``).
        title: Optional display title for the attachment.
        extra_tags: Optional tags to add to the attachment.

    Returns:
        The new attachment's Zotero key on success, ``None`` on failure.
    """
    if not webdav_enabled():
        logger.warning("WebDAV not configured (ZOTERO_WEBDAV_USER/PASS missing)")
        return None

    template = zot.item_template("attachment", "imported_file")
    template["parentItem"] = parent_key
    template["filename"] = filename
    template["contentType"] = content_type
    template["title"] = title or filename
    if extra_tags:
        template["tags"] = [{"tag": t} for t in extra_tags]

    result = zot.create_items([template])
    if not result.get("successful"):
        logger.warning("Zotero attachment create failed: %s", result.get("failed"))
        return None
    att = list(result["successful"].values())[0]
    att_key = att["key"]
    att_version = att.get("version")

    try:
        md5_hex, mtime_ms = upload_attachment_bytes(att_key, filename, file_bytes)
    except Exception as e:
        logger.exception("WebDAV upload failed for %s: %s", att_key, e)
        return None

    # patch md5 + mtime so Zotero desktop recognizes the upload.
    # Use direct REST PATCH — pyzotero's ``update_item`` expects a specific
    # dict shape that diverges across versions, and its internal error
    # handling swallows 412s we want to surface. Direct urllib keeps the
    # semantics explicit.
    try:
        api_base = zot.endpoint.rstrip("/")
        # pyzotero stores library_type already plural ("users" / "groups"),
        # NOT "user"/"group" — do NOT append another "s".
        url = f"{api_base}/{zot.library_type}/{zot.library_id}/items/{att_key}"
        body = json.dumps({"md5": md5_hex, "mtime": mtime_ms}).encode()
        headers = {
            "Zotero-API-Key": zot.api_key,
            "Zotero-API-Version": "3",
            "Content-Type": "application/json",
        }
        if att_version is not None:
            headers["If-Unmodified-Since-Version"] = str(att_version)
        req = urllib.request.Request(url, data=body, method="PATCH", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception as e:
        # md5/mtime patch is load-bearing (Zotero desktop won't recognize
        # the upload without it). Log at warning, not debug.
        logger.warning("md5/mtime PATCH failed for %s: %s", att_key, e)

    return att_key

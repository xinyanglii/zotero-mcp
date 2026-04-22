#!/usr/bin/env python3
"""Retro-upload MinerU markdown as Zotero child attachments.

For every paper in the KG SQLite store, create a `<paperId>.md` attachment in
Zotero and upload the corresponding zip+prop to Jianguoyun WebDAV (so Zotero
desktop and the MCP server both see it). Idempotent: skips papers that already
have a `kg:extracted_md` tag on their parent item.

Effect: downstream agents (小兰 / future Claude sessions) can read the markdown
straight from the Zotero attachment without re-running MinerU.

Usage:
    python -m scripts.backfill_md_attachments             # all papers
    python -m scripts.backfill_md_attachments --limit 50  # smoke test
    python -m scripts.backfill_md_attachments --keys ABC123 DEF456
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

logger = logging.getLogger("backfill_md")

WEBDAV_BASE = "https://dav.jianguoyun.com/dav/zotero"
TAG_DONE = "kg:extracted_md"


def _env():
    """Load .env.local into process env."""
    env = subprocess.check_output(
        "grep -E '^ZOTERO' /home/xinyang/.claude/.env.local", shell=True, text=True)
    for line in env.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _zotero_headers():
    return {
        "Zotero-API-Key": os.environ["ZOTERO_API_KEY"],
        "Zotero-API-Version": "3",
        "Content-Type": "application/json",
    }


def _zotero_base():
    return f"https://api.zotero.org/users/{os.environ['ZOTERO_LIBRARY_ID']}"


def _webdav_auth():
    creds = f"{os.environ['ZOTERO_WEBDAV_USER']}:{os.environ['ZOTERO_WEBDAV_PASS']}"
    return "Basic " + base64.b64encode(creds.encode()).decode()


def _z_req(method, path, body=None, extra_headers=None):
    h = _zotero_headers().copy()
    if extra_headers:
        h.update(extra_headers)
    data = json.dumps(body).encode() if body is not None else None
    url = _zotero_base() + path
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=h)
            return urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", "5"))
                logger.warning("429 sleep %ds", ra)
                time.sleep(ra)
                continue
            if e.code == 412:
                logger.warning("412 version mismatch at %s, skip", path)
                return None
            raise
    return None


def _wd_put(url, data, max_attempts=4):
    req = urllib.request.Request(url, data=data, method="PUT",
                                  headers={"Authorization": _webdav_auth()})
    delay = 5.0
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            urllib.request.urlopen(req, timeout=60).read()
            return
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) or 500 <= e.code < 600:
                last_err = e
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    sleep_s = int(ra) if ra else delay
                except ValueError:
                    sleep_s = delay
                sleep_s = min(sleep_s, 40)
                logger.warning("webdav put %d on %d/%d — sleeping %.1fs",
                               e.code, attempt, max_attempts, sleep_s)
                time.sleep(sleep_s)
                delay = min(delay * 2, 40)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            logger.warning("webdav put transport err %d/%d (%s) sleep %.1fs",
                           attempt, max_attempts, type(e).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 40)
    raise RuntimeError(f"webdav put unavailable after {max_attempts}: {last_err!r}")


def parent_has_tag(parent_key: str, tag: str) -> bool:
    r = _z_req("GET", f"/items/{parent_key}")
    if not r:
        return False
    data = json.loads(r.read())["data"]
    return any(t["tag"] == tag for t in data.get("tags", []))


def add_parent_tag(parent_key: str, tag: str):
    r = _z_req("GET", f"/items/{parent_key}")
    if not r:
        return
    version = r.headers.get("Last-Modified-Version", "0")
    data = json.loads(r.read())["data"]
    existing = {t["tag"] for t in data.get("tags", [])}
    if tag in existing:
        return
    merged = list(existing | {tag})
    _z_req("PATCH", f"/items/{parent_key}",
           body={"tags": [{"tag": t} for t in merged]},
           extra_headers={"If-Unmodified-Since-Version": version})


def upload_md_as_attachment(parent_key: str, md_text: str) -> str | None:
    """Create attachment item + upload zip+prop. Returns attachment key or None."""
    filename = f"{parent_key}.md"

    # 1. create attachment item
    body = [{
        "itemType": "attachment",
        "linkMode": "imported_file",
        "parentItem": parent_key,
        "title": "MinerU Markdown",
        "filename": filename,
        "contentType": "text/markdown",
        "tags": [{"tag": "kg:auto_extracted"}],
    }]
    r = _z_req("POST", "/items", body=body)
    if not r:
        logger.error("failed to create attachment for %s", parent_key)
        return None
    result = json.loads(r.read())
    if not result.get("successful"):
        logger.error("attachment create not successful: %s", result.get("failed"))
        return None
    att = list(result["successful"].values())[0]
    att_key, att_ver = att["key"], att["version"]

    # 2. zip md + compute md5 + mtime
    md_bytes = md_text.encode("utf-8")
    md5 = hashlib.md5(md_bytes).hexdigest()
    mtime_ms = int(time.time() * 1000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, md_bytes)
    zip_bytes = buf.getvalue()
    prop_xml = (
        f'<properties version="1">\n'
        f'<mtime>{mtime_ms}</mtime>\n'
        f'<hash>{md5}</hash>\n'
        f'</properties>\n'
    ).encode()

    # 3. upload
    try:
        _wd_put(f"{WEBDAV_BASE}/{att_key}.zip", zip_bytes)
        _wd_put(f"{WEBDAV_BASE}/{att_key}.prop", prop_xml)
    except Exception as e:
        logger.error("webdav put %s failed: %s", att_key, e)
        return None

    # 4. PATCH attachment with md5/mtime so desktop client sees match
    _z_req("PATCH", f"/items/{att_key}",
           body={"md5": md5, "mtime": mtime_ms},
           extra_headers={"If-Unmodified-Since-Version": str(att_ver)})

    return att_key


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite",
                    default=os.path.expanduser("~/.cache/zotero-mcp/kg.sqlite"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--min-chars", type=int, default=300,
                    help="skip trivially short markdown (metadata-only fallbacks)")
    args = ap.parse_args()

    _env()
    c = sqlite3.connect(args.sqlite)
    c.row_factory = sqlite3.Row

    if args.keys:
        placeholders = ",".join("?" * len(args.keys))
        rows = c.execute(
            f"SELECT paper_id, md_text, md_chars FROM papers WHERE paper_id IN ({placeholders})",
            args.keys,
        ).fetchall()
    else:
        q = "SELECT paper_id, md_text, md_chars FROM papers WHERE md_chars >= ?"
        params: list = [args.min_chars]
        if args.limit:
            q += " LIMIT ?"
            params.append(args.limit)
        rows = c.execute(q, params).fetchall()

    logger.info("candidates: %d", len(rows))
    stats = {"done": 0, "skip_already": 0, "fail": 0}
    for i, row in enumerate(rows, 1):
        pid, md_text, md_chars = row["paper_id"], row["md_text"], row["md_chars"]
        try:
            if parent_has_tag(pid, TAG_DONE):
                stats["skip_already"] += 1
                if i % 50 == 0:
                    logger.info("[%d/%d] progress %s", i, len(rows), stats)
                continue
            att_key = upload_md_as_attachment(pid, md_text)
            if att_key:
                add_parent_tag(pid, TAG_DONE)
                stats["done"] += 1
                logger.info("[%d/%d] %s → %s (%d chars)", i, len(rows), pid, att_key, md_chars)
            else:
                stats["fail"] += 1
        except Exception as e:
            logger.exception("[%d/%d] %s failed: %s", i, len(rows), pid, e)
            stats["fail"] += 1
        time.sleep(0.1)  # gentle on APIs
    logger.info("DONE. stats=%s", stats)


if __name__ == "__main__":
    main()

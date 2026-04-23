#!/usr/bin/env python3
"""T2 · Entity canonicalization — per spec `plans/zotero-kg-t2-spec.md` v2.

Deduplicate semantically equivalent :Concept / :Method / :Dataset nodes in
Neo4j via Qwen v4 embeddings + per-label batched cosine + LLM judge for
the gray zone. Writes ``SAME_WORK_AS`` … wait no: writes ``SAME_AS`` edges
(T1 used SAME_WORK_AS for Paper ↔ Paper/ExternalRef; T2 uses SAME_AS for
entity-level canonicalization — a different semantic axis).

Usage:
    # Step 1: dry-run + threshold calibration + pricing probe
    python scripts/canonicalize_entities.py --dry-run --limit 100

    # Step 2: smoke 1k nodes end-to-end
    python scripts/canonicalize_entities.py --limit 1000

    # Step 3: full 51k run
    python scripts/canonicalize_entities.py

Idempotent via ``t2_edge_log`` gate + ``apoc.coll.toSet`` on MERGE.
``--resume`` picks up the embeddings cache + edge_log intact.

Design decisions live in the spec file. Key invariants:
  * SAME_AS edge direction: deterministic min(id) → max(id)
  * Embeddings unit-normalized once at load → cosine == dot product
  * Per-label batched GEMM (5k × N_label) not N × N
  * ``_canon(name) == canonical_name`` → embed `name` only (no double encode)
  * Dataset label: `_collapse()` exact-match pre-pass before embedding
  * CJK detection → threshold bumps to 0.95
  * Hub guard (degree > 200) → forces LLM judge even at cosine ≥ 0.92
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env.local"), override=False)
except ImportError:
    pass

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zotero_mcp.kg_store import Neo4jWriter  # noqa: E402

_canon = Neo4jWriter._canon   # @staticmethod on Neo4jWriter, expose flat

logger = logging.getLogger("t2_canon")
logging.basicConfig(
    level=logging.INFO, force=True,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

PROGRESS_DB = Path(os.getenv(
    "T2_PROGRESS_DB",
    os.path.expanduser("~/.cache/zotero-mcp/t2_canonicalization.sqlite"),
))

EMBED_MODEL = os.getenv("T2_EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.getenv("T2_EMBED_DIM", "2048"))
EMBED_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

DEFAULT_LABELS = ("Concept", "Method", "Dataset")
_ALLOWED_LABELS = frozenset(DEFAULT_LABELS)
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_label(label: str) -> str:
    """Guard Cypher f-string interpolation — only allow identifiers from the
    spec-closed set. (Finding 2: raw CLI values into Cypher is injectable.)
    """
    if label not in _ALLOWED_LABELS:
        raise ValueError(
            f"label {label!r} not in allowed set {sorted(_ALLOWED_LABELS)} "
            "— T2 v1 operates intra-label on Concept / Method / Dataset only"
        )
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(f"label {label!r} fails identifier regex")
    return label

# Thresholds (tune via dry-run A)
TH_AUTO = 0.92
TH_AMBIG = 0.85
TH_AUTO_CJK = 0.95
HUB_DEGREE = 200

# CJK detection (spec §8 finding 9)
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ]")


def _has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


def _collapse(s: str) -> str:
    """Strip non-word chars + lowercase. For :Dataset alias pre-pass —
    matches ``CIFAR-10`` ↔ ``CIFAR10`` ↔ ``cifar 10`` as identical."""
    return re.sub(r"[^\w]+", "", (s or "").lower())


# ---------------------------------------------------------------------------
# SQLite progress DB
# ---------------------------------------------------------------------------
def _init_progress_db() -> sqlite3.Connection:
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROGRESS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS t2_embeddings (
            label        TEXT NOT NULL,
            node_id      TEXT NOT NULL,
            name         TEXT NOT NULL,
            canonical    TEXT NOT NULL,
            embed_input  TEXT NOT NULL,
            embed_model  TEXT NOT NULL,
            embedding    BLOB,
            computed_at  REAL,
            PRIMARY KEY (label, node_id, embed_model)
        );
        CREATE INDEX IF NOT EXISTS idx_embed_label
            ON t2_embeddings(label, embed_model);

        CREATE TABLE IF NOT EXISTS t2_llm_judge (
            pair_key     TEXT PRIMARY KEY,
            label_a      TEXT NOT NULL,
            label_b      TEXT NOT NULL,
            cosine       REAL,
            same_entity  INTEGER,
            reason       TEXT,
            model        TEXT,
            judged_at    REAL
        );

        CREATE TABLE IF NOT EXISTS t2_edge_log (
            pair_key     TEXT NOT NULL,
            embed_model  TEXT NOT NULL,
            kind         TEXT,
            written_at   REAL,
            PRIMARY KEY (pair_key, embed_model)
        );

        CREATE TABLE IF NOT EXISTS t2_failures (
            stage       TEXT NOT NULL,
            target      TEXT NOT NULL,
            err         TEXT,
            recorded_at REAL
        );
    """)
    conn.commit()
    return conn


def _record_failure(conn, stage: str, target: str, err: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO t2_failures (stage, target, err, recorded_at) VALUES (?,?,?,?)",
            (stage, target, err, time.time()),
        )


# ---------------------------------------------------------------------------
# Neo4j node loading
# ---------------------------------------------------------------------------
def load_label_nodes(ne: Neo4jWriter, label: str,
                     limit: int | None = None
                     ) -> list[tuple[str, str, str, int]]:
    """Return list of (key, name, canonical_name, degree) for a label.

    Key = canonical_name (entity nodes have no ``id`` property; canonical_name
    is the unique index). Degree = total undirected incident edges, used for
    hub guard (§5.3).
    """
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    q = (
        f"MATCH (n:{label}) "
        "WHERE n.name IS NOT NULL "
        "  AND n.canonical_name IS NOT NULL "
        "RETURN n.canonical_name AS key, n.name AS name, "
        "       n.canonical_name AS canon, count{(n)--()} AS deg "
        "ORDER BY n.canonical_name "
        + limit_clause
    )
    with ne.driver.session() as s:
        rows = s.run(q).data()
    return [(r["key"], r["name"], r["canon"], int(r["deg"])) for r in rows]


# ---------------------------------------------------------------------------
# Embedding — reuse kg_store style, standalone so T2 doesn't need KGStore init
# ---------------------------------------------------------------------------
def _embed_inputs_for(nodes: list[tuple[str, str, str, int]]) -> list[str]:
    """Spec §1.2: single-encode when _canon(name) == canonical_name."""
    out = []
    for _id, name, canon, _deg in nodes:
        if _canon(name) == canon:
            out.append(name)
        else:
            out.append(f"{name} | {canon}")
    return out


def embed_batch(texts: list[str], api_key: str,
                model: str = EMBED_MODEL,
                dim: int = EMBED_DIM,
                batch: int = 10) -> list[np.ndarray]:
    """DashScope text-embedding-v4, batch cap 10, 3x retry w/ backoff.

    Returns list of unit-normalized float32 np.ndarray (so cosine == dot).
    """
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        payload = json.dumps({
            "model": model,
            "input": chunk,
            "dimensions": dim,
        }).encode()
        req = urllib.request.Request(
            f"{EMBED_BASE_URL}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if attempt == 3:
                    raise
                sleep = [2, 5, 10, 30][attempt]
                logger.warning("embed HTTP %s (attempt %d/4) — sleep %ds",
                                e.code, attempt + 1, sleep)
                time.sleep(sleep)
            except urllib.error.URLError as e:
                if attempt == 3:
                    raise
                sleep = [2, 5, 10, 30][attempt]
                logger.warning("embed URL err %s — sleep %ds", e, sleep)
                time.sleep(sleep)
        for d in resp["data"]:
            v = np.asarray(d["embedding"], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n          # unit-norm so dot == cosine
            out.append(v)
    return out


def embed_nodes_with_cache(
    conn: sqlite3.Connection,
    label: str,
    nodes: list[tuple[str, str, str, int]],
    inputs: list[str],
    api_key: str,
) -> np.ndarray:
    """Return (N, dim) float32 unit-normed matrix (rows align with nodes).

    Cache hits come straight from SQLite; misses go to DashScope in batches
    of 10 with per-batch commits so a crash loses ≤ 10 embeddings.

    Note: embedding always runs, even in --dry-run (dry-run's purpose is to
    skip the EXPENSIVE step — LLM judge + Neo4j writes, not the cheap cache-
    building embed pass which the full run will inherit).
    """
    cached: dict[str, np.ndarray] = {}
    ids = [n[0] for n in nodes]
    # Chunk IN(…) to stay under SQLite SQLITE_MAX_VARIABLE_NUMBER (default
    # 32766 on modern SQLite, 999 on old; Finding 1). 500/chunk is safe for
    # either + negligible query overhead for 51k nodes.
    CHUNK = 500
    for start in range(0, len(ids), CHUNK):
        sub = ids[start:start + CHUNK]
        placeholders = ",".join("?" * len(sub))
        rows = conn.execute(
            f"SELECT node_id, embedding FROM t2_embeddings "
            f"WHERE label=? AND embed_model=? AND node_id IN ({placeholders})",
            [label, EMBED_MODEL, *sub],
        ).fetchall()
        for nid, blob in rows:
            cached[nid] = np.frombuffer(blob, dtype=np.float32)
    missing_idx = [i for i, nid in enumerate(ids) if nid not in cached]
    logger.info("label=%s: %d cached, %d to embed", label,
                len(cached), len(missing_idx))

    if missing_idx:
        # batch to 10 per API call
        BATCH = 10
        for chunk_start in range(0, len(missing_idx), BATCH):
            chunk_idx = missing_idx[chunk_start:chunk_start + BATCH]
            chunk_texts = [inputs[i] for i in chunk_idx]
            try:
                embeds = embed_batch(chunk_texts, api_key)
            except Exception as e:
                logger.error("embed batch failed: %s", e)
                for i in chunk_idx:
                    _record_failure(conn, "embed", ids[i], str(e))
                continue
            with conn:
                for local_i, vec in enumerate(embeds):
                    i = chunk_idx[local_i]
                    nid, name, canon, _deg = nodes[i]
                    cached[nid] = vec
                    conn.execute(
                        "INSERT OR REPLACE INTO t2_embeddings "
                        "(label, node_id, name, canonical, embed_input, "
                        "embed_model, embedding, computed_at) VALUES (?,?,?,?,?,?,?,?)",
                        (label, nid, name, canon, inputs[i],
                         EMBED_MODEL, vec.tobytes(), time.time()),
                    )
            if (chunk_start // BATCH) % 10 == 0:
                done = chunk_start + len(chunk_idx)
                logger.info("  embed progress %d/%d", done, len(missing_idx))

    # Assemble aligned matrix; any still-missing rows (API failures) stay
    # zero and won't pass cosine threshold.
    mat = np.zeros((len(nodes), EMBED_DIM), dtype=np.float32)
    for i, nid in enumerate(ids):
        if nid in cached:
            mat[i] = cached[nid]
    return mat


# ---------------------------------------------------------------------------
# Batched GEMM pair finder (spec §6.1)
# ---------------------------------------------------------------------------
def find_similar_pairs(embeddings: np.ndarray, threshold: float,
                        batch: int = 5000) -> list[tuple[int, int, float]]:
    """Return list of (i, j, cosine) with i < j and cosine >= threshold.

    Assumes embeddings are unit-normalized so ``E @ E.T`` directly gives
    cosine. Processes ``batch`` rows at a time so peak RAM stays at
    ``batch × N × 4 bytes``, not ``N × N × 4``.
    """
    N = embeddings.shape[0]
    pairs: list[tuple[int, int, float]] = []
    if N < 2:
        return pairs
    for i0 in range(0, N, batch):
        i1 = min(i0 + batch, N)
        S = embeddings[i0:i1] @ embeddings.T   # (batch, N) float32
        for bi in range(i1 - i0):
            gi = i0 + bi
            # Only upper triangle (i < j): start at gi+1
            if gi + 1 >= N:
                continue
            row = S[bi, gi + 1:]
            hits = np.nonzero(row >= threshold)[0]
            for offset in hits:
                j = gi + 1 + int(offset)
                pairs.append((gi, j, float(row[offset])))
        del S
    return pairs


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def llm_judge(kimi_key: str,
              node_a: tuple[str, str, str, int],
              node_b: tuple[str, str, str, int],
              cosine: float) -> dict:
    """Ask kimi-for-coding whether two entity nodes refer to the same thing.
    Returns {"same_entity": bool, "reason": str, "model": str}.
    """
    a_id, a_name, a_canon, a_deg = node_a
    b_id, b_name, b_canon, b_deg = node_b
    prompt = (
        "You are judging whether two knowledge-graph entity nodes refer "
        "to the same real-world concept / method / dataset. Respond with "
        "strict JSON only: {\"same_entity\": true|false, \"reason\": \"...\"}.\n\n"
        f"Entity A (degree={a_deg}):\n"
        f"  name: {a_name}\n  canonical: {a_canon}\n\n"
        f"Entity B (degree={b_deg}):\n"
        f"  name: {b_name}\n  canonical: {b_canon}\n\n"
        f"Cosine similarity: {cosine:.3f}\n\n"
        "Rules:\n"
        "- Different instances of the same concept family (e.g. ISCC vs ISAC) "
        "  are NOT the same entity.\n"
        "- Different phrasings / abbreviations of identical concept ARE same.\n"
        "- Subset / superset relationships are NOT 'same' (e.g. 'Transformer' "
        "  vs 'Encoder-only Transformer').\n"
        "Output ONLY the JSON object, no other text."
    )
    payload = json.dumps({
        "model": "kimi-for-coding",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.kimi.com/coding/v1/messages",
        data=payload,
        headers={
            "x-api-key": kimi_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        text = resp["content"][0]["text"].strip()
        # Defensive: strip fences if kimi emits ```json...```
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        j = json.loads(text)
        return {
            "same_entity": bool(j.get("same_entity", False)),
            "reason": str(j.get("reason", ""))[:500],
            "model": "kimi-for-coding",
        }
    except Exception as e:
        return {"same_entity": False,
                "reason": f"judge_err:{type(e).__name__}:{e}"[:500],
                "model": "kimi-for-coding"}


# ---------------------------------------------------------------------------
# Neo4j SAME_AS writer (canonical min→max direction)
# ---------------------------------------------------------------------------
def write_same_as(ne: Neo4jWriter,
                  label: str,
                  key_a: str, key_b: str,
                  cosine: float, source_tag: str,
                  embed_model: str) -> None:
    """MERGE SAME_AS between two entity nodes (Concept / Method / Dataset),
    with deterministic min→max canonical_name direction. Both endpoints must
    carry the same label (v1 intra-label only; cross-label punt per §5.2).

    Idempotent; source_tags accumulates via apoc.coll.toSet.
    """
    key_lo, key_hi = (key_a, key_b) if key_a < key_b else (key_b, key_a)
    with ne.driver.session() as s:
        s.run(
            f"""
            MATCH (a:{label} {{canonical_name: $lo}})
            MATCH (b:{label} {{canonical_name: $hi}})
            MERGE (a)-[r:SAME_AS]->(b)
              ON CREATE SET r.source_tags = [$tag],
                            r.cosine = $cos,
                            r.embed_model = $em,
                            r.created_at = timestamp()
              ON MATCH  SET r.source_tags = apoc.coll.toSet(
                              coalesce(r.source_tags, []) + [$tag])
            """,
            lo=key_lo, hi=key_hi, cos=cosine, tag=source_tag, em=embed_model,
        )


def pair_key(id_a: str, id_b: str) -> str:
    lo, hi = (id_a, id_b) if id_a < id_b else (id_b, id_a)
    return f"{lo}::{hi}"


def log_edge_written(conn, pk: str, kind: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO t2_edge_log (pair_key, embed_model, kind, written_at) "
            "VALUES (?,?,?,?)",
            (pk, EMBED_MODEL, kind, time.time()),
        )


def was_edge_written(conn, pk: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM t2_edge_log WHERE pair_key=? AND embed_model=?",
        (pk, EMBED_MODEL),
    ).fetchone()
    return r is not None


def get_cached_judge(conn, pk: str) -> dict | None:
    """Return prior LLM judge verdict for pair_key, or None if uncached /
    unusable. Skips rows with reason prefixed ``judge_err:`` — those record
    transient LLM-call failures (see ``llm_judge`` exception handler), not
    real verdicts, so callers must retry.

    The judge is keyed on pair_key only — stable across embed_model changes
    because ``llm_judge`` decides on entity name / canonical (not embedding).
    """
    r = conn.execute(
        "SELECT same_entity, reason, model FROM t2_llm_judge WHERE pair_key=?",
        (pk,),
    ).fetchone()
    if r is None:
        return None
    same_entity, reason, model = r
    if reason and str(reason).startswith("judge_err:"):
        return None
    return {
        "same_entity": bool(same_entity),
        "reason": str(reason) if reason is not None else "",
        "model": str(model) if model is not None else "",
    }


# ---------------------------------------------------------------------------
# Dataset exact-match pre-pass (spec §8 finding 8)
# ---------------------------------------------------------------------------
def dataset_exact_match_pairs(nodes: list[tuple[str, str, str, int]]
                                ) -> list[tuple[int, int, str]]:
    """Return (i, j, 'exact') for Dataset nodes where _collapse(name) matches.
    Short-circuits the embed path for CIFAR-10 ↔ CIFAR10 type cases."""
    buckets: dict[str, list[int]] = {}
    for i, (_id, name, _canon, _deg) in enumerate(nodes):
        key = _collapse(name)
        if not key:
            continue
        buckets.setdefault(key, []).append(i)
    pairs = []
    for key, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b], "exact"))
    return pairs


# ---------------------------------------------------------------------------
# Main per-label pipeline
# ---------------------------------------------------------------------------
def process_label(label: str, conn, ne: Neo4jWriter,
                    dashscope_key: str, kimi_key: str,
                    limit: int | None, dry_run: bool,
                    stats: dict) -> None:
    logger.info("=== label=%s ===", label)
    nodes = load_label_nodes(ne, label, limit=limit)
    stats[f"{label}_nodes"] = len(nodes)
    logger.info("loaded %d nodes", len(nodes))
    if len(nodes) < 2:
        logger.info("  < 2 nodes, skip")
        return

    # Step 1: Dataset exact-match shortcut (cheap, runs before embed)
    exact_pairs: list[tuple[int, int, str]] = []
    if label == "Dataset":
        exact_pairs = dataset_exact_match_pairs(nodes)
        stats[f"{label}_exact_pairs"] = len(exact_pairs)
        logger.info("  Dataset exact-match pairs: %d", len(exact_pairs))

    # Step 2: embed (cache-aware). Runs in both dry-run and full mode —
    # dry-run only skips the expensive LLM-judge + Neo4j-write later.
    inputs = _embed_inputs_for(nodes)
    embeddings = embed_nodes_with_cache(
        conn, label, nodes, inputs, dashscope_key,
    )

    # Step 3: batched GEMM, threshold TH_AMBIG (the floor)
    sim_pairs = find_similar_pairs(embeddings, threshold=TH_AMBIG, batch=5000)
    stats[f"{label}_candidate_pairs"] = len(sim_pairs)
    logger.info("  candidate pairs cos>=%.2f: %d", TH_AMBIG, len(sim_pairs))

    # Step 4: classify each candidate + maybe LLM judge + write edge
    judged = 0
    judged_from_cache = 0
    auto = 0
    rejected = 0
    hub_forced_judge = 0
    errors = 0

    # Unified list: (i, j, cos, kind) — exact first, then sim
    all_pairs: list[tuple[int, int, float, str]] = []
    for i, j, _kind in exact_pairs:
        all_pairs.append((i, j, 1.0, "exact"))
    for i, j, cos in sim_pairs:
        all_pairs.append((i, j, cos, "sim"))

    for i, j, cos, kind in all_pairs:
        node_a, node_b = nodes[i], nodes[j]
        pk = pair_key(node_a[0], node_b[0])
        if was_edge_written(conn, pk):
            continue   # idempotent

        # Decide action
        if kind == "exact":
            action = "write"
            src_tag = "exact_match"
        else:
            # Adjust threshold if CJK or hub
            cjk = _has_cjk(node_a[1]) or _has_cjk(node_b[1])
            th_auto = TH_AUTO_CJK if cjk else TH_AUTO
            hub = node_a[3] > HUB_DEGREE or node_b[3] > HUB_DEGREE
            if cos >= th_auto and not hub:
                action, src_tag = "write", "embed_auto"
            elif cos >= TH_AMBIG:
                if hub and cos >= th_auto:
                    hub_forced_judge += 1
                if dry_run:
                    # Skip judge in dry run; just count as pending
                    stats[f"{label}_dryrun_judge_skipped"] = \
                        stats.get(f"{label}_dryrun_judge_skipped", 0) + 1
                    continue
                # LLM judge — check cache first (prior verdicts are stable
                # across reruns; ``judge_err:`` rows skipped by helper).
                cached = get_cached_judge(conn, pk)
                if cached is not None:
                    verdict = cached
                    judged_from_cache += 1
                else:
                    verdict = llm_judge(kimi_key, node_a, node_b, cos)
                    judged += 1
                    with conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO t2_llm_judge "
                            "(pair_key, label_a, label_b, cosine, same_entity, "
                            " reason, model, judged_at) VALUES (?,?,?,?,?,?,?,?)",
                            (pk, label, label, cos,
                             int(verdict["same_entity"]), verdict["reason"],
                             verdict["model"], time.time()),
                        )
                if verdict["same_entity"]:
                    action, src_tag = "write", "llm_judge"
                else:
                    action, src_tag = "skip", None
                    rejected += 1
            else:
                action = "skip"
                src_tag = None

        if action == "write":
            if dry_run:
                stats[f"{label}_dryrun_would_write"] = \
                    stats.get(f"{label}_dryrun_would_write", 0) + 1
                continue
            try:
                write_same_as(ne, label, node_a[0], node_b[0],
                                cos, src_tag, EMBED_MODEL)
                log_edge_written(conn, pk, src_tag)
                auto += 1
            except Exception as e:
                errors += 1
                logger.warning("neo4j err on pair %s: %s", pk, e)
                _record_failure(conn, "neo4j_merge", pk, str(e))

    stats[f"{label}_edges_written"] = auto
    stats[f"{label}_llm_judged"] = judged
    stats[f"{label}_llm_judged_from_cache"] = judged_from_cache
    stats[f"{label}_llm_rejected"] = rejected
    stats[f"{label}_hub_forced_judge"] = hub_forced_judge
    stats[f"{label}_errors"] = errors
    logger.info(
        "  done: wrote %d / judged %d (+%d from cache) / rejected %d / "
        "hub-judge %d / err %d",
        auto, judged, judged_from_cache, rejected, hub_forced_judge, errors,
    )


def run(args) -> None:
    # Validate labels BEFORE anything else (Finding 2 — Cypher injection).
    for label in args.labels:
        _validate_label(label)

    conn = _init_progress_db()
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or ""
    if not dashscope_key:
        logger.error("DASHSCOPE_API_KEY missing")
        sys.exit(2)
    kimi_key = os.environ.get("KIMI_API_KEY") or ""
    if not kimi_key and not args.dry_run:
        logger.error("KIMI_API_KEY missing (needed for LLM judge)")
        sys.exit(2)

    ne = Neo4jWriter(
        os.environ["NEO4J_ZOTERO_URI"],
        os.environ.get("NEO4J_ZOTERO_USER", "neo4j"),
        os.environ["NEO4J_ZOTERO_PASSWORD"],
    )
    stats: dict = {}
    t0 = time.time()
    try:
        for label in args.labels:
            process_label(label, conn, ne, dashscope_key, kimi_key,
                            args.limit, args.dry_run, stats)
    finally:
        ne.close()
        conn.close()

    elapsed = time.time() - t0
    logger.info("DONE in %.0fs. stats=%s", elapsed, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                    help="Which labels to process (default: Concept Method Dataset)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N nodes per label (smoke/calibration)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip LLM judge + Neo4j writes; only compute + cache embeddings")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

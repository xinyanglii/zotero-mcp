"""Tests for the concurrent multi-provider LLM judge path in
scripts/canonicalize_entities.py (2026-04-23).

Covers:
- Z.AI OpenAI-compat request shape + response parse
- Kimi Anthropic-compat request shape (regression after refactor)
- Provider dispatcher routing (known / unknown)
- Round-robin provider picker (single-thread + thread-safety)
- Env-driven provider list parsing (default / override / invalid filtering)
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import canonicalize_entities as t2  # noqa: E402


NODE_A = ("id-a", "attention mechanism", "attention mechanism", 10)
NODE_B = ("id-b", "Attention", "attention", 5)


# ---------------------------------------------------------------------------
# Provider impls (Zai + Kimi) — request shape + response parse
# ---------------------------------------------------------------------------

class _FakeResp:
    """Minimal urllib response mock that supports context manager + read()."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _zai_response(same: bool, reason: str = "r") -> bytes:
    return json.dumps({
        "choices": [{"message": {"content": json.dumps({
            "same_entity": same, "reason": reason,
        })}}]
    }).encode()


def _kimi_response(same: bool, reason: str = "r") -> bytes:
    return json.dumps({
        "content": [{"text": json.dumps({
            "same_entity": same, "reason": reason,
        })}]
    }).encode()


def test_zai_request_uses_openai_compat_bearer(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=60):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_zai_response(True, "aliases"))

    monkeypatch.setenv("Z_AI_API_KEY", "test-zai-key")
    monkeypatch.setenv("ZAI_JUDGE_MODEL", "glm-5.1")
    monkeypatch.setattr(t2.urllib.request, "urlopen", fake_urlopen)

    v = t2._llm_judge_zai(NODE_A, NODE_B, 0.88)

    # Endpoint + auth style
    assert captured["url"] == "https://api.z.ai/api/coding/paas/v4/chat/completions"
    headers_lower = {k.lower(): val for k, val in captured["headers"].items()}
    assert headers_lower["authorization"] == "Bearer test-zai-key"
    assert headers_lower["content-type"] == "application/json"
    # OpenAI-compat body
    assert captured["body"]["model"] == "glm-5.1"
    assert captured["body"]["messages"][0]["role"] == "user"
    assert "Entity A" in captured["body"]["messages"][0]["content"]
    assert "Cosine similarity: 0.880" in captured["body"]["messages"][0]["content"]
    # Verdict parse
    assert v == {"same_entity": True, "reason": "aliases", "model": "glm-5.1"}


def test_zai_env_overrides(monkeypatch):
    """ZAI_JUDGE_URL / ZAI_JUDGE_MODEL must be honoured."""
    captured: dict = {}

    def fake_urlopen(req, timeout=60):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_zai_response(False, "different"))

    monkeypatch.setenv("Z_AI_API_KEY", "x")
    monkeypatch.setenv("ZAI_JUDGE_URL", "https://custom.example/chat")
    monkeypatch.setenv("ZAI_JUDGE_MODEL", "custom-model")
    monkeypatch.setattr(t2.urllib.request, "urlopen", fake_urlopen)

    v = t2._llm_judge_zai(NODE_A, NODE_B, 0.9)
    assert captured["url"] == "https://custom.example/chat"
    assert captured["body"]["model"] == "custom-model"
    assert v["model"] == "custom-model"
    assert v["same_entity"] is False


def test_zai_http_error_returns_judge_err(monkeypatch):
    from urllib.error import HTTPError

    def fake_urlopen(req, timeout=60):
        raise HTTPError(req.full_url, 429, "rate limit", {}, io.BytesIO(b""))

    monkeypatch.setenv("Z_AI_API_KEY", "x")
    monkeypatch.setattr(t2.urllib.request, "urlopen", fake_urlopen)

    v = t2._llm_judge_zai(NODE_A, NODE_B, 0.88)
    assert v["same_entity"] is False
    assert v["reason"].startswith("judge_err:HTTPError:")
    # Downstream `get_cached_judge` uses this prefix to force retry.
    assert v["model"] == os.getenv("ZAI_JUDGE_MODEL", "glm-5.1")


def test_zai_reasoning_truncation_returns_judge_err(monkeypatch):
    """GLM-5.x reasoning models may spend the full token budget on
    ``reasoning_content`` and truncate ``message.content`` mid-JSON
    (``finish_reason: length``). The partial-JSON parse must fail into
    the ``judge_err:`` path so next run retries."""
    truncated = json.dumps({
        "choices": [{
            "finish_reason": "length",
            "message": {"content": '{"same',
                        "reasoning_content": "x" * 2000}
        }]
    }).encode()

    monkeypatch.setenv("Z_AI_API_KEY", "x")
    monkeypatch.setattr(t2.urllib.request, "urlopen",
                        lambda req, timeout=60: _FakeResp(truncated))
    v = t2._llm_judge_zai(NODE_A, NODE_B, 0.88)
    assert v["same_entity"] is False
    assert v["reason"].startswith("judge_err:")


def test_zai_missing_content_key_returns_judge_err(monkeypatch):
    """Response with no ``choices[0].message.content`` (API edge case,
    empty response, provider-side bug) → judge_err, not uncaught KeyError."""
    payloads = [
        b'{"choices": [{"message": {}}]}',            # content key missing
        b'{"choices": []}',                            # no choices
        b'{}',                                         # no choices key
    ]
    for body in payloads:
        monkeypatch.setenv("Z_AI_API_KEY", "x")
        monkeypatch.setattr(t2.urllib.request, "urlopen",
                            lambda req, timeout=60, _b=body: _FakeResp(_b))
        v = t2._llm_judge_zai(NODE_A, NODE_B, 0.9)
        assert v["same_entity"] is False, f"body={body!r} → {v}"
        assert v["reason"].startswith("judge_err:"), f"body={body!r} → {v}"


def test_zai_fenced_json_parses(monkeypatch):
    """Model may emit ```json ...``` fences; _strip_json_fences handles."""
    body = json.dumps({
        "choices": [{"message": {"content":
            '```json\n{"same_entity": true, "reason": "ok"}\n```'}}]
    }).encode()

    monkeypatch.setenv("Z_AI_API_KEY", "x")
    monkeypatch.setattr(t2.urllib.request, "urlopen",
                        lambda req, timeout=60: _FakeResp(body))
    v = t2._llm_judge_zai(NODE_A, NODE_B, 0.9)
    assert v["same_entity"] is True


def test_kimi_request_uses_anthropic_compat(monkeypatch):
    """Regression: after refactor, Kimi still hits Anthropic-compat endpoint."""
    captured: dict = {}

    def fake_urlopen(req, timeout=60):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_kimi_response(True, "ok"))

    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setattr(t2.urllib.request, "urlopen", fake_urlopen)

    v = t2._llm_judge_kimi(NODE_A, NODE_B, 0.9)
    assert captured["url"] == "https://api.kimi.com/coding/v1/messages"
    headers_lower = {k.lower(): val for k, val in captured["headers"].items()}
    assert headers_lower["x-api-key"] == "test-kimi-key"
    assert headers_lower["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "kimi-for-coding"
    assert v == {"same_entity": True, "reason": "ok", "model": "kimi-for-coding"}


# ---------------------------------------------------------------------------
# _dispatch_judge routing
# ---------------------------------------------------------------------------

def test_dispatch_routes_kimi(monkeypatch):
    sentinel = {"same_entity": True, "reason": "kimi-called", "model": "kimi-for-coding"}
    monkeypatch.setitem(t2._PROVIDERS_IMPL, "kimi",
                        lambda a, b, c: sentinel)
    out = t2._dispatch_judge(NODE_A, NODE_B, 0.9, "kimi")
    assert out is sentinel


def test_dispatch_routes_zai(monkeypatch):
    sentinel = {"same_entity": False, "reason": "zai-called", "model": "glm-5.1"}
    monkeypatch.setitem(t2._PROVIDERS_IMPL, "zai",
                        lambda a, b, c: sentinel)
    out = t2._dispatch_judge(NODE_A, NODE_B, 0.9, "zai")
    assert out is sentinel


def test_dispatch_unknown_provider_returns_judge_err():
    v = t2._dispatch_judge(NODE_A, NODE_B, 0.9, "no-such-provider")
    assert v["same_entity"] is False
    assert v["reason"] == "judge_err:unknown_provider:no-such-provider"
    # Still matches judge_err: prefix so get_cached_judge will retry next run.
    assert v["reason"].startswith("judge_err:")


# ---------------------------------------------------------------------------
# Round-robin + thread safety
# ---------------------------------------------------------------------------

def test_pick_provider_round_robin_basic():
    # Reset the module-level counter so order is deterministic.
    t2._provider_counter = __import__("itertools").count()
    providers = ["kimi", "zai"]
    picks = [t2._pick_provider(providers) for _ in range(6)]
    assert picks == ["kimi", "zai", "kimi", "zai", "kimi", "zai"]


def test_pick_provider_round_robin_three():
    t2._provider_counter = __import__("itertools").count()
    providers = ["kimi", "zai", "other"]
    picks = [t2._pick_provider(providers) for _ in range(6)]
    assert picks == ["kimi", "zai", "other", "kimi", "zai", "other"]


def test_pick_provider_is_thread_safe():
    """itertools.count() next() is atomic under GIL; total assigned should
    equal total calls with roughly even distribution."""
    t2._provider_counter = __import__("itertools").count()
    providers = ["kimi", "zai"]
    counts = {"kimi": 0, "zai": 0}
    lock = threading.Lock()
    N_THREADS = 8
    PER_THREAD = 250

    def worker():
        local = {"kimi": 0, "zai": 0}
        for _ in range(PER_THREAD):
            local[t2._pick_provider(providers)] += 1
        with lock:
            counts["kimi"] += local["kimi"]
            counts["zai"] += local["zai"]

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = N_THREADS * PER_THREAD
    assert counts["kimi"] + counts["zai"] == total
    # Even split (2 providers) — tolerate ±10% skew
    assert abs(counts["kimi"] - total / 2) < total * 0.10


# ---------------------------------------------------------------------------
# _provider_list env parsing
# ---------------------------------------------------------------------------

def test_provider_list_default(monkeypatch):
    monkeypatch.delenv("T2_PROVIDERS", raising=False)
    assert t2._provider_list() == ["kimi", "zai"]


def test_provider_list_single(monkeypatch):
    monkeypatch.setenv("T2_PROVIDERS", "kimi")
    assert t2._provider_list() == ["kimi"]


def test_provider_list_with_spaces(monkeypatch):
    monkeypatch.setenv("T2_PROVIDERS", " zai , kimi ")
    assert t2._provider_list() == ["zai", "kimi"]


def test_provider_list_filters_invalid(monkeypatch):
    """Unknown provider names are silently dropped; list stays valid."""
    monkeypatch.setenv("T2_PROVIDERS", "kimi,bogus,zai")
    assert t2._provider_list() == ["kimi", "zai"]


def test_provider_list_all_invalid_falls_back_to_kimi(monkeypatch):
    """If env is entirely invalid, fall back to kimi rather than empty list
    (empty list would crash round-robin modulo)."""
    monkeypatch.setenv("T2_PROVIDERS", "bogus,alsobogus")
    assert t2._provider_list() == ["kimi"]


def test_provider_list_empty_env_defaults(monkeypatch):
    monkeypatch.setenv("T2_PROVIDERS", "")
    assert t2._provider_list() == ["kimi", "zai"]


# ---------------------------------------------------------------------------
# Legacy llm_judge() shim
# ---------------------------------------------------------------------------

def test_prompt_contains_invariants():
    """Guard the judge prompt against silent drift. If you tweak the
    instructions, update this test consciously — both providers share it,
    so a drift here silently changes **every** verdict."""
    p = t2._build_judge_prompt(NODE_A, NODE_B, 0.87)
    # Inputs appear in prompt
    assert NODE_A[1] in p  # name_a
    assert NODE_B[1] in p  # name_b
    assert NODE_A[2] in p  # canonical_a
    assert NODE_B[2] in p  # canonical_b
    assert "degree=10" in p and "degree=5" in p
    assert "Cosine similarity: 0.870" in p
    # Core rules that differentiate judgments
    assert "ISCC vs ISAC" in p                   # family ≠ same-entity rule
    assert "abbreviations of identical" in p    # abbrev = same rule
    assert "Transformer" in p                    # subset/superset rule
    assert "Output ONLY the JSON object" in p    # response shape discipline


def test_legacy_llm_judge_ignores_kimi_key_arg(monkeypatch):
    """The old `llm_judge(kimi_key, ...)` signature must keep working, but
    ignore the positional key and read env inside the provider impl."""
    calls = []
    monkeypatch.setitem(t2._PROVIDERS_IMPL, "kimi",
                        lambda a, b, c: (calls.append((a, b, c))
                                         or {"same_entity": True, "reason": "x",
                                             "model": "kimi-for-coding"}))
    v = t2.llm_judge("unused-legacy-key", NODE_A, NODE_B, 0.91)
    assert v["model"] == "kimi-for-coding"
    assert calls == [(NODE_A, NODE_B, 0.91)]

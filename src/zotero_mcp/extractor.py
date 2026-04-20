"""
Structured extractor for academic papers.

Given a paper's markdown (produced by MinerU/markitdown), call an LLM with a
rigid JSON schema and return a validated `ExtractedPaper` pydantic object. This
is the bridge between raw text and the Neo4j/Qdrant ingest layer (M3).

Default provider: **Kimi coding subscription** (Anthropic-compatible endpoint
at `api.kimi.com/coding/v1/messages`). Override via env for other providers
(Moonshot platform, DashScope/Qwen, DeepSeek, OpenAI). Canonical reference:
`~/workspace/claude-workspace/.claude/skills/daily-ops-morning-briefing/scripts/helper_llm.py`.

Env vars:
    HELPER_LLM_URL       endpoint (default: https://api.kimi.com/coding/v1/messages)
    HELPER_LLM_MODEL     model name (default: kimi-for-coding)
    HELPER_LLM_API_KEY   API key (or KIMI_API_KEY fallback)
    HELPER_LLM_PROVIDER  "anthropic" | "openai" (default: "anthropic")

Design (see plan §3.1):
- Minimal 6-entity-class schema (Paper/Method/Concept/Dataset/Author/Venue).
- `references` field carries `role` (baseline | prior-work | related |
  motivation | contrast | extends | contradicts).
- Up to 2 retries on JSON-parse failures; persistent failures logged to
  `schema_errors.jsonl`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

MAX_INPUT_TOKENS = int(os.getenv("EXTRACTOR_MAX_INPUT_TOKENS", "180000"))
SCHEMA_ERR_LOG = Path(os.getenv(
    "ZOTERO_SCHEMA_ERR_LOG",
    str(Path.home() / ".cache" / "zotero-mcp" / "schema_errors.jsonl"),
))

# Provider chain: tried in order per call. On recoverable errors (HTTP 429,
# 402 quota, 5xx) we move to the next provider. This lets Kimi burn subscription
# quota as primary, and DashScope/Qwen take over when Kimi's rolling window is
# saturated. The chain is defined in code (not env) so the fallback is always on
# by default; override via EXTRACTOR_DISABLE_FALLBACK=1 if you want single-provider.
PROVIDER_CHAIN = [
    {
        "name":      "kimi",
        "url":       os.getenv("KIMI_URL", "https://api.kimi.com/coding/v1/messages"),
        "model":     os.getenv("KIMI_MODEL", "kimi-for-coding"),
        "key_envs":  ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "protocol":  "anthropic",  # x-api-key header, Anthropic message schema
    },
    {
        "name":      "qwen",
        "url":       os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        "model":     os.getenv("QWEN_MODEL", "qwen3-max"),
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
]


def _active_providers() -> list[dict]:
    if os.environ.get("EXTRACTOR_DISABLE_FALLBACK"):
        return PROVIDER_CHAIN[:1]
    if "HELPER_LLM_URL" in os.environ:
        # legacy single-provider override — respect it, no fallback
        return [{
            "name": "env-override",
            "url":  os.environ["HELPER_LLM_URL"],
            "model": os.environ.get("HELPER_LLM_MODEL", "kimi-for-coding"),
            "key_envs": ("HELPER_LLM_API_KEY", "KIMI_API_KEY"),
            "protocol": os.environ.get("HELPER_LLM_PROVIDER", "anthropic").lower(),
        }]
    return PROVIDER_CHAIN


# ---------- schema ----------
# Canonical role enum. LLM outputs are normalized into this set via a
# validator rather than failing — real-world extraction produces synonyms
# like "used" / "cited" / "benchmark" which we coerce to the nearest
# canonical role instead of rejecting the whole paper.
CANONICAL_ROLES = {
    "baseline", "prior-work", "related", "motivation",
    "contrast", "extends", "contradicts",
}
ROLE_ALIASES = {
    "used": "related", "use": "related", "uses": "related",
    "applies": "related", "applied": "related",
    "cited": "related", "citation": "related", "cite": "related",
    "benchmark": "baseline", "comparison": "baseline", "compared": "baseline",
    "reference": "related", "references": "related",
    "background": "prior-work", "foundation": "prior-work",
    "prior": "prior-work", "priorwork": "prior-work", "prior_work": "prior-work",
    "improves": "extends", "extension": "extends",
    "disagrees": "contradicts", "disagrees-with": "contradicts",
    "contrasts": "contrast",
    "motivates": "motivation",
}
ContributionType = Literal["theory", "method", "system", "survey", "dataset", "tool"]


class MethodImproved(BaseModel):
    base: str
    aspect: str


class Reference(BaseModel):
    cited_title: str
    cited_author_year: str
    role: str
    context_quote: str = ""

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        if not isinstance(v, str):
            return "related"
        s = v.strip().lower().replace(" ", "-")
        if s in CANONICAL_ROLES:
            return s
        return ROLE_ALIASES.get(s.replace("-", "_"), ROLE_ALIASES.get(s, "related"))

    @field_validator("context_quote", mode="before")
    @classmethod
    def _trunc_quote(cls, v):
        if not isinstance(v, str):
            return ""
        return v[:240]


class ExtractedPaper(BaseModel):
    paper_id: str
    title: str
    tldr: str = ""
    problem: str = ""
    methods_used: list[str] = Field(default_factory=list)
    methods_proposed: list[str] = Field(default_factory=list)
    methods_improved: list[MethodImproved] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    key_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    contribution_type: ContributionType | None = None

    @field_validator("tldr", "problem", mode="before")
    @classmethod
    def _trunc_str(cls, v):
        if not isinstance(v, str):
            return ""
        return v[:500]

    @field_validator("contribution_type", mode="before")
    @classmethod
    def _norm_ctype(cls, v):
        if not isinstance(v, str):
            return None
        s = v.strip().lower()
        return s if s in {"theory","method","system","survey","dataset","tool"} else None


SYSTEM_PROMPT = """你是学术文献结构化抽取助手。读以下论文 Markdown，严格按 JSON schema 输出。

规则：
- references 每条必须标 role，role ∈ {baseline, prior-work, related, motivation, contrast, extends, contradicts}；同义词会被系统归一，但**你输出时也请用标准 role**
- references **最多 25 条**，只挑最重要的（survey paper 可以全挑 top cites；regular paper 挑 baseline+核心对比+动机即可）
- methods_used vs methods_proposed 要区分：paper 自己提出的新方法 → proposed；复用他人的方法 → used
- concepts 控制在 5-10 个，宁少勿滥；不抽形容词、通用词（如 "novel"、"efficient"）
- context_quote 必须是原文 quote **≤240 字符**（硬上限，超过会被截断）
- tldr **≤280 字符**，problem **≤500 字符**
- contribution_type ∈ {theory, method, system, survey, dataset, tool}
- 输出必须是合法的 JSON 对象（**不要 markdown 围栏，不要解释**）

JSON schema:
{
  "paper_id": "<预设>",
  "title": "",
  "tldr": "≤280 chars",
  "problem": "1-2 句核心问题",
  "methods_used": ["..."],
  "methods_proposed": ["..."],
  "methods_improved": [{"base":"...","aspect":"..."}],
  "datasets": ["..."],
  "concepts": ["..."],
  "key_claims": ["2-5 条实验结论"],
  "limitations": ["作者自述"],
  "references": [{"cited_title":"","cited_author_year":"","role":"","context_quote":""}],
  "venue": "",
  "year": 2024,
  "contribution_type": "method"
}"""


def _lookup_key(envs: tuple[str, ...]) -> str | None:
    for e in envs:
        v = os.environ.get(e)
        if v:
            return v
    return None


def _truncate(md: str, budget_tokens: int) -> str:
    # ≈ 3.5 chars/token for mixed zh/en scientific text
    max_chars = int(budget_tokens * 3.5)
    if len(md) <= max_chars:
        return md
    head = md[: max_chars - 500]
    return head + f"\n\n[...truncated {len(md) - max_chars} chars...]"


def _is_fallback_worthy(http_code: int, body: str) -> bool:
    """Return True if this provider's failure is the kind we should try the
    next provider for (quota / capacity / server down), vs a schema/request
    bug where another provider would hit the same wall."""
    if http_code in (401, 403, 429, 402, 500, 502, 503, 504, 529):
        return True
    b = (body or "").lower()
    for marker in ("quota", "rate limit", "too many", "overloaded",
                   "capacity", "insufficient_balance", "billing"):
        if marker in b:
            return True
    return False


def _call_one(provider: dict, user_prompt: str,
              *, max_tokens: int, timeout: int) -> str:
    """Single-provider attempt. Raises urllib.error.HTTPError on HTTP failure
    (body embedded) or ConnectionError on transport failure."""
    api_key = _lookup_key(provider["key_envs"])
    if not api_key:
        raise RuntimeError(f"no API key for provider {provider['name']}: "
                           f"tried {provider['key_envs']}")
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt}]

    if provider["protocol"] == "anthropic":
        payload = json.dumps({
            "model": provider["model"],
            "max_tokens": max_tokens,
            "messages": messages,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:  # openai-compatible
        payload = json.dumps({
            "model": provider["model"],
            "max_tokens": max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    req = urllib.request.Request(provider["url"], data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())

    if provider["protocol"] == "anthropic":
        return data["content"][0]["text"].strip()
    return data["choices"][0]["message"]["content"].strip()


def _call_llm(user_prompt: str, *, max_tokens: int = 16000, timeout: int = 360) -> str:
    """Provider-chain aware call. Yields the first successful response text.

    Raises the last seen HTTPError if every provider fails on a fallback-worthy
    condition, or propagates immediately on a non-fallback error (so we don't
    silently mask schema bugs).
    """
    providers = _active_providers()
    last_exc: Exception | None = None
    for i, p in enumerate(providers):
        try:
            out = _call_one(p, user_prompt, max_tokens=max_tokens, timeout=timeout)
            if i > 0:
                logger.info("extractor: fell back to %s OK", p["name"])
            return out
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            fallback = _is_fallback_worthy(e.code, body)
            logger.warning(
                "extractor: %s HTTP %d%s  body=%s",
                p["name"], e.code,
                " (fallback-worthy)" if fallback else "",
                body[:200],
            )
            last_exc = e
            if not fallback:
                raise
        except RuntimeError as e:
            # missing key etc — skip provider
            logger.warning("extractor: skip %s: %s", p["name"], e)
            last_exc = e
        except Exception as e:
            logger.warning("extractor: %s transport err: %s", p["name"], e)
            last_exc = e
    assert last_exc is not None
    raise last_exc


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first top-level JSON object out of a string.

    Kimi sometimes wraps output in markdown fences or adds a short preamble; be
    lenient. Falls back to raising ValueError.
    """
    text = text.strip()
    # fenced
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # raw JSON
    if text.startswith("{"):
        return json.loads(text)
    # greedy: find outermost braces
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON object in response: {text[:300]}")


def _log_schema_error(paper_id: str, raw: str, err: str) -> None:
    SCHEMA_ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SCHEMA_ERR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "paper_id": paper_id, "error": err,
            "raw_head": raw[:1500], "ts": time.time(),
        }) + "\n")


def extract_structured(
    markdown: str,
    *,
    title: str,
    paper_id: str,
    max_retries: int = 2,
) -> ExtractedPaper | None:
    """Run the LLM on the markdown and return a validated ExtractedPaper.

    Returns None after logging when the model repeatedly fails to produce
    valid JSON — better to skip than corrupt the graph.
    """
    truncated = _truncate(markdown, MAX_INPUT_TOKENS - 4000)
    user_msg = f"paper_id={paper_id}\ntitle={title}\n\n=== Markdown ===\n{truncated}"

    last_err, last_raw = "unknown", ""
    for attempt in range(max_retries + 1):
        try:
            raw = _call_llm(user_msg)
            last_raw = raw
            data = _extract_json_object(raw)
            data["paper_id"] = paper_id
            paper = ExtractedPaper.model_validate(data)
            logger.info("extracted paper_id=%s", paper_id)
            return paper
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:400]
            last_err = f"HTTP {e.code}: {body}"
            logger.warning("extract API err attempt %d: %s", attempt + 1, last_err)
            time.sleep(min(2 ** attempt, 10))
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("extract parse err attempt %d: %s", attempt + 1, last_err)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("extract err attempt %d: %s", attempt + 1, last_err)
            time.sleep(min(2 ** attempt, 10))

    _log_schema_error(paper_id, last_raw, last_err)
    return None

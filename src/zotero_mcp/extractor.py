"""
Structured extractor for academic papers.

Given a paper's markdown (produced by MinerU/markitdown), call an LLM with a
rigid JSON schema and return a validated `ExtractedPaper` pydantic object. This
is the bridge between raw text and the Neo4j/Qdrant ingest layer (M3).

Default provider chain: **Kimi coding subscription** (primary, Anthropic-compat
at `api.kimi.com/coding/v1/messages`) → **Z.AI coding plan GLM-5** (fallback,
OpenAI-compat at `api.z.ai/api/coding/paas/v4/chat/completions`). Override via
env for other providers. Canonical reference:
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
        # Moonshot Kimi coding subscription (currently K2.6). ~17s/paper on
        # real 20KB markdown, highest quality tldr/methods_used. Anthropic
        # protocol. Primary when quota available.
        "url":       os.getenv("KIMI_URL", "https://api.kimi.com/coding/v1/messages"),
        "model":     os.getenv("KIMI_MODEL", "kimi-for-coding"),
        "key_envs":  ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "protocol":  "anthropic",
    },
    # 2026-04-22 02:40 refresh: glm-4.7/5/5.1 free tiers all exhausted (403);
    # qwen3-max / qwen3-max-preview also exhausted. Re-probed DashScope and
    # picked three currently-callable models covering different strengths:
    #   - dashscope-glm-4.6: newest GLM non-stream on DashScope (glm-4.5/4.5-air
    #     stream-only; glm-5.x free tier gone).
    #   - qwen3-coder-plus: coder variant tuned for structured output / JSON,
    #     ideal for our ExtractedPaper schema.
    #   - qwen-plus-latest: Kimi-parity general quality, ¥0.8/1M input — cheap
    #     safety net when coder-plus falters.
    # Burn DashScope free tier FIRST, then fall back to Z.AI GLM-5 direct.
    # 2026-04-22 07:12 refresh: 3 个 qwen3-coder snapshots 今日耗尽已移除；
    # qwen-plus-latest 也删除（走这个在消耗付费额度）。替换为 8 个 qwen-plus
    # dated snapshots，每个独立 1M-token 免费额度（expire 2026-06-24），总计
    # 8M tokens 纯 free 容量。zai（订阅，零边际成本）作为 #2 主力，DashScope
    # snapshots 做深度兜底。排序：newest → oldest。
    {
        "name":      "zai",
        "url":       os.getenv("ZAI_URL", "https://api.z.ai/api/coding/paas/v4/chat/completions"),
        "model":     os.getenv("ZAI_MODEL", "glm-5"),
        "key_envs":  ("Z_AI_API_KEY", "ZAI_API_KEY"),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-1201",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-12-01",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0911",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-09-11",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0728",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-07-28",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0714",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-07-14",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0428",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-04-28",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0125",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-2025-01-25",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-1220",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-1220",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    {
        "name":      "dashscope-qwen-plus-0112",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-0112",
        "key_envs":  ("DASHSCOPE_API_KEY",),
        "protocol":  "openai",
    },
    # qwen-plus-latest：付费 alias（¥0.8/1M in, ¥2/1M out）。8 个 dated snapshot
    # 共 8M 免费 + kimi/zai 订阅之后，这是真·最后兜底，避免 paper 进 extract_err。
    {
        "name":      "dashscope-qwen-plus-latest",
        "url":       "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":     "qwen-plus-latest",
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


# VLM chain for figure-aware extraction. T0 2026-04-22 · grounded against
# real probe tests:
#   - kimi-for-coding on coding endpoint accepts vision (1.3s)
#   - glm-4.6v on Z.AI coding plan accepts vision (3.7s); glm-5v-turbo
#     requires paid recharge, not in current subscription
#   - qwen3-vl-plus on DashScope compatible-mode accepts vision (6.4s)
# ``extract_structured(figures=[...])`` tries this chain first; on all-fail
# (or empty figures) it falls back to the text PROVIDER_CHAIN.
VLM_PROVIDER_CHAIN = [
    {
        "name":     "kimi-vl",
        "url":      os.getenv("KIMI_URL", "https://api.kimi.com/coding/v1/messages"),
        "model":    os.getenv("KIMI_VL_MODEL", "kimi-for-coding"),
        "key_envs": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "protocol": "anthropic-vision",
    },
    {
        "name":     "zai-glm-4.6v",
        "url":      os.getenv("ZAI_URL", "https://api.z.ai/api/coding/paas/v4/chat/completions"),
        "model":    os.getenv("ZAI_VL_MODEL", "glm-4.6v"),
        "key_envs": ("Z_AI_API_KEY", "ZAI_API_KEY"),
        "protocol": "openai-vision",
    },
    {
        "name":     "dashscope-qwen3-vl-plus",
        "url":      "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":    os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus"),
        "key_envs": ("DASHSCOPE_API_KEY",),
        "protocol": "openai-vision",
    },
]

# Cap on how many figures to actually send to the VLM per paper. Everything
# beyond this is still saved to SQLite/WebDAV — only the VLM input is capped
# to keep token budget predictable. 12 × ~1.5k tokens/img ≈ 18k tokens extra.
VLM_MAX_FIGURES = int(os.getenv("VLM_MAX_FIGURES", "12"))


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
    # 2026-04-22: zai glm-5 returns HTTP 400 "Prompt exceeds max length" (code
    # 1261) on long papers. Other providers (qwen-plus) have larger context
    # windows, so fall through instead of aborting the chain.
    for marker in ("quota", "rate limit", "too many", "overloaded",
                   "capacity", "insufficient_balance", "billing",
                   "prompt exceeds", "max length", "context length",
                   "token limit", "input too long", "input is too long"):
        if marker in b:
            return True
    return False


def _call_one(provider: dict, user_prompt: str,
              *, max_tokens: int, timeout: int,
              figures: list[dict] | None = None) -> str:
    """Single-provider attempt. Raises urllib.error.HTTPError on HTTP failure
    (body embedded) or ConnectionError on transport failure.

    ``figures`` (optional): list of ``{"bytes": raw_jpeg_bytes, "caption": str}``
    entries. Only consumed when ``provider["protocol"]`` is one of the
    ``*-vision`` variants, in which case the content is built as a multimodal
    message (text + image parts). For text-only protocols figures are ignored.
    """
    import base64 as _b64
    api_key = _lookup_key(provider["key_envs"])
    if not api_key:
        raise RuntimeError(f"no API key for provider {provider['name']}: "
                           f"tried {provider['key_envs']}")
    proto = provider["protocol"]
    prompt_text = SYSTEM_PROMPT + "\n\n" + user_prompt

    # ---- Build `messages` per protocol ----
    if proto == "anthropic":
        messages = [{"role": "user", "content": prompt_text}]
    elif proto == "openai":
        messages = [{"role": "user", "content": prompt_text}]
    elif proto == "anthropic-vision":
        content = [{"type": "text", "text": prompt_text}]
        for f in (figures or [])[:VLM_MAX_FIGURES]:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64", "media_type": f.get("mime", "image/jpeg"),
                    "data": _b64.b64encode(f["bytes"]).decode(),
                },
            })
        content.append({"type": "text", "text": VISION_TAIL_PROMPT})
        messages = [{"role": "user", "content": content}]
    elif proto == "openai-vision":
        content = [{"type": "text", "text": prompt_text}]
        for f in (figures or [])[:VLM_MAX_FIGURES]:
            b64 = _b64.b64encode(f["bytes"]).decode()
            mime = f.get("mime", "image/jpeg")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": VISION_TAIL_PROMPT})
        messages = [{"role": "user", "content": content}]
    else:
        raise RuntimeError(f"unknown provider protocol: {proto!r}")

    # ---- Build payload + headers per protocol family ----
    if proto in ("anthropic", "anthropic-vision"):
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
    else:  # openai / openai-vision
        body: dict[str, Any] = {
            "model": provider["model"],
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": 0.2,
        }
        # response_format=json_object is text-only in most providers. For
        # VL dashscope/Z.AI the hint is sometimes rejected with 400 — only
        # add for pure text protocol, VLM output gets parsed by
        # _extract_json_object either way.
        if proto == "openai":
            body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    req = urllib.request.Request(provider["url"], data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())

    if proto in ("anthropic", "anthropic-vision"):
        # Anthropic vision puts text responses as content[0].text (same as text)
        return data["content"][0]["text"].strip()
    return data["choices"][0]["message"]["content"].strip()


VISION_TAIL_PROMPT = (
    "\n\nThe images above are figures extracted from the paper PDF. If any "
    "show architectures / flowcharts / experiment curves / comparison tables, "
    "describe their **visual structure** (components, hierarchy, axes meaning, "
    "compared items) when populating the `methods_proposed` and `results` "
    "fields of the JSON — do not merely copy the figure caption. Return the "
    "same JSON schema as for text-only extraction."
)


def _call_llm_chain(
    providers: list[dict], user_prompt: str,
    *,
    figures: list[dict] | None = None,
    max_tokens: int = 16000, timeout: int = 360,
    chain_label: str = "extractor",
) -> tuple[str, str]:
    """Try ``providers`` in order; return ``(response_text, provider_name)``
    on first success. Raises the last exception if every provider fails on a
    fallback-worthy condition, or propagates a non-fallback error immediately.

    ``figures`` is threaded through to ``_call_one`` — only providers with a
    ``*-vision`` protocol actually use them; text providers ignore.
    """
    # Once-per-process log: which chain is active.
    key = f"_logged_chain_{chain_label}"
    if not getattr(_call_llm_chain, key, False):
        logger.info("%s active providers: %s",
                    chain_label, [p["name"] for p in providers])
        setattr(_call_llm_chain, key, True)
    last_exc: Exception | None = None
    for i, p in enumerate(providers):
        try:
            out = _call_one(
                p, user_prompt,
                max_tokens=max_tokens, timeout=timeout, figures=figures,
            )
            if i > 0:
                logger.info("%s: fell back to %s OK", chain_label, p["name"])
            return out, p["name"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            finally:
                # Close the HTTPError's underlying socket. urllib raises
                # HTTPError during urlopen() (before `with ... as r:` binds)
                # so `with` never runs — the socket stays in CLOSE_WAIT
                # until GC unless we close explicitly. Hundreds of 429s
                # add up to 359-socket leaks against CF-fronted endpoints.
                try:
                    e.close()
                except Exception:
                    pass
            fallback = _is_fallback_worthy(e.code, body)
            logger.warning(
                "%s: %s HTTP %d%s  body=%s",
                chain_label, p["name"], e.code,
                " (fallback-worthy)" if fallback else "",
                body[:200],
            )
            last_exc = e
            if not fallback:
                raise
        except RuntimeError as e:
            # missing key etc — skip provider
            logger.warning("%s: skip %s: %s", chain_label, p["name"], e)
            last_exc = e
        except Exception as e:
            logger.warning("%s: %s transport err: %s", chain_label, p["name"], e)
            last_exc = e
    assert last_exc is not None
    raise last_exc


def _call_llm(user_prompt: str, *, max_tokens: int = 16000, timeout: int = 360) -> str:
    """Legacy text-only entry. Delegates to ``_call_llm_chain`` with the text
    PROVIDER_CHAIN and drops the provider-name for back-compat callers."""
    text, _name = _call_llm_chain(
        _active_providers(), user_prompt,
        max_tokens=max_tokens, timeout=timeout, chain_label="extractor",
    )
    return text


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
    figures: list[dict] | None = None,
) -> tuple[ExtractedPaper | None, str | None]:
    """Run the LLM on the markdown and return ``(paper, provider_name)``.

    - ``figures=None`` or ``figures=[]``  → text PROVIDER_CHAIN only.
    - ``figures=[{"bytes": b"...", ...}]`` → VLM_PROVIDER_CHAIN first; on
      *all-fail* (every VLM provider exhausts its fallback paths) falls back
      to the text chain (paper still ingests, just without figure context).

    Returns ``(None, None)`` after logging when every chain + retry fails to
    produce valid JSON — better to skip than corrupt the graph. Callers use
    the returned ``provider_name`` as ``papers.extract_provider`` for audit.
    """
    truncated = _truncate(markdown, MAX_INPUT_TOKENS - 4000)
    user_msg = f"paper_id={paper_id}\ntitle={title}\n\n=== Markdown ===\n{truncated}"
    wants_vlm = bool(figures)
    last_err, last_raw = "unknown", ""

    for attempt in range(max_retries + 1):
        try:
            raw, provider_name = _run_extract_call(user_msg, figures if wants_vlm else None)
            last_raw = raw
            data = _extract_json_object(raw)
            data["paper_id"] = paper_id
            paper = ExtractedPaper.model_validate(data)
            logger.info("extracted paper_id=%s via %s", paper_id, provider_name)
            return paper, provider_name
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
    return None, None


def _run_extract_call(
    user_msg: str, figures: list[dict] | None,
) -> tuple[str, str]:
    """One attempt of extract. If ``figures`` provided, try VLM chain first;
    on VLM-all-fail (every provider exhausted), fall back to text chain.
    Returns ``(raw_response_text, provider_name)``."""
    if figures:
        try:
            return _call_llm_chain(
                VLM_PROVIDER_CHAIN, user_msg, figures=figures,
                chain_label="vlm-extractor",
            )
        except Exception as e:
            # Any VLM-chain exhaustion (HTTP 4xx/5xx, RuntimeError for missing
            # keys, transport errors) → degrade to text chain so the paper
            # still ingests with a less rich but valid JSON. Log the concrete
            # error for diagnosis. Spec §4 "VLM 全挂 → 降级纯文本".
            if isinstance(e, urllib.error.HTTPError):
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                logger.warning(
                    "VLM chain exhausted (HTTP %d: %s), degrading to text chain",
                    e.code, body)
            else:
                logger.warning(
                    "VLM chain exhausted (%s: %s), degrading to text chain",
                    type(e).__name__, e)
        text, name = _call_llm_chain(
            _active_providers(), user_msg, chain_label="extractor",
        )
        return text, f"{name}+text-fallback"
    # No figures → text chain directly.
    return _call_llm_chain(
        _active_providers(), user_msg, chain_label="extractor",
    )

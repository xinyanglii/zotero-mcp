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

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

HELPER_URL = os.getenv("HELPER_LLM_URL", "https://api.kimi.com/coding/v1/messages")
HELPER_MODEL = os.getenv("HELPER_LLM_MODEL", "kimi-for-coding")
HELPER_PROVIDER = os.getenv("HELPER_LLM_PROVIDER", "anthropic").lower()
MAX_INPUT_TOKENS = int(os.getenv("EXTRACTOR_MAX_INPUT_TOKENS", "180000"))
SCHEMA_ERR_LOG = Path(os.getenv(
    "ZOTERO_SCHEMA_ERR_LOG",
    str(Path.home() / ".cache" / "zotero-mcp" / "schema_errors.jsonl"),
))


# ---------- schema ----------
Role = Literal[
    "baseline", "prior-work", "related", "motivation",
    "contrast", "extends", "contradicts",
]
ContributionType = Literal["theory", "method", "system", "survey", "dataset", "tool"]


class MethodImproved(BaseModel):
    base: str
    aspect: str


class Reference(BaseModel):
    cited_title: str
    cited_author_year: str
    role: Role
    context_quote: str = Field(default="", max_length=240)


class ExtractedPaper(BaseModel):
    paper_id: str
    title: str
    tldr: str = Field(max_length=280)
    problem: str
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


SYSTEM_PROMPT = """你是学术文献结构化抽取助手。读以下论文 Markdown，严格按 JSON schema 输出。

规则：
- references 每条必须标 role，role ∈ {baseline, prior-work, related, motivation, contrast, extends, contradicts}
- methods_used vs methods_proposed 要区分：paper 自己提出的新方法 → proposed；复用他人的方法 → used
- concepts 控制在 5-10 个，宁少勿滥；不抽形容词、通用词（如 "novel"、"efficient"）
- context_quote 必须是原文 quote（≤240 字符），不允许改写
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


def _api_key() -> str:
    k = (os.getenv("HELPER_LLM_API_KEY")
         or os.getenv("KIMI_API_KEY")
         or os.getenv("MOONSHOT_API_KEY")
         or "")
    if not k:
        raise RuntimeError("HELPER_LLM_API_KEY / KIMI_API_KEY not set")
    return k


def _truncate(md: str, budget_tokens: int) -> str:
    # ≈ 3.5 chars/token for mixed zh/en scientific text
    max_chars = int(budget_tokens * 3.5)
    if len(md) <= max_chars:
        return md
    head = md[: max_chars - 500]
    return head + f"\n\n[...truncated {len(md) - max_chars} chars...]"


def _call_llm(user_prompt: str, *, max_tokens: int = 8000, timeout: int = 240) -> str:
    """Send a single prompt and return text response. Raises on HTTP error."""
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt},
    ]
    if HELPER_PROVIDER == "anthropic":
        payload = json.dumps({
            "model": HELPER_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": _api_key(),
            "anthropic-version": "2023-06-01",
        }
    else:  # openai-compatible
        payload = json.dumps({
            "model": HELPER_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        }
    req = urllib.request.Request(HELPER_URL, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if HELPER_PROVIDER == "anthropic":
        return data["content"][0]["text"].strip()
    return data["choices"][0]["message"]["content"].strip()


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

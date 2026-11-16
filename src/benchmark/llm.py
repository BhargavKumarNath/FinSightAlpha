"""Throttled, cached Groq chat client for the Phase 3 evaluation.

Why not ChatGroq here: the evaluation needs provider-reported token usage per
call, deterministic on-disk caching so interrupted runs resume without paying
twice, and a per-model tokens-per-minute limiter (the free tier allows 8,000
TPM per model). Every call is appended to a JSONL log with usage and latency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
URL = "https://api.groq.com/openai/v1/chat/completions"
TPM_LIMIT = {"openai/gpt-oss-120b": 8000, "openai/gpt-oss-20b": 8000, "qwen/qwen3.8-27b": 6000}
CACHE_DIR = Path("results/phase3/llm_cache")
LOG_PATH = Path("results/phase3/llm_calls.jsonl")


class DailyLimitError(RuntimeError):
    """Raised when the provider reports the daily token cap is exhausted."""


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    model: str


class _Limiter:
    """Sliding 60 s window of tokens spent, per model."""

    def __init__(self) -> None:
        self.events: dict[str, deque[tuple[float, int]]] = {}
        self.lock = threading.Lock()

    def wait(self, model: str, need: int) -> None:
        cap = int(TPM_LIMIT.get(model, 6000) * 0.9)
        while True:
            with self.lock:
                q = self.events.setdefault(model, deque())
                now = time.time()
                while q and now - q[0][0] > 60:
                    q.popleft()
                used = sum(t for _, t in q)
                if used + need <= cap or not q:
                    return
                sleep = 60 - (now - q[0][0]) + 0.2
            time.sleep(max(sleep, 0.5))

    def add(self, model: str, tokens: int) -> None:
        with self.lock:
            self.events.setdefault(model, deque()).append((time.time(), tokens))


_limiter = _Limiter()
_log_lock = threading.Lock()


def _key(model: str, messages: list[dict], params: dict) -> str:
    raw = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def chat(
    model: str,
    messages: list[dict],
    *,
    json_mode: bool = False,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    tag: str = "",
    use_cache: bool = True,
) -> LLMResult:
    """One chat completion. Cached by (model, messages, params)."""
    params: dict = {"max_tokens": max_tokens, "temperature": temperature, "json": json_mode,
                    "re": reasoning_effort}
    key = _key(model, messages, params)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        d = json.loads(path.read_text())
        return LLMResult(d["content"], d["prompt_tokens"], d["completion_tokens"],
                         d["latency_s"], True, model)

    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort and model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    est = sum(len(m["content"]) for m in messages) // 3 + max_tokens // 6

    for attempt in range(8):
        _limiter.wait(model, est)
        t0 = time.time()
        r = requests.post(URL, headers=headers, json=body, timeout=120)
        dt = time.time() - t0
        if r.status_code == 200:
            j = r.json()
            u = j["usage"]
            content = j["choices"][0]["message"].get("content") or ""
            total = u["prompt_tokens"] + u["completion_tokens"]
            _limiter.add(model, total)
            res = LLMResult(content, u["prompt_tokens"], u["completion_tokens"], dt, False, model)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"content": content, "prompt_tokens": u["prompt_tokens"],
                                        "completion_tokens": u["completion_tokens"],
                                        "latency_s": dt}))
            with _log_lock:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with LOG_PATH.open("a") as f:
                    f.write(json.dumps({"t": time.time(), "model": model, "tag": tag,
                                        "prompt_tokens": u["prompt_tokens"],
                                        "completion_tokens": u["completion_tokens"],
                                        "latency_s": round(dt, 3)}) + "\n")
            return res
        msg = r.text
        if r.status_code == 429:
            if re.search(r"per day|TPD|RPD", msg):
                raise DailyLimitError(msg[:400])
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 15 * (attempt + 1)
            time.sleep(min(wait, 90))
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"Groq {r.status_code}: {msg[:300]}")
    raise RuntimeError("Groq: retries exhausted")

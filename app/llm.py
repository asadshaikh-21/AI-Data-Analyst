"""Provider-agnostic LLM client (Anthropic / OpenAI over plain HTTP) with retries and caching."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import httpx

log = logging.getLogger("analyst.llm")


class LLMError(RuntimeError):
    pass


class LLM:
    name = "base"
    _cache: dict[str, str] = {}

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        last = None
        for attempt in range(5):
            try:
                r = httpx.post(url, headers=headers, json=payload, timeout=60)
                if r.status_code in (429, 500, 502, 503, 529):
                    last = LLMError(f"{self.name} API {r.status_code}")
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"{self.name} API error {r.status_code}: {r.text[:200]}")
                return r.json()
            except httpx.HTTPError as e:
                last = LLMError(f"{self.name} network error: {e}")
                time.sleep(1.5 * (attempt + 1))
        raise last or LLMError("LLM request failed")

    def _call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        raise NotImplementedError

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:
        key = hashlib.sha256(json.dumps([self.model, system, messages], sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        t = time.perf_counter()
        out = self._call(system, messages, max_tokens)
        log.info("llm=%s model=%s ms=%d", self.name, self.model, (time.perf_counter() - t) * 1000)
        if len(self._cache) > 256:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out


class AnthropicLLM(LLM):
    name = "anthropic"

    def _call(self, system, messages, max_tokens):
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages},
        )
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAILLM(LLM):
    name = "openai"

    def _call(self, system, messages, max_tokens):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        data = self._post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {"model": self.model, "max_tokens": max_tokens,
             "messages": [{"role": "system", "content": system}, *messages]},
        )
        return data["choices"][0]["message"]["content"]


def get_llm() -> LLM | None:
    """Return a configured LLM, or None to use the offline heuristic planner."""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "offline":
        return None
    a_key, o_key = os.getenv("ANTHROPIC_API_KEY"), os.getenv("OPENAI_API_KEY")
    if provider in ("anthropic", "auto") and a_key:
        return AnthropicLLM(a_key, os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    if provider in ("openai", "auto") and o_key:
        return OpenAILLM(o_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return None

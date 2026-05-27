"""Switchable LLM backend.

A single thin ``complete(system, user) -> str`` interface with three implementations:

* ``AnthropicClient`` / ``OpenAIClient`` — real calls via plain HTTP (no SDK dependency).
* ``MockLLMClient``                       — deterministic, offline; ``is_mock`` tells the
                                            encoder to use its keyword path instead.

``build_llm_client`` picks one based on config + available API keys, so the exact same
agent runs with a real model for the demo video and fully offline for a grader.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import requests

from research_radar.safety.guards import RateLimiter


class LLMError(RuntimeError):
    pass


class MockLLMClient:
    """No-network stand-in. The encoder branches on ``is_mock`` and never calls ``complete``."""

    is_mock = True
    name = "mock"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - not used
        raise LLMError("MockLLMClient.complete should not be called; encoder uses keyword path.")


class _HTTPClient:
    is_mock = False

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.timeout = cfg.get("timeout", 30)
        self.max_tokens = cfg.get("max_tokens", 400)
        self.temperature = cfg.get("temperature", 0.0)
        self._rl = RateLimiter(cfg.get("max_calls_per_min", 30), period=60.0)

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        self._rl.acquire()
        last_exc: Exception | None = None
        for attempt in range(2):  # one light retry on transient failure
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429 and attempt == 0:
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
        raise LLMError(f"LLM request failed: {last_exc}")


class AnthropicClient(_HTTPClient):
    def __init__(self, cfg: Dict[str, Any], api_key: str) -> None:
        super().__init__(cfg)
        self.api_key = api_key
        self.model = cfg.get("anthropic_model", "claude-haiku-4-5-20251001")
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        parts = data.get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()


class OpenAIClient(_HTTPClient):
    def __init__(self, cfg: Dict[str, Any], api_key: str) -> None:
        super().__init__(cfg)
        self.api_key = api_key
        self.model = cfg.get("openai_model", "gpt-4o-mini")
        self.name = f"openai:{self.model}"

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return data["choices"][0]["message"]["content"].strip()


def build_llm_client(cfg: Dict[str, Any]):
    """Select a backend from config + environment.

    provider=auto: use Anthropic if ANTHROPIC_API_KEY is set, else OpenAI if
    OPENAI_API_KEY is set, else fall back to the offline mock.
    """
    llm_cfg = cfg.get("llm", {})
    provider = (llm_cfg.get("provider") or "auto").lower()
    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    oai_key = os.environ.get("OPENAI_API_KEY")

    if provider == "mock":
        return MockLLMClient()
    if provider == "anthropic":
        if anth_key:
            return AnthropicClient(llm_cfg, anth_key)
        print("[llm] provider=anthropic but ANTHROPIC_API_KEY missing; falling back to mock.")
        return MockLLMClient()
    if provider == "openai":
        if oai_key:
            return OpenAIClient(llm_cfg, oai_key)
        print("[llm] provider=openai but OPENAI_API_KEY missing; falling back to mock.")
        return MockLLMClient()

    # auto
    if anth_key:
        return AnthropicClient(llm_cfg, anth_key)
    if oai_key:
        return OpenAIClient(llm_cfg, oai_key)
    return MockLLMClient()

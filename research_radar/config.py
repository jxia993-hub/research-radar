"""Configuration loading.

Defaults live in this module so the package works with zero setup. A `config.json`
at the repo root (or any path passed to :func:`load_config`) is deep-merged on top,
and a handful of operationally important values can be overridden by environment
variables so the demo can be steered without editing files.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

# The taxonomy is the *feature space* the RL agent learns over. Each topic maps to
# trigger keywords used by (a) the offline/mock encoder and (b) the simulated user.
# Keeping it here guarantees the package runs even if config.json is missing.
DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        "provider": "auto",  # auto | anthropic | openai | mock
        "anthropic_model": "claude-haiku-4-5-20251001",
        "openai_model": "gpt-4o-mini",
        "max_tokens": 400,
        "temperature": 0.0,
        "timeout": 30,
        "max_calls_per_min": 30,
    },
    "bandit": {"algo": "linucb", "alpha": 1.0, "lam": 1.0, "ts_v": 0.25},
    "perception": {
        "default_query": "reinforcement learning language model agent",
        "max_results": 12,
        "arxiv_sleep": 3.0,
    },
    "memory": {"path": ".radar_state/memory.json"},
    "safety": {"require_confirmation": True, "max_query_len": 300, "max_papers_per_cycle": 50},
    "reward_map": {"save": 1.0, "like": 0.8, "read": 0.6, "skip": 0.1, "dislike": 0.0},
    "taxonomy": {
        "reinforcement_learning": ["reinforcement learning", "policy gradient", "q-learning", "actor-critic", "ppo", "contextual bandit", "markov decision", "reward shaping", "exploration", "off-policy"],
        "large_language_models": ["large language model", " llm", "transformer", "pretraining", "language model", "in-context learning", "scaling law", "foundation model", "next-token"],
        "llm_agents": ["agent", "tool use", "tool-use", "planning", "react", "autonomous", "multi-agent", "function calling", "agentic", "orchestration"],
        "rlhf_alignment": ["rlhf", "human feedback", "preference optimization", "alignment", "dpo", "reward model", "constitutional", "safety", "guardrail", "red-team"],
        "multimodal_vlm": ["multimodal", "vision-language", "vlm", "visual", "video understanding", "image-text", "captioning", "cross-modal"],
        "reasoning": ["reasoning", "chain-of-thought", "chain of thought", "step-by-step", "mathematical reasoning", "logical", "deduction", "self-consistency", "verifier"],
        "efficiency": ["efficient", "quantization", "distillation", "pruning", "inference acceleration", "kv cache", "lora", "parameter-efficient", "sparse", "speedup", "low-rank"],
        "theory": ["convergence", "generalization bound", "regret bound", "sample complexity", "provable", "optimization theory", "pac", "analysis"],
        "robotics_control": ["robot", "manipulation", "locomotion", "control policy", "embodied", "sim-to-real", "dexterous", "navigation"],
        "retrieval_rag": ["retrieval", "retrieval-augmented", " rag", "knowledge base", "vector database", "dense retrieval", "long-term memory"],
        "generative_models": ["diffusion", "generative model", "text-to-image", "flow matching", "score-based", "image generation", "variational autoencoder"],
        "evaluation_benchmarks": ["benchmark", "evaluation", "leaderboard", "new dataset", " metric", "assessment", "probing"],
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Allow a few env vars to steer the demo without editing config.json."""
    provider = os.environ.get("RADAR_LLM_PROVIDER")
    if provider:
        cfg["llm"]["provider"] = provider
    model = os.environ.get("RADAR_LLM_MODEL")
    if model:
        # Set whichever model field matches the active provider; harmless otherwise.
        cfg["llm"]["anthropic_model"] = model
        cfg["llm"]["openai_model"] = model
    algo = os.environ.get("RADAR_BANDIT_ALGO")
    if algo:
        cfg["bandit"]["algo"] = algo
    mem = os.environ.get("RADAR_MEMORY_PATH")
    if mem:
        cfg["memory"]["path"] = mem
    return cfg


def load_config(path: str | os.PathLike | None = "config.json") -> Dict[str, Any]:
    """Return the merged configuration dict.

    Order of precedence (low -> high): DEFAULT_CONFIG < config.json < env vars.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if p.exists():
            try:
                cfg = _deep_merge(cfg, json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
                print(f"[config] warning: could not read {p} ({exc}); using defaults")
    return _apply_env_overrides(cfg)


def topic_names(cfg: Dict[str, Any]) -> list[str]:
    """Stable, ordered list of taxonomy topics — defines the feature dimension order."""
    return list(cfg["taxonomy"].keys())

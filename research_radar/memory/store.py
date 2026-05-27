"""Durable memory.

A single JSON file holds everything the agent should remember between runs:

* ``profile``  — the user's stated interests (free text).
* ``bandit``   — the serialised contextual-bandit state (the learned preferences).
* ``history``  — every (paper, action, reward) interaction.
* ``seen``     — arXiv ids already shown, so the queue keeps moving.

Persisting the bandit state is what makes the agent *improve over sessions* rather than
restarting from scratch each time — the "learning" half of the perceive→…→learn loop.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_radar.decision.bandit import BanditPolicy, make_bandit


class MemoryStore:
    def __init__(self, path: str, topics: List[str]) -> None:
        self.path = Path(path)
        self.topics = topics
        self.data: Dict[str, Any] = {
            "profile": {"interests": "", "created_at": _now()},
            "bandit": None,  # {"algo": ..., "state": {...}}
            "history": [],
            "seen": [],
            "pending": {},  # arxiv_id -> {title, vector, summary, source, top_topics}
        }
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
                print(f"[memory] could not read {self.path} ({exc}); starting fresh.")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic-ish write

    # ------------------------------------------------------------------ profile
    @property
    def interests(self) -> str:
        return self.data["profile"].get("interests", "")

    def set_interests(self, interests: str) -> None:
        self.data["profile"]["interests"] = interests

    # ------------------------------------------------------------------ bandit
    def get_or_init_bandit(self, algo: str, d: int, bandit_cfg: Dict, seed: Optional[int] = None) -> BanditPolicy:
        """Reconstruct the saved policy if its algo/dimension match, else create fresh."""
        bandit = make_bandit(algo, d, bandit_cfg, seed=seed)
        saved = self.data.get("bandit")
        if saved and saved.get("algo") == algo and saved.get("state", {}).get("d") == d:
            try:
                bandit.load_state(saved["state"])
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[memory] could not restore bandit ({exc}); starting fresh.")
        return bandit

    def save_bandit(self, algo: str, bandit: BanditPolicy) -> None:
        self.data["bandit"] = {"algo": algo, "state": bandit.state()}

    # ------------------------------------------------------------------ history / seen
    def is_seen(self, arxiv_id: str) -> bool:
        return arxiv_id in set(self.data.get("seen", []))

    def mark_seen(self, arxiv_id: str) -> None:
        if arxiv_id not in self.data["seen"]:
            self.data["seen"].append(arxiv_id)

    def record_interaction(self, *, arxiv_id: str, title: str, action: str, reward: float,
                           source: str, top_topics: List[str]) -> None:
        self.data["history"].append({
            "ts": _now(),
            "arxiv_id": arxiv_id,
            "title": title,
            "action": action,
            "reward": reward,
            "source": source,
            "top_topics": top_topics,
        })
        self.mark_seen(arxiv_id)

    # ------------------------------------------------------------------ pending
    # Recommendations shown to the user but not yet acted on. Stored so that a later,
    # separate CLI invocation (`radar feedback <id> save`) can recover the feature
    # vector and update the bandit — the agent "remembers what it showed you".
    def set_pending(self, items: Dict[str, Dict[str, Any]]) -> None:
        self.data["pending"] = items

    def get_pending(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get("pending", {}).get(arxiv_id)

    def clear_pending(self, arxiv_id: str) -> None:
        self.data.get("pending", {}).pop(arxiv_id, None)

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, Any]:
        hist = self.data.get("history", [])
        by_action: Dict[str, int] = {}
        total_reward = 0.0
        for h in hist:
            by_action[h["action"]] = by_action.get(h["action"], 0) + 1
            total_reward += float(h.get("reward", 0.0))
        return {
            "interactions": len(hist),
            "by_action": by_action,
            "avg_reward": (total_reward / len(hist)) if hist else 0.0,
            "seen": len(self.data.get("seen", [])),
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

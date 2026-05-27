"""Contextual-bandit policies — the RL component of the agent.

Each paper is an arm described by a feature vector ``x`` (produced by the LLM encoder).
The agent learns a linear value function ``r ≈ θ·x`` from user feedback and uses it to
rank papers, balancing **exploitation** (high predicted value) against **exploration**
(high uncertainty). This is the contextual-bandit setting from Sutton & Barto ch.2 and
Li et al. 2010 ("A contextual-bandit approach to personalized news recommendation") —
swap "news" for "papers" and that is exactly this agent.

Policies:
    LinUCB         — ridge regression + UCB exploration bonus (our default).
    LinTS          — Bayesian linear model + Thompson sampling.
    EpsilonGreedy  — greedy value + ε random exploration.
    RandomPolicy   — pick uniformly (lower-bound baseline).
    StaticLLMPolicy— rank by a fixed weight vector, never learns ("pure LLM" baseline).

All linear policies expose the same ``evaluate / select / update`` API and can be
serialised into the memory store so learning persists across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Evaluation:
    """Per-candidate breakdown so the UI can explain *why* a paper was ranked here."""

    score: np.ndarray   # value used for ranking
    mean: np.ndarray    # exploitation term  (θ·x)
    bonus: np.ndarray   # exploration term   (uncertainty)


class _LinearModel:
    """Shared ridge-regression state: A = λI + ΣxxᵀΣ, b = Σ r·x, θ = A⁻¹b."""

    def __init__(self, d: int, lam: float = 1.0) -> None:
        self.d = int(d)
        self.lam = float(lam)
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.n_updates = 0

    @property
    def A_inv(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.A += np.outer(x, x)
        self.b += float(reward) * x
        self.n_updates += 1

    # --- (de)serialisation for the memory store ---
    def state(self) -> dict:
        return {"d": self.d, "lam": self.lam, "A": self.A.tolist(), "b": self.b.tolist(), "n": self.n_updates}

    def load_state(self, s: dict) -> None:
        self.d = int(s["d"]); self.lam = float(s["lam"])
        self.A = np.array(s["A"], dtype=float)
        self.b = np.array(s["b"], dtype=float)
        self.n_updates = int(s.get("n", 0))


class BanditPolicy(_LinearModel):
    name = "linear"
    learns = True

    def evaluate(self, X: np.ndarray) -> Evaluation:  # pragma: no cover - overridden
        raise NotImplementedError

    def select(self, X: np.ndarray, k: int = 1) -> List[int]:
        """Return indices of the top-k arms by score (descending)."""
        scores = self.evaluate(np.asarray(X, dtype=float)).score
        order = np.argsort(-scores)
        return order[: max(1, k)].tolist()


class LinUCB(BanditPolicy):
    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0) -> None:
        super().__init__(d, lam)
        self.alpha = float(alpha)
        self.name = f"linucb(α={alpha:g})" if alpha > 0 else "greedy"

    def evaluate(self, X: np.ndarray) -> Evaluation:
        X = np.asarray(X, dtype=float)
        A_inv = self.A_inv
        mean = X @ self.theta()
        # bonus_i = sqrt(x_iᵀ A⁻¹ x_i) — large when this feature direction is under-explored.
        bonus = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, A_inv, X), 0.0))
        return Evaluation(score=mean + self.alpha * bonus, mean=mean, bonus=self.alpha * bonus)


class LinTS(BanditPolicy):
    """Linear Thompson sampling: sample θ̃ ~ N(θ̂, v²A⁻¹) and rank by θ̃·x."""

    def __init__(self, d: int, v: float = 0.25, lam: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(d, lam)
        self.v = float(v)
        self.rng = np.random.default_rng(seed)
        self.name = f"lints(v={v:g})"

    def evaluate(self, X: np.ndarray) -> Evaluation:
        X = np.asarray(X, dtype=float)
        A_inv = self.A_inv
        theta_hat = A_inv @ self.b
        cov = self.v ** 2 * A_inv
        cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(self.d)  # symmetrise for the sampler
        theta_tilde = self.rng.multivariate_normal(theta_hat, cov)
        mean = X @ theta_hat
        score = X @ theta_tilde
        return Evaluation(score=score, mean=mean, bonus=score - mean)


class EpsilonGreedy(BanditPolicy):
    def __init__(self, d: int, epsilon: float = 0.1, lam: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(d, lam)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.name = f"ε-greedy(ε={epsilon:g})"

    def evaluate(self, X: np.ndarray) -> Evaluation:
        X = np.asarray(X, dtype=float)
        mean = X @ self.theta()
        return Evaluation(score=mean.copy(), mean=mean, bonus=np.zeros(len(X)))

    def select(self, X: np.ndarray, k: int = 1) -> List[int]:
        X = np.asarray(X, dtype=float)
        n = len(X)
        if self.rng.random() < self.epsilon:  # explore: random ordering
            return self.rng.permutation(n)[: max(1, k)].tolist()
        return super().select(X, k)


class RandomPolicy(BanditPolicy):
    learns = False

    def __init__(self, d: int, seed: Optional[int] = None) -> None:
        super().__init__(d, lam=1.0)
        self.rng = np.random.default_rng(seed)
        self.name = "random"

    def evaluate(self, X: np.ndarray) -> Evaluation:
        n = len(np.asarray(X))
        s = self.rng.random(n)
        return Evaluation(score=s, mean=np.zeros(n), bonus=s)

    def update(self, x: np.ndarray, reward: float) -> None:
        return  # learns nothing


class StaticLLMPolicy(BanditPolicy):
    """Ranks by a fixed weight vector (e.g. the LLM's zero-shot relevance guess) and
    never updates. Stands in for "just trust the LLM score" — the baseline RL must beat."""

    learns = False

    def __init__(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        super().__init__(len(w), lam=1.0)
        self.b = w.copy()
        self.A = np.eye(len(w))  # so theta() == weights
        self.name = "static-llm"

    def evaluate(self, X: np.ndarray) -> Evaluation:
        X = np.asarray(X, dtype=float)
        mean = X @ self.b
        return Evaluation(score=mean.copy(), mean=mean, bonus=np.zeros(len(X)))

    def update(self, x: np.ndarray, reward: float) -> None:
        return  # static by design


# Convenience alias used by the memory store for typing/serialisation.
BanditState = dict


def make_bandit(algo: str, d: int, bandit_cfg: Dict, seed: Optional[int] = None) -> BanditPolicy:
    """Factory for the agent-facing policies (the ones that learn online)."""
    algo = (algo or "linucb").lower()
    lam = float(bandit_cfg.get("lam", 1.0))
    if algo == "linucb":
        return LinUCB(d, alpha=float(bandit_cfg.get("alpha", 1.0)), lam=lam)
    if algo in ("lints", "thompson", "ts"):
        return LinTS(d, v=float(bandit_cfg.get("ts_v", 0.25)), lam=lam, seed=seed)
    if algo in ("egreedy", "epsilon", "epsilon_greedy"):
        return EpsilonGreedy(d, epsilon=float(bandit_cfg.get("epsilon", 0.1)), lam=lam, seed=seed)
    if algo == "greedy":
        return LinUCB(d, alpha=0.0, lam=lam)
    if algo == "random":
        return RandomPolicy(d, seed=seed)
    raise ValueError(f"unknown bandit algo: {algo!r}")

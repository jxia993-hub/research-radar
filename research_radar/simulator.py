"""Simulated user + episode runner for evaluating the RL agent.

Real human feedback is too slow to show learning in a 2-minute demo, so we model a user
with a *hidden* preference vector over topics. The reward for a paper is the (clipped,
noisy) inner product of those preferences with the paper's topic features. A good agent
should recover the hidden preferences from feedback and steer recommendations toward
high-reward papers — visible as a rising reward / falling regret curve.

This is a faithful, if simplified, contextual-bandit environment: features come from the
same encoder taxonomy the live agent uses, and ``StaticLLMPolicy`` models "just trust the
LLM's zero-shot guess" so we can show what online learning adds on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from research_radar.decision.bandit import BanditPolicy


def make_synthetic_pool(n: int, n_topics: int, rng: np.random.Generator,
                        active_min: int = 1, active_max: int = 3) -> np.ndarray:
    """Generate a pool of paper feature vectors that mimics the encoder's output:
    each paper is 'about' 1-3 topics (values in [0.4, 1.0]); a bias term (=1) is appended."""
    pool = np.zeros((n, n_topics + 1))
    for i in range(n):
        k = rng.integers(active_min, active_max + 1)
        topics = rng.choice(n_topics, size=k, replace=False)
        pool[i, topics] = rng.uniform(0.4, 1.0, size=k)
    pool[:, -1] = 1.0  # bias feature
    return pool


@dataclass
class SimulatedUser:
    """Hidden linear preferences over topics (the bias feature is ignored by the user)."""

    weights: np.ndarray            # length = n_topics
    noise: float = 0.05
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @property
    def n_topics(self) -> int:
        return len(self.weights)

    def expected_reward(self, x: np.ndarray) -> float:
        return float(np.clip(np.dot(self.weights, x[: self.n_topics]), 0.0, 1.0))

    def expected_reward_batch(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X[:, : self.n_topics] @ self.weights, 0.0, 1.0)

    def sample_reward(self, x: np.ndarray) -> float:
        return float(np.clip(self.expected_reward(x) + self.rng.normal(0.0, self.noise), 0.0, 1.0))


def make_user(n_topics: int, rng: np.random.Generator, n_pref: int = 4,
              pref_weight: float = 0.6, other_weight: float = 0.05, noise: float = 0.05) -> SimulatedUser:
    """A user who strongly prefers ``n_pref`` random topics and is mildly interested in the rest."""
    weights = np.full(n_topics, other_weight)
    pref = rng.choice(n_topics, size=n_pref, replace=False)
    weights[pref] = pref_weight
    return SimulatedUser(weights=weights, noise=noise, rng=rng)


def run_episode(bandit: BanditPolicy, pool: np.ndarray, user: SimulatedUser,
                rounds: int, candidates_per_round: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Run one bandit episode. Returns (expected_reward_of_choice, instant_regret) per round."""
    n = len(pool)
    rewards = np.zeros(rounds)
    regrets = np.zeros(rounds)
    for t in range(rounds):
        idx = rng.choice(n, size=min(candidates_per_round, n), replace=False)
        X = pool[idx]
        chosen = bandit.select(X, k=1)[0]
        x = X[chosen]
        exp_pool = user.expected_reward_batch(X)
        chosen_reward = user.expected_reward(x)
        bandit.update(x, user.sample_reward(x))  # the agent only ever sees the noisy reward
        rewards[t] = chosen_reward
        regrets[t] = float(exp_pool.max() - chosen_reward)
    return rewards, regrets

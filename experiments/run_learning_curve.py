"""Evaluate the RL component: do the contextual bandits actually learn the user's
preferences, and do they beat "just trust the LLM's zero-shot score"?

We run every policy in the simulated-user environment for many rounds across several
seeds, holding the paper pool, candidate draws and reward noise identical across policies
so the *only* difference is the decision rule. Outputs:

  * results/learning_curve.png  — avg reward (smoothed) and cumulative regret, mean ± std
  * a printed summary table

Run from the repo root:   python experiments/run_learning_curve.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `research_radar` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research_radar.config import load_config, topic_names  # noqa: E402
from research_radar.decision.bandit import (  # noqa: E402
    EpsilonGreedy, LinTS, LinUCB, RandomPolicy, StaticLLMPolicy,
)
from research_radar.simulator import make_synthetic_pool, run_episode, SimulatedUser  # noqa: E402

# Plot order / colours kept stable so the figure reads the same every run.
METHODS = ["random", "static-llm", "ε-greedy", "linucb", "lints"]


def _build(method: str, d: int, bcfg: dict, guess: np.ndarray, seed: int):
    if method == "random":
        return RandomPolicy(d, seed=seed)
    if method == "static-llm":
        return StaticLLMPolicy(guess)
    if method == "ε-greedy":
        return EpsilonGreedy(d, epsilon=0.1, lam=bcfg["lam"], seed=seed)
    if method == "linucb":
        return LinUCB(d, alpha=bcfg["alpha"], lam=bcfg["lam"])
    if method == "lints":
        return LinTS(d, v=bcfg["ts_v"], lam=bcfg["lam"], seed=seed)
    raise ValueError(method)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")  # edge-pad to avoid boundary dips
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def run_experiment(cfg, rounds, seeds, pool_size, candidates, sigma_guess, n_pref):
    topics = topic_names(cfg)
    n_topics = len(topics)
    d = n_topics + 1
    bcfg = cfg["bandit"]

    rewards = {m: np.zeros((seeds, rounds)) for m in METHODS}
    regrets = {m: np.zeros((seeds, rounds)) for m in METHODS}

    for s in range(seeds):
        base = np.random.default_rng(1000 + s)
        pool = make_synthetic_pool(pool_size, n_topics, base)
        # Hidden user preferences for this seed.
        weights = np.full(n_topics, 0.05)
        weights[base.choice(n_topics, size=n_pref, replace=False)] = 0.6
        # The LLM's zero-shot guess = true preferences corrupted by noise (length d, bias=0).
        guess = np.concatenate([np.clip(weights + base.normal(0, sigma_guess, n_topics), 0, None), [0.0]])

        for m in METHODS:
            # Identical environment per method: same candidate draws and same reward noise.
            user = SimulatedUser(weights=weights, noise=0.05, rng=np.random.default_rng(2000 + s))
            ep_rng = np.random.default_rng(3000 + s)
            bandit = _build(m, d, bcfg, guess, seed=4000 + s)
            r, g = run_episode(bandit, pool, user, rounds, candidates, ep_rng)
            rewards[m][s] = r
            regrets[m][s] = g

    return topics, rewards, regrets


def plot(rewards, regrets, rounds, out_path: Path) -> None:
    win = max(1, rounds // 20)
    x = np.arange(rounds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for m in METHODS:
        n_seeds = rewards[m].shape[0]
        mean_r = _smooth(rewards[m].mean(axis=0), win)
        se_r = _smooth(rewards[m].std(axis=0) / np.sqrt(n_seeds), win)  # standard error of the mean
        ax1.plot(x, mean_r, label=m, linewidth=2)
        ax1.fill_between(x, mean_r - se_r, mean_r + se_r, alpha=0.15)

        cum = np.cumsum(regrets[m], axis=1)
        mean_c, se_c = cum.mean(axis=0), cum.std(axis=0) / np.sqrt(n_seeds)
        ax2.plot(x, mean_c, label=m, linewidth=2)
        ax2.fill_between(x, mean_c - se_c, mean_c + se_c, alpha=0.15)

    ax1.set_title("Average reward per round (higher = better)")
    ax1.set_xlabel("round"); ax1.set_ylabel("expected reward of chosen paper")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    ax2.set_title("Cumulative regret (lower = better)")
    ax2.set_xlabel("round"); ax2.set_ylabel("cumulative regret")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

    fig.suptitle("Research Radar — LLM features + contextual bandit vs. baselines", fontsize=13)
    n_seeds = rewards[METHODS[0]].shape[0]
    fig.text(0.5, 0.005, f"Mean over {n_seeds} seeds; shaded = ±1 standard error. "
                         "Identical environment per seed across all policies.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved figure -> {out_path}")


def summarise(rewards, regrets, rounds) -> None:
    tail = max(1, rounds // 5)
    print("\n" + "=" * 64)
    print(f"{'method':<14}{'final reward':>16}{'total regret':>18}")
    print("-" * 64)
    for m in METHODS:
        final_reward = rewards[m][:, -tail:].mean()
        total_regret = np.cumsum(regrets[m], axis=1)[:, -1].mean()
        print(f"{m:<14}{final_reward:>16.3f}{total_regret:>18.1f}")
    print("=" * 64)
    print(f"(final reward = mean over last {tail} rounds, averaged across seeds)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RL learning-curve experiment")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--sigma-guess", type=float, default=0.25,
                    help="noise on the static-LLM baseline's zero-shot preference guess")
    ap.add_argument("--n-pref", type=int, default=4, help="number of topics the user prefers")
    ap.add_argument("--out", default="results/learning_curve.png")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"Running {args.seeds} seeds x {args.rounds} rounds, pool={args.pool}, "
          f"candidates/round={args.candidates} ...")
    _, rewards, regrets = run_experiment(
        cfg, args.rounds, args.seeds, args.pool, args.candidates, args.sigma_guess, args.n_pref
    )
    plot(rewards, regrets, args.rounds, Path(args.out))
    summarise(rewards, regrets, args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

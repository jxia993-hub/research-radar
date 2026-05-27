"""Sanity tests — run with:  python -m unittest discover -s tests   (or: pytest)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_radar.agent import ResearchRadarAgent
from research_radar.config import load_config
from research_radar.decision.bandit import LinUCB, RandomPolicy
from research_radar.perception.arxiv_source import Paper
from research_radar.reasoning.encoder import PaperEncoder
from research_radar.reasoning.llm_client import MockLLMClient
from research_radar.simulator import SimulatedUser, make_synthetic_pool, run_episode


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class TestBanditLearns(unittest.TestCase):
    def _env(self, seed=0, n_topics=8, n_pref=3):
        rng = np.random.default_rng(seed)
        pool = make_synthetic_pool(200, n_topics, rng)
        weights = np.full(n_topics, 0.05)
        weights[rng.choice(n_topics, size=n_pref, replace=False)] = 0.7
        user = SimulatedUser(weights=weights, noise=0.03, rng=np.random.default_rng(seed + 1))
        return pool, weights, user, n_topics

    def test_linucb_improves_and_recovers_preferences(self):
        pool, weights, user, n_topics = self._env()
        bandit = LinUCB(n_topics + 1, alpha=1.0, lam=1.0)
        rewards, _ = run_episode(bandit, pool, user, rounds=400, candidates_per_round=10,
                                 rng=np.random.default_rng(7))
        # Compare the pre-learning window (cold start) to the converged window.
        early, late = rewards[:20].mean(), rewards[-100:].mean()
        self.assertGreater(late, early + 0.05, "reward should rise as the bandit learns")
        # Learned θ over topics should align with the hidden preference vector.
        self.assertGreater(cosine(bandit.theta()[:n_topics], weights), 0.6)

    def test_linucb_beats_random(self):
        pool, weights, user, n_topics = self._env(seed=3)
        common = dict(rounds=300, candidates_per_round=10)
        lin = LinUCB(n_topics + 1, alpha=1.0, lam=1.0)
        r_lin, _ = run_episode(lin, pool, user, rng=np.random.default_rng(11), **common)
        user2 = SimulatedUser(weights=weights, noise=0.03, rng=np.random.default_rng(4))
        rnd = RandomPolicy(n_topics + 1, seed=5)
        r_rnd, _ = run_episode(rnd, pool, user2, rng=np.random.default_rng(11), **common)
        self.assertGreater(r_lin.mean(), r_rnd.mean() + 0.05)


class TestEncoder(unittest.TestCase):
    def _cfg(self):
        return load_config(str(Path(__file__).resolve().parents[1] / "config.json"))

    def test_mock_vector_shape_and_range(self):
        cfg = self._cfg()
        enc = PaperEncoder(MockLLMClient(), cfg)
        paper = Paper(arxiv_id="0000.0001",
                      title="A contextual bandit for reinforcement learning agents",
                      abstract="We study policy gradient and exploration in reinforcement learning.")
        feats = enc.encode(paper, interests="reinforcement learning")
        n_topics = len(cfg["taxonomy"])
        vec = feats.vector(list(cfg["taxonomy"].keys()))
        self.assertEqual(len(vec), n_topics + 1)
        self.assertEqual(vec[-1], 1.0)  # bias term
        self.assertTrue(all(0.0 <= s <= 1.0 for s in feats.topic_scores.values()))
        self.assertGreater(feats.topic_scores["reinforcement_learning"], 0.0)


class TestMemoryRoundTrip(unittest.TestCase):
    def test_bandit_state_persists(self):
        b1 = LinUCB(6, alpha=1.0, lam=1.0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            b1.update(rng.random(6), float(rng.random()))
        b2 = LinUCB(6, alpha=1.0, lam=1.0)
        b2.load_state(b1.state())
        np.testing.assert_allclose(b1.theta(), b2.theta(), rtol=1e-9)


class TestAgentOffline(unittest.TestCase):
    def _agent(self, tmp):
        cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.json"))
        cfg["llm"]["provider"] = "mock"
        cfg["memory"]["path"] = str(Path(tmp) / "mem.json")
        return ResearchRadarAgent(cfg, offline=True, seed=0)

    def test_reason_decide_learn(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            papers = [
                Paper(arxiv_id="1.1", title="RLHF and preference optimization",
                      abstract="Reinforcement learning from human feedback and DPO for alignment."),
                Paper(arxiv_id="1.2", title="Diffusion image generation",
                      abstract="A new text-to-image diffusion model."),
            ]
            feats = agent.reason(papers)
            recs = agent.decide(papers, feats, k=2)
            self.assertEqual(len(recs), 2)
            # Mimic run_cycle's pending bookkeeping so feedback can update the policy.
            agent.memory.set_pending({
                r.paper.arxiv_id: {"title": r.paper.title,
                                   "vector": r.features.vector(agent.topics).tolist(),
                                   "summary": r.features.summary, "source": r.features.source,
                                   "top_topics": r.top_topics()} for r in recs})
            reward = agent.learn("1.1", "save")
            self.assertEqual(reward, 1.0)
            self.assertEqual(agent.bandit.n_updates, 1)
            self.assertEqual(agent.stats()["interactions"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""ResearchRadarAgent — orchestrates the full agent loop.

    perceive(query)        fetch candidate papers from arXiv          [perception]
        -> reason(papers)  encode each into a topic feature vector    [reasoning / LLM]
        -> decide(...)     rank with the contextual bandit            [decision / RL]
        -> act(...)        present the queue + provenance             [action]
        -> learn(id,fb)    map feedback to reward, update the policy  [decision + memory]

Memory persists the policy and history between runs; safety guards wrap every external
call and every output. The same object backs both the interactive CLI and the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from research_radar.config import topic_names
from research_radar.decision.bandit import BanditPolicy
from research_radar.memory.store import MemoryStore
from research_radar.perception.arxiv_source import Paper, fetch_papers
from research_radar.reasoning.encoder import PaperEncoder, PaperFeatures, _first_sentences
from research_radar.reasoning.llm_client import build_llm_client
from research_radar.safety.guards import grounding_check, sanitize_query


@dataclass
class Recommendation:
    rank: int
    paper: Paper
    features: PaperFeatures
    score: float
    exploit: float   # θ·x   — predicted value
    explore: float   # bonus — exploration / uncertainty term
    warnings: List[str]

    def top_topics(self, k: int = 3, threshold: float = 0.3) -> List[str]:
        ranked = sorted(self.features.topic_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, s in ranked[:k] if s >= threshold]


class ResearchRadarAgent:
    def __init__(self, cfg: Dict, *, offline: bool = False, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.offline = offline
        self.topics = topic_names(cfg)
        self.d = len(self.topics) + 1  # +1 for bias term
        self.client = build_llm_client(cfg)
        self.encoder = PaperEncoder(self.client, cfg)
        self.memory = MemoryStore(cfg["memory"]["path"], self.topics)
        self.algo = cfg["bandit"]["algo"]
        self.bandit: BanditPolicy = self.memory.get_or_init_bandit(self.algo, self.d, cfg["bandit"], seed)
        self.reward_map: Dict[str, float] = cfg["reward_map"]
        self.safety = cfg["safety"]

    @property
    def interests(self) -> str:
        return self.memory.interests

    def set_interests(self, interests: str) -> None:
        self.memory.set_interests(interests)
        self.memory.save()

    # ----------------------------------------------------------------- perceive
    def perceive(self, query: str, max_results: int, *, skip_seen: bool = True) -> List[Paper]:
        query = sanitize_query(query, self.safety.get("max_query_len", 300))
        max_results = min(max_results, self.safety.get("max_papers_per_cycle", 50))
        papers = fetch_papers(
            query,
            max_results=max_results,
            offline=self.offline,
            arxiv_sleep=self.cfg["perception"].get("arxiv_sleep", 0.0) if not self.offline else 0.0,
        )
        if skip_seen:
            papers = [p for p in papers if not self.memory.is_seen(p.arxiv_id)]
        return papers

    # ----------------------------------------------------------------- reason
    def reason(self, papers: List[Paper]) -> List[PaperFeatures]:
        feats = self.encoder.encode_many(papers, self.interests)
        # Safety: keep summaries grounded in the source abstract.
        for paper, f in zip(papers, feats):
            ok, warns = grounding_check(f.summary, paper.abstract)
            if not ok:
                f.summary = _first_sentences(paper.abstract)  # fall back to verbatim text
                f.source = f.source + "+grounded"
        return feats

    # ----------------------------------------------------------------- decide
    def decide(self, papers: List[Paper], feats: List[PaperFeatures], k: int) -> List[Recommendation]:
        if not papers:
            return []
        X = np.vstack([f.vector(self.topics) for f in feats])
        ev = self.bandit.evaluate(X)
        order = np.argsort(-ev.score)
        recs: List[Recommendation] = []
        for rank, i in enumerate(order[:k], start=1):
            paper, f = papers[i], feats[i]
            _, warns = grounding_check(f.summary, paper.abstract)
            recs.append(Recommendation(
                rank=rank, paper=paper, features=f,
                score=float(ev.score[i]), exploit=float(ev.mean[i]), explore=float(ev.bonus[i]),
                warnings=warns,
            ))
        return recs

    # ----------------------------------------------------------------- act
    def run_cycle(self, query: str, k: int = 5, max_results: Optional[int] = None) -> List[Recommendation]:
        """perceive -> reason -> decide, and remember what was shown for later feedback."""
        max_results = max_results or self.cfg["perception"]["max_results"]
        papers = self.perceive(query, max_results)
        feats = self.reason(papers)
        recs = self.decide(papers, feats, k)
        pending = {
            r.paper.arxiv_id: {
                "title": r.paper.title,
                "vector": r.features.vector(self.topics).tolist(),
                "summary": r.features.summary,
                "source": r.features.source,
                "top_topics": r.top_topics(),
            }
            for r in recs
        }
        self.memory.set_pending(pending)
        self.memory.save()
        return recs

    # ----------------------------------------------------------------- learn
    def learn(self, arxiv_id: str, action: str) -> float:
        """Map feedback to a reward and update the policy. Returns the reward applied."""
        if action not in self.reward_map:
            raise ValueError(f"unknown action {action!r}; choose from {list(self.reward_map)}")
        pending = self.memory.get_pending(arxiv_id)
        if pending is None:
            raise KeyError(f"no pending recommendation for {arxiv_id!r}; run a recommendation first")
        reward = float(self.reward_map[action])
        x = np.array(pending["vector"], dtype=float)
        self.bandit.update(x, reward)
        self.memory.record_interaction(
            arxiv_id=arxiv_id, title=pending["title"], action=action, reward=reward,
            source=pending.get("source", ""), top_topics=pending.get("top_topics", []),
        )
        self.memory.save_bandit(self.algo, self.bandit)
        self.memory.clear_pending(arxiv_id)
        self.memory.save()
        return reward

    # ----------------------------------------------------------------- introspection
    def topic_weights(self) -> List[tuple[str, float]]:
        """Learned preference per topic (θ without the bias term), high to low."""
        theta = self.bandit.theta()
        pairs = list(zip(self.topics, theta[: len(self.topics)]))
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)

    def stats(self) -> Dict:
        s = self.memory.stats()
        s["bandit"] = self.bandit.name
        s["llm_backend"] = self.client.name
        s["bias"] = float(self.bandit.theta()[-1])
        return s

"""LLM-as-state-encoder.

This is the bridge between the LLM and the RL agent. Each paper is compressed into:

* ``topic_scores`` — a vector over the taxonomy. This *is* the contextual feature
  vector ``x`` the bandit learns a value function over.
* ``summary``      — a short, grounded blurb shown to the user.
* ``novelty`` / ``relevance_hint`` — scalar signals (the latter seeds the static-LLM baseline).

With a real backend we prompt the LLM for strict JSON; with the mock backend (or if the
LLM output can't be parsed) we fall back to deterministic keyword scoring, so the feature
space is identical in both modes and the RL results are reproducible offline.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from research_radar.perception.arxiv_source import Paper


@dataclass
class PaperFeatures:
    arxiv_id: str
    topic_scores: Dict[str, float]
    summary: str = ""
    novelty: float = 0.5
    relevance_hint: float = 0.5
    source: str = "mock"  # llm | mock | fallback
    _topics: List[str] = field(default_factory=list, repr=False)

    def vector(self, topics: Optional[List[str]] = None) -> np.ndarray:
        """Dense feature vector in fixed topic order, with a trailing bias term (=1.0)."""
        topics = topics or self._topics or list(self.topic_scores.keys())
        x = np.array([float(self.topic_scores.get(t, 0.0)) for t in topics], dtype=float)
        return np.concatenate([x, [1.0]])  # bias allows the model to learn a baseline


def _first_sentences(text: str, n: int = 2, limit: int = 240) -> str:
    """Extractive fallback summary — guaranteed grounded because it's copied verbatim."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:n]).strip()
    return (out[:limit] + "…") if len(out) > limit else out


class PaperEncoder:
    def __init__(self, client, cfg: Dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.taxonomy: Dict[str, List[str]] = cfg["taxonomy"]
        self.topics: List[str] = list(self.taxonomy.keys())

    # ------------------------------------------------------------------ keyword path
    def _keyword_scores(self, paper: Paper) -> Dict[str, float]:
        blob = " " + paper.text.lower() + " "
        scores: Dict[str, float] = {}
        for topic, keywords in self.taxonomy.items():
            hits = sum(blob.count(kw.lower()) for kw in keywords)
            # Saturating: 1 hit ~0.45, 2 ~0.9, 3+ -> 1.0. Keeps features in [0, 1].
            scores[topic] = min(1.0, 0.45 * hits)
        return scores

    def _keyword_encode(self, paper: Paper, interests: str, source: str = "mock") -> PaperFeatures:
        scores = self._keyword_scores(paper)
        text = paper.text.lower()
        novelty_markers = ("novel", "first", "state-of-the-art", "state of the art", "outperform", "new")
        novelty = 0.5 + 0.1 * sum(text.count(m) > 0 for m in novelty_markers)
        relevance = self._interest_overlap(paper, interests)
        return PaperFeatures(
            arxiv_id=paper.arxiv_id,
            topic_scores=scores,
            summary=_first_sentences(paper.abstract),
            novelty=min(1.0, novelty),
            relevance_hint=relevance,
            source=source,
            _topics=self.topics,
        )

    def _interest_overlap(self, paper: Paper, interests: str) -> float:
        if not interests:
            return 0.5
        terms = [t for t in re.findall(r"\w+", interests.lower()) if len(t) > 2]
        if not terms:
            return 0.5
        blob = paper.text.lower()
        hit = sum(1 for t in set(terms) if t in blob)
        return min(1.0, hit / max(3, len(set(terms))) * 1.5)

    # ------------------------------------------------------------------ LLM path
    def _build_prompt(self, paper: Paper, interests: str) -> tuple[str, str]:
        topics_list = "\n".join(f"- {t}" for t in self.topics)
        system = (
            "You are a precise research-paper feature extractor. Read ONLY the provided "
            "title and abstract. Do not use outside knowledge or invent facts. "
            "Respond with a single JSON object and nothing else."
        )
        user = (
            f"Researcher interests: {interests or 'general ML/AI'}\n\n"
            f"Title: {paper.title}\nAbstract: {paper.abstract}\n\n"
            "Return JSON with exactly these keys:\n"
            '  "topic_scores": an object scoring EACH of these topics from 0.0 to 1.0 by how '
            "central the topic is to the paper:\n"
            f"{topics_list}\n"
            '  "summary": one or two sentences, strictly grounded in the abstract.\n'
            '  "novelty": 0.0-1.0 estimate of how novel the paper claims to be.\n'
            '  "relevance": 0.0-1.0 estimate of relevance to the researcher interests above.\n'
        )
        return system, user

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        if not text:
            return None
        start, depth = text.find("{"), 0
        if start < 0:
            return None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _llm_encode(self, paper: Paper, interests: str) -> PaperFeatures:
        system, user = self._build_prompt(paper, interests)
        raw = self.client.complete(system, user)
        data = self._extract_json(raw)
        if not data or "topic_scores" not in data:
            # Graceful degradation: bad/unparriable output -> deterministic keyword features.
            return self._keyword_encode(paper, interests, source="fallback")
        raw_scores = data.get("topic_scores", {}) or {}
        scores = {t: _clamp01(raw_scores.get(t, 0.0)) for t in self.topics}
        return PaperFeatures(
            arxiv_id=paper.arxiv_id,
            topic_scores=scores,
            summary=str(data.get("summary", "")).strip() or _first_sentences(paper.abstract),
            novelty=_clamp01(data.get("novelty", 0.5)),
            relevance_hint=_clamp01(data.get("relevance", 0.5)),
            source="llm",
            _topics=self.topics,
        )

    # ------------------------------------------------------------------ public API
    def encode(self, paper: Paper, interests: str = "") -> PaperFeatures:
        if getattr(self.client, "is_mock", False):
            return self._keyword_encode(paper, interests)
        try:
            return self._llm_encode(paper, interests)
        except Exception as exc:  # never let reasoning crash the agent loop
            print(f"[reasoning] LLM encode failed for {paper.arxiv_id} ({exc}); keyword fallback.")
            return self._keyword_encode(paper, interests, source="fallback")

    def encode_many(self, papers: List[Paper], interests: str = "") -> List[PaperFeatures]:
        return [self.encode(p, interests) for p in papers]


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))

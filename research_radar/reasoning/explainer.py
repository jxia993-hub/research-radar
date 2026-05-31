"""LLM recommendation explainer.

This is the agent's *second* use of the LLM, and a heavier one than feature encoding:
it performs a small reasoning step. Given a candidate paper, the topics the RL policy
weighted it on, and the user's recent saved papers, it produces a short, personalised
"why you're seeing this" rationale — turning the bandit's numeric decision into a human
explanation grounded in the user's own history.

Switchable like the encoder: a real Claude/GPT backend writes the rationale; with no key
(mock backend) or on any failure, a deterministic template fallback keeps it working
offline. The template is honest — it states the matched topics and a recent related save
rather than inventing prose.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _template_explanation(title: str, matched_topics: List[str],
                          liked: List[Dict[str, Any]], exploit: float, explore: float) -> str:
    """Deterministic, grounded fallback used by the mock backend / on LLM failure."""
    topics_str = ", ".join(t.replace("_", " ") for t in matched_topics[:3]) or "your general area"
    # Find the most recent liked paper that shares a topic, to make it concrete.
    related: Optional[str] = None
    matched = set(matched_topics)
    for h in liked:
        if matched.intersection(h.get("top_topics", [])):
            related = h.get("title")
            break
    if not liked:
        # Cold start: no history yet, so it's exploration-driven.
        return (f"Surfaced while exploring (the agent hasn't learned your taste yet) — "
                f"this paper centres on {topics_str}. Save or skip it to start teaching the agent.")
    lead = "explore" if explore > abs(exploit) else "match"
    if lead == "match":
        base = f"Recommended because it matches topics you tend to save: {topics_str}."
    else:
        base = f"A mostly exploratory pick in {topics_str} — the agent is testing a less-certain direction for you."
    if related:
        base += f" Related to your earlier save “{related[:70]}”."
    return base


class RecommendationExplainer:
    def __init__(self, client, cfg: Dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg

    def _build_prompt(self, *, title: str, abstract: str, matched_topics: List[str],
                      liked: List[Dict[str, Any]], interests: str) -> tuple[str, str]:
        liked_lines = "\n".join(
            f"- {h.get('title','')[:90]} (topics: {', '.join(h.get('top_topics', [])) or 'n/a'})"
            for h in liked[:6]
        ) or "(none yet — this is a new user)"
        system = (
            "You explain to a researcher why a paper-recommender surfaced a given paper. "
            "Be concrete and grounded ONLY in the provided abstract and the user's saved "
            "papers. Do not invent results or facts. One or two sentences, no preamble, "
            "address the user as 'you'."
        )
        user = (
            f"User's stated interests: {interests or 'general ML/AI'}\n\n"
            f"Papers the user previously saved/liked:\n{liked_lines}\n\n"
            f"Candidate paper being recommended:\n"
            f"Title: {title}\nAbstract: {abstract}\n"
            f"Topics the recommender matched it on: {', '.join(matched_topics) or 'none strongly'}\n\n"
            "Write a 1-2 sentence rationale for why this paper is being recommended to THIS user, "
            "connecting it to their interests or prior saves where relevant."
        )
        return system, user

    def explain(self, *, title: str, abstract: str, matched_topics: List[str],
                liked: List[Dict[str, Any]], interests: str = "",
                exploit: float = 0.0, explore: float = 0.0) -> str:
        """Return a short rationale. Never raises — falls back to the template on any issue."""
        if getattr(self.client, "is_mock", False):
            return _template_explanation(title, matched_topics, liked, exploit, explore)
        try:
            system, user = self._build_prompt(
                title=title, abstract=abstract, matched_topics=matched_topics,
                liked=liked, interests=interests,
            )
            text = (self.client.complete(system, user) or "").strip()
            return text or _template_explanation(title, matched_topics, liked, exploit, explore)
        except Exception as exc:  # never break the UI over an explanation
            print(f"[explainer] LLM explain failed ({exc}); using template fallback.")
            return _template_explanation(title, matched_topics, liked, exploit, explore)

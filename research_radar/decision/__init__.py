"""Decision layer: contextual-bandit policies that rank papers and learn from feedback."""
from research_radar.decision.bandit import (
    BanditState,
    EpsilonGreedy,
    LinTS,
    LinUCB,
    RandomPolicy,
    StaticLLMPolicy,
    make_bandit,
)

__all__ = [
    "BanditState",
    "EpsilonGreedy",
    "LinTS",
    "LinUCB",
    "RandomPolicy",
    "StaticLLMPolicy",
    "make_bandit",
]

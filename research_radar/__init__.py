"""Research Radar — an LLM + RL agent that learns which arXiv papers you care about.

The package is organised around a classic agent loop:

    perceive (perception)  ->  reason (reasoning)  ->  decide (decision)
        ^                                                     |
        |------------------ learn (decision + memory) <-------+

`agent.ResearchRadarAgent` wires these together. See `docs/architecture.md`.
"""

__version__ = "0.4.0"

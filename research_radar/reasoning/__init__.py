"""Reasoning layer: an LLM turns raw paper text into a structured state for the RL agent."""
from research_radar.reasoning.encoder import PaperEncoder, PaperFeatures
from research_radar.reasoning.llm_client import build_llm_client

__all__ = ["PaperEncoder", "PaperFeatures", "build_llm_client"]

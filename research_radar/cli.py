"""Command-line interface for Research Radar.

    python -m research_radar.cli init --interests "RL, LLM agents, RLHF"
    python -m research_radar.cli recommend --query "llm agent reinforcement learning" --top 5
    python -m research_radar.cli feedback 2401.01234 save
    python -m research_radar.cli stats
    python -m research_radar.cli demo            # scripted end-to-end loop (great for the video)

Add --offline to any command to run without network (uses the bundled paper cache), and
--mock to force the offline encoder even if an API key is set.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from research_radar.agent import Recommendation, ResearchRadarAgent
from research_radar.config import load_config
from research_radar.safety.guards import confirm_action

BAR = "─" * 78


def _build_agent(args) -> ResearchRadarAgent:
    cfg = load_config(args.config)
    if getattr(args, "mock", False):
        cfg["llm"]["provider"] = "mock"
    if getattr(args, "memory", None):
        cfg["memory"]["path"] = args.memory
    return ResearchRadarAgent(cfg, offline=getattr(args, "offline", False), seed=getattr(args, "seed", None))


def _print_rec(r: Recommendation) -> None:
    tag = "exploring" if r.explore > abs(r.exploit) else "exploiting"
    topics = ", ".join(r.top_topics()) or "(no strong topic)"
    print(f"[{r.rank}] {r.paper.arxiv_id:<16} score {r.score:+.3f} "
          f"= exploit {r.exploit:+.3f} + explore {r.explore:.3f}  ({tag})")
    print(f"    {r.paper.title}")
    print(f"    topics : {topics}")
    print(f"    summary: {r.features.summary}")
    print(f"    link   : {r.paper.abs_url or 'https://arxiv.org/abs/' + r.paper.arxiv_id}")
    if r.warnings:
        print(f"    ⚠ safety: {'; '.join(r.warnings)}")
    print()


def _cmd_init(args) -> int:
    agent = _build_agent(args)
    agent.set_interests(args.interests)
    print(f"Saved interests: {args.interests!r}")
    print(f"Memory: {agent.memory.path}")
    return 0


def _cmd_recommend(args) -> int:
    agent = _build_agent(args)
    if args.interests:
        agent.set_interests(args.interests)
    query = args.query or agent.cfg["perception"]["default_query"]
    print(BAR)
    print(f"Research Radar · backend={agent.client.name} · policy={agent.bandit.name} · "
          f"interests={agent.interests or '(none set)'}")
    print(f"Query: {query!r}")
    print(BAR)
    recs = agent.run_cycle(query, k=args.top)
    if not recs:
        print("No new papers (all candidates already seen). Try a different --query or add --fresh.")
        return 0
    for r in recs:
        _print_rec(r)
    print("Give feedback, e.g.:  python -m research_radar.cli feedback "
          f"{recs[0].paper.arxiv_id} save")
    return 0


def _cmd_feedback(args) -> int:
    agent = _build_agent(args)
    try:
        reward = agent.learn(args.arxiv_id, args.action)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"Recorded '{args.action}' on {args.arxiv_id} (reward={reward:.2f}). Policy updated & saved.")
    s = agent.stats()
    print(f"Interactions so far: {s['interactions']} · avg reward {s['avg_reward']:.3f}")
    return 0


def _print_weights(agent: ResearchRadarAgent, top: int = 8) -> None:
    weights = agent.topic_weights()
    if not weights:
        return
    scale = max((abs(w) for _, w in weights), default=1.0) or 1.0
    print("Learned topic preferences (θ):")
    for topic, w in weights[:top]:
        bars = int(round(abs(w) / scale * 24))
        sign = "+" if w >= 0 else "-"
        print(f"  {topic:<24} {sign}{abs(w):.3f} {'█' * bars}")


def _cmd_stats(args) -> int:
    agent = _build_agent(args)
    s = agent.stats()
    print(BAR)
    print(f"backend={s['llm_backend']} · policy={s['bandit']}")
    print(f"interactions={s['interactions']} · avg_reward={s['avg_reward']:.3f} · "
          f"seen={s['seen']} · bias={s['bias']:+.3f}")
    print(f"by_action={s['by_action']}")
    print(BAR)
    _print_weights(agent)
    return 0


def _cmd_demo(args) -> int:
    """Scripted perceive->reason->decide->act->learn loop, then a second round that
    shows the queue adapting. Designed to be screen-recorded for the 2-minute video."""
    args.offline = True if args.offline is None else args.offline
    agent = _build_agent(args)
    interests = args.interests or "reinforcement learning, LLM agents, RLHF and alignment"
    agent.set_interests(interests)
    query = args.query or "reinforcement learning language model agent"

    print(BAR)
    print("RESEARCH RADAR — scripted demo")
    print(f"backend={agent.client.name} · policy={agent.bandit.name}")
    print(f"interests={interests}")
    print(BAR)

    print("\n[ROUND 1] perceive -> reason -> decide\n")
    recs = agent.run_cycle(query, k=args.top)
    for r in recs:
        _print_rec(r)

    # Simulate a user who likes papers matching their stated interests.
    liked = _liked_topics(interests, agent.topics)
    print("[LEARN] simulating feedback (save = matches your interests, skip = does not):\n")
    for r in recs:
        action = "save" if (set(r.top_topics()) & liked) else "skip"
        reward = agent.learn(r.paper.arxiv_id, action)
        print(f"  {action:<5} {r.paper.arxiv_id}  (reward {reward:.1f})  {r.paper.title[:54]}…")

    print("\n[INTROSPECT] the agent has updated its preferences:\n")
    _print_weights(agent)

    print("\n[ROUND 2] same query, new papers, ranked by the *updated* policy:\n")
    recs2 = agent.run_cycle(query, k=args.top)
    if recs2:
        for r in recs2:
            _print_rec(r)
    else:
        print("  (offline cache exhausted — in live mode arXiv returns fresh papers here)")
    print(BAR)
    print("Demo complete. Run `python experiments/run_learning_curve.py` for the RL evaluation.")
    return 0


def _cmd_reset(args) -> int:
    """Erase learned preferences & history — an irreversible action, so it is gated by the
    safety confirmation guard (demonstrating where human-in-the-loop approval belongs)."""
    agent = _build_agent(args)
    path = agent.memory.path
    if not path.exists():
        print(f"No memory at {path}; nothing to reset.")
        return 0
    if not confirm_action(f"This permanently deletes learned preferences & history at {path}.",
                          require_confirmation=agent.safety.get("require_confirmation", True),
                          auto_yes=args.yes):
        print("Aborted — nothing was deleted.")
        return 1
    path.unlink()
    print(f"Reset complete: deleted {path}.")
    return 0


def _liked_topics(interests: str, topics: List[str]) -> set:
    text = interests.lower()
    return {t for t in topics if any(w in text for w in t.split("_"))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research-radar", description="LLM + RL arXiv triage agent")
    p.add_argument("--config", default="config.json", help="path to config.json")
    p.add_argument("--offline", action="store_true", default=False, help="use bundled cache, no network")
    p.add_argument("--mock", action="store_true", default=False, help="force offline keyword encoder")
    p.add_argument("--memory", default=None, help="override memory file path")
    p.add_argument("--seed", type=int, default=None, help="seed for stochastic policies")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="set your research interests")
    pi.add_argument("--interests", required=True)
    pi.set_defaults(func=_cmd_init)

    pr = sub.add_parser("recommend", help="fetch + rank papers")
    pr.add_argument("--query", default=None)
    pr.add_argument("--interests", default=None, help="optionally (re)set interests")
    pr.add_argument("--top", type=int, default=5)
    pr.set_defaults(func=_cmd_recommend)

    pf = sub.add_parser("feedback", help="record feedback on a paper")
    pf.add_argument("arxiv_id")
    pf.add_argument("action", choices=["save", "like", "read", "skip", "dislike"])
    pf.set_defaults(func=_cmd_feedback)

    ps = sub.add_parser("stats", help="show learned preferences + history")
    ps.set_defaults(func=_cmd_stats)

    pd = sub.add_parser("demo", help="scripted end-to-end demonstration")
    pd.add_argument("--query", default=None)
    pd.add_argument("--interests", default=None)
    pd.add_argument("--top", type=int, default=4)
    pd.set_defaults(func=_cmd_demo, offline=None)

    prs = sub.add_parser("reset", help="erase learned preferences & history (asks to confirm)")
    prs.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    prs.set_defaults(func=_cmd_reset)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Research Radar — Streamlit web UI.

A thin visual layer over the same `ResearchRadarAgent` used by the CLI. It makes the
LLM+RL loop tangible: type a query, see ranked paper cards, click 👍/📖/👎, and watch the
agent's *learned preference* bar chart update live as the contextual bandit adapts to you.

Run from the repo root:
    pip install -r requirements-ui.txt
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from research_radar.agent import ResearchRadarAgent  # noqa: E402
from research_radar.config import load_config  # noqa: E402

st.set_page_config(page_title="Research Radar", page_icon="🛰️", layout="wide")

ACTION_EMOJI = {"save": "👍 Save", "read": "📖 Read", "skip": "👎 Skip"}


def build_agent(offline: bool, mock: bool) -> ResearchRadarAgent:
    """Fresh agent each rerun — it reloads the learned policy from the memory file, so
    learning persists across Streamlit reruns via the existing JSON store."""
    cfg = load_config("config.json")
    if mock:
        cfg["llm"]["provider"] = "mock"
    return ResearchRadarAgent(cfg, offline=offline, seed=0)


def rec_to_dict(r) -> dict:
    return {
        "id": r.paper.arxiv_id, "title": r.paper.title,
        "topics": r.top_topics(), "summary": r.features.summary,
        "link": r.paper.abs_url or f"https://arxiv.org/abs/{r.paper.arxiv_id}",
        "score": r.score, "exploit": r.exploit, "explore": r.explore,
    }


def preference_chart(agent: ResearchRadarAgent):
    weights = agent.topic_weights()[:10][::-1]  # top 10, highest plotted at top
    topics = [t for t, _ in weights]
    vals = [w for _, w in weights]
    colors = ["#2e7d32" if v > 0.01 else "#bdbdbd" for v in vals]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.barh(topics, vals, color=colors)
    ax.set_xlabel("learned weight  θ")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar (controls)
st.sidebar.title("🛰️ Research Radar")
st.sidebar.caption("LLM encodes papers → contextual bandit (RL) learns your taste from feedback.")

interests = st.sidebar.text_input("Your interests", value="reinforcement learning, LLM agents, RLHF")
query = st.sidebar.text_input("Search query", value="llm agent reinforcement learning")
top_k = st.sidebar.slider("How many to show", 3, 10, 5)
offline = st.sidebar.checkbox("Offline (use cached papers)", value=True)
mock = st.sidebar.checkbox("Force offline LLM (mock encoder)", value=False)

agent = build_agent(offline, mock)
st.sidebar.caption(f"backend: `{agent.client.name}`  ·  policy: `{agent.bandit.name}`")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("🔍 Recommend", use_container_width=True):
    agent.set_interests(interests)
    st.session_state.recs = [rec_to_dict(r) for r in agent.run_cycle(query, k=top_k)]
if col_b.button("♻️ Reset", use_container_width=True):
    if agent.memory.path.exists():
        agent.memory.path.unlink()
    st.session_state.recs = []
    st.rerun()

st.session_state.setdefault("recs", [])


def rate(arxiv_id: str, action: str) -> None:
    agent.learn(arxiv_id, action)
    st.session_state.recs = [r for r in st.session_state.recs if r["id"] != arxiv_id]
    st.toast(f"Recorded '{action}' — policy updated", icon="✅")
    st.rerun()


# ---------------------------------------------------------------- main layout
st.title("🛰️ Research Radar")
st.caption("An arXiv triage agent: the **LLM** turns each abstract into a topic feature vector; "
           "a **contextual bandit (RL)** ranks papers — balancing *exploit* (predicted value) and "
           "*explore* (uncertainty) — and learns your taste from each 👍 / 👎.")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Your reading queue")
    if not st.session_state.recs:
        st.info("Set your interests and a query in the sidebar, then click **🔍 Recommend**.")
    for r in st.session_state.recs:
        with st.container(border=True):
            st.markdown(f"**{r['title']}**")
            topics = " · ".join(r["topics"]) or "(no strong topic)"
            st.caption(f"🏷️ {topics}  ·  arXiv:{r['id']}")
            st.write(r["summary"])
            st.caption(f"score **{r['score']:.2f}**  =  exploit {r['exploit']:.2f}  +  "
                       f"explore {r['explore']:.2f}")
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            if c1.button(ACTION_EMOJI["save"], key=f"save_{r['id']}", use_container_width=True):
                rate(r["id"], "save")
            if c2.button(ACTION_EMOJI["read"], key=f"read_{r['id']}", use_container_width=True):
                rate(r["id"], "read")
            if c3.button(ACTION_EMOJI["skip"], key=f"skip_{r['id']}", use_container_width=True):
                rate(r["id"], "skip")
            c4.markdown(f"[open on arXiv ↗]({r['link']})")

with right:
    st.subheader("What the agent has learned")
    st.pyplot(preference_chart(agent))
    plt.close("all")
    stats = agent.stats()
    m1, m2 = st.columns(2)
    m1.metric("Interactions", stats["interactions"])
    m2.metric("Avg reward", f"{stats['avg_reward']:.2f}")
    st.caption("Higher bars = topics you tend to save. The bandit ranks new papers by these "
               "weights, so the queue adapts as you give feedback.")
    with st.expander("📈 RL evaluation (offline experiment)"):
        curve = Path(__file__).resolve().parent / "results" / "learning_curve.png"
        if curve.exists():
            st.image(str(curve), caption="LinUCB/LinTS beat a static-LLM score and random.")
        else:
            st.caption("Run `python experiments/run_learning_curve.py` to generate this figure.")

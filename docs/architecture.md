# Research Radar — System Design

## 1. Goal and agent framing

**Goal the agent pursues:** *keep a researcher on top of the arXiv literature that matters
to them, with as little wasted reading as possible.*

Research Radar is structured as a closed perceive → reason → decide → act → learn loop, with
**memory** and **safety** as cross-cutting concerns:

| Stage | Responsibility | Module |
|-------|----------------|--------|
| Perceive | Observe the world: fetch candidate papers | `perception/arxiv_source.py` |
| Reason | Encode each observation into a structured state | `reasoning/encoder.py`, `reasoning/llm_client.py` |
| Decide | Choose what to surface (explore vs exploit) | `decision/bandit.py` |
| Act | Present a ranked reading queue with provenance | `agent.py`, `cli.py` |
| Learn | Convert feedback to reward, update the policy | `agent.py` → `decision/bandit.py` |
| Remember | Persist profile, policy and history across runs | `memory/store.py` |
| Stay safe | Rate-limit, sanitise, ground, confirm | `safety/guards.py` |

## 2. Why LLM **and** RL (the core design decision)

The course is about LLMs *and* RL, so the design uses each for what it is best at and lets
them compose:

* **LLM as the state encoder.** Abstracts are unstructured text. The LLM reads each abstract
  and emits a structured vector of topic scores over a fixed taxonomy (12 topics such as
  `reinforcement_learning`, `llm_agents`, `rlhf_alignment`, …), plus a grounded one-line
  summary. This vector `x ∈ ℝ¹³` (12 topics + bias) is the **context / state**.

* **RL as the decision-maker.** A contextual bandit learns a linear value function
  `r ≈ θ·x` from the user's feedback. Given a fresh batch of papers it scores each by

  ```
  score(x) = θ·x                 (exploit: predicted value)
           + α · sqrt(xᵀ A⁻¹ x)  (explore: uncertainty bonus, LinUCB)
  ```

  and ranks by `score`. Feedback (save/read/skip/…) is mapped to a scalar reward and the
  model is updated online: `A ← A + xxᵀ`, `b ← b + r·x`, `θ = A⁻¹b`.

This is a faithful instance of the **contextual-bandit** problem (Sutton & Barto, ch. 2;
Li et al., 2010). It is genuine RL — sequential decisions under uncertainty with a reward
signal and an explicit exploration/exploitation trade-off — while staying simple enough to
learn online within a single session and to evaluate reproducibly.

**What RL adds over a pure LLM recommender:** an LLM-only system ranks by a fixed zero-shot
relevance score — it never adapts to *you* and never explores. The bandit (a) personalises
from feedback and (b) deliberately explores under-seen topic directions, which is why it
beats the static-LLM baseline in `experiments/run_learning_curve.py`.

## 3. Component detail

### Perception (`perception/arxiv_source.py`)
Queries the public arXiv Atom API (no key → reproducible). Parses entries into `Paper`
objects. On any network failure or `offline=True`, it transparently loads a bundled cache of
**real** papers (`data/sample_papers.json`, built by `data/build_cache.py`). Perception never
raises — it degrades gracefully so the agent loop is robust.

### Reasoning (`reasoning/`)
`build_llm_client` selects a backend from config + environment: Anthropic or OpenAI via plain
HTTP (no SDK dependency), else an offline **mock**. `PaperEncoder` prompts a real model for
strict JSON; for the mock (or if parsing fails) it computes the same topic vector by keyword
matching against the taxonomy. Identical feature space in both modes ⇒ offline reproducibility.
The mock path also makes the RL experiment deterministic.

### Decision (`decision/bandit.py`)
A shared ridge-regression core (`A`, `b`, `θ = A⁻¹b`) with five policies behind one
`evaluate / select / update` interface: **LinUCB** (default), **LinTS** (Thompson sampling),
**ε-greedy**, **random** (floor baseline) and **static-LLM** (rank by a fixed weight vector,
never learns — the "just trust the LLM" baseline). Each `evaluate` returns the exploit term,
the explore term and the combined score, so the CLI can explain *why* a paper was ranked
where it was.

### Memory (`memory/store.py`)
A single JSON file persists the user profile (interests), the serialised bandit state (the
learned `A`, `b` — i.e. the learned preferences), the full interaction history, the set of
seen ids, and a small "pending" table (recommendations shown but not yet rated, so feedback
issued in a *later* CLI invocation can still recover the feature vector and update the policy).
Persisting the policy is what makes the agent improve **across** sessions, not just within one.

### Safety (`safety/guards.py`)
`RateLimiter` (sliding window) caps external calls; `sanitize_query` bounds/cleans input;
`grounding_check` flags ungrounded summaries (extra numbers / over-length) and triggers a
verbatim-extractive fallback; `confirm_action` gates any irreversible action behind a human
y/n — it is wired to the `reset` command, which refuses to delete learned state without
explicit approval (`--yes` or an interactive *y*). Honest scope: prototype-level tripwires
that show *where* safety hooks belong.

## 4. Evaluation design (`experiments/run_learning_curve.py`)

Because real human feedback is too slow to show learning in a short demo, evaluation uses a
**simulated user** with a hidden linear preference vector over topics; reward is the clipped,
noisy inner product with a paper's features. All policies see an *identical* environment per
seed (same paper pool, candidate draws and reward noise) so the only difference is the
decision rule. We report average reward per round and cumulative regret (mean ± std over
seeds). Expected results: `linucb ≈ lints > ε-greedy > static-llm > random`.

## 5. Key design trade-offs

* **Contextual bandit over full sequential RL (Q-learning/PG).** Recommendation reward is
  effectively immediate, so a bandit is the right tool: it converges within a session, has a
  principled UCB/Thompson exploration story, and is easy to evaluate and explain. A multi-step
  MDP would add state-transition machinery for little benefit at this scale.
* **LLM topic vector over raw embeddings.** Topic scores keep the learned `θ` *interpretable*
  ("you value RL and alignment, not vision"), need no embedding endpoint, and let the mock and
  real backends share one feature space. The cost is a hand-specified taxonomy (in `config.json`).
* **Switchable backend with offline fallback.** Maximises both demo quality (real LLM) and
  reproducibility (zero-setup offline), at the cost of a thin branch in the encoder.

## 6. Possible extensions

Dense embeddings as additional features; a non-linear value head; per-topic LinUCB (disjoint
model); active-learning-style diversity in the shown batch; a real inbox/RSS perception source;
and an "open/email digest" action gated by the existing `confirm_action` hook.

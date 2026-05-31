# Changelog

All notable changes to Research Radar are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); the project uses semantic-ish versioning.

## [0.5.0] - 2026-05-31
### Added
- **LLM "Why this?" explainer** (`reasoning/explainer.py`): generates a personalised,
  history-grounded natural-language rationale for each recommendation — the LLM now acts
  as a reasoner, not only a feature encoder. Exposed via `recommend --why`, a `why <id>`
  command, the scripted `demo`, and a "Why this?" button in the web UI.
- **Pip-installable package** (`pyproject.toml`) with a `research-radar` console entry
  point, so the CLI runs from any directory (no more `python -m` path pitfalls).
- **Continuous integration** (GitHub Actions): runs the test suite + CLI/experiment smoke
  tests on Python 3.9 / 3.11 / 3.12.
- Project metadata for citation (`CITATION.cff`), contribution guide, and this changelog.

## [0.4.0] - 2026-05-27
### Added
- **Streamlit web UI** (`app.py`): query box, ranked paper cards with save/read/skip
  buttons, and a live bar chart of the bandit's learned topic preferences.
- `reset` command gated by the safety confirmation guard.

### Changed
- ASCII-safe terminal output (removed decorative Unicode that rendered as mojibake).

## [0.3.0] - 2026-05-27
### Added
- Simulated-user environment and RL learning-curve experiment
  (`experiments/run_learning_curve.py`) comparing LinUCB / LinTS against ε-greedy,
  static-LLM and random baselines.
- Offline cache of real arXiv papers (RSS-based builder) for zero-network reproduction.
- Unit tests (bandit-learns, encoder, memory round-trip, agent loop).

## [0.2.0] - 2026-05-27
### Added
- Full agent loop: perception (arXiv + offline fallback), reasoning (switchable
  Anthropic / OpenAI / mock LLM as a topic-feature encoder), decision (LinUCB / LinTS /
  baselines), durable memory, and prototype safety guards.

## [0.1.0] - 2026-05-27
### Added
- Initial scaffold: package layout, configuration with a research-topic taxonomy.

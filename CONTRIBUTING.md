# Contributing to Research Radar

Thanks for your interest! This started as a COMPSCI 767 (LLM + RL) course project and is
kept small and readable on purpose. Contributions that preserve that spirit are welcome.

## Development setup

```bash
git clone https://github.com/jxia993/research-radar
cd research-radar
pip install -e ".[dev,ui]"      # editable install + test/build + Streamlit
```

## Running the checks

```bash
python -m unittest discover -s tests -v       # unit tests
research-radar --offline demo                 # end-to-end smoke test (no network/key)
python experiments/run_learning_curve.py      # regenerate the RL figure
```

Please make sure the tests pass before opening a PR. CI runs the same checks on
Python 3.9 / 3.11 / 3.12.

## Project layout

See the "Project layout" section of the [README](README.md) and the design rationale in
[`docs/architecture.md`](docs/architecture.md). The package follows the agent loop:
`perception → reasoning → decision → memory/safety`, orchestrated by `agent.py`.

## Ideas that fit the project

- New perception sources (RSS, Semantic Scholar, a local PDF folder).
- Additional bandit policies, or per-topic (disjoint) LinUCB.
- Dense embeddings as extra features alongside the topic vector.
- Diversity-aware re-ranking of the shown batch.

## Guidelines

- Keep dependencies minimal (numpy / matplotlib / requests; Streamlit is optional).
- Match the existing style: clear docstrings explaining *why*, not just *what*.
- Every external call should pass through the safety guards; LLM use should degrade
  gracefully to the offline path when no API key is present.
- Add or update a test for behavioural changes.

By contributing, you agree your contributions are licensed under the project's MIT license.

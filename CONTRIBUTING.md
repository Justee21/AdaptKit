# Contributing

Use Python (3.11)–(3.13):

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,evaluation]'
.venv/bin/python -m pytest
.venv/bin/mypy adaptkit examples benchmarks tests
.venv/bin/pyright adaptkit examples benchmarks
```

Build checks:

```bash
.venv/bin/python scripts/release_build.py --check
```

The guarded script derives the version from `pyproject.toml`, runs Twine against the two verified paths, and installs the wheel and source distribution in separate temporary environments outside the repository. Never upload with a wildcard. Review and publish only the two exact artifacts it reports.

Normal tests must remain offline and provider-independent. Real-model benchmarks are explicit, paid, nondeterministic operations; never run them in CI or in a contribution without authorization. Do not add request text, responses, API keys, `.env`, local databases, or raw shadow traffic to artifacts.

Keep changes within the small SDK boundary. Add tests for public behavior and run both type checkers before opening a pull request.

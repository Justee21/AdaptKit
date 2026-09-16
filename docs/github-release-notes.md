AdaptKit 0.1.0 is a public-alpha, provider-independent Python SDK that learns per-user agent behavior preferences using contextual bandits.

## What is included

- Per-user, per-context Thompson Sampling over application-defined actions.
- Explicit, implicit, passive, and pairwise feedback with idempotent atomic updates.
- Initial-prompt routing for clear first-turn behavior requests.
- Active and shadow learning modes.
- In-memory and single-host SQLite storage.
- Per-action cold-start priors, plus optional one-time Mem0 and Letta examples.
- Provider-neutral sync and async extractor contracts.
- An optional OpenAI Responses API Structured Outputs example using `store=False`.

AdaptKit returns a structured action key. Your application maps that key to model instructions, workflow branches, tool permissions, or UI behavior. It does not own the model call or automatically rewrite prompts.

## Install

```bash
python -m pip install adaptkit
```

See the README for the quick start, optional dependencies, interactive playground, evaluation methodology, and privacy boundaries.

## Public-alpha limits

This release is package-tested and suitable for evaluation; it is not production-validated. It uses mutually exclusive actions, discrete contexts, immediate binary rewards, and single-host SQLite. It does not provide cross-context generalization, preference decay, long-horizon credit assignment, distributed storage, or automatic memory synchronization.

Production integrations should begin in shadow mode and calibrate evaluator thresholds on representative traffic.

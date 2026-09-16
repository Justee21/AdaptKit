# Changelog

## 0.1.0 — public alpha

Package-tested and suitable for evaluation. Production integrations should begin in shadow mode and calibrate thresholds on representative traffic.

- Added per-user, per-context Thompson Sampling with atomic explicit, implicit, passive, and pairwise feedback updates.
- Added immutable decisions, idempotent observations, active/shadow learning modes, per-action priors, stable action-set fingerprints, and SQLite schema 1 → 2 migration.
- Added current-prompt routing, sync/async evaluator interfaces, application-owned action-to-instruction mapping examples, and privacy-safe structured exports.
- Added optional OpenAI evaluation examples, Mem0 and Letta cold-start examples, deterministic smoke tests, opt-in real-model benchmarks, and shadow-traffic scoring.

Known boundaries: mutually exclusive discrete actions, discrete contexts, one-host SQLite, binary immediate rewards, no preference decay or cross-context generalization, and no production-validation claim. This is a contextual-bandit personalization SDK, not a general reinforcement-learning framework.

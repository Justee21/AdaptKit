# `0.1.0` release checklist

- [ ] Review the changelog, README, package metadata, and known alpha boundaries.
- [ ] Run tests, mypy, pyright, builds, Twine validation, and clean wheel/sdist install checks on Python (3.11)–(3.13).
- [ ] Run `python scripts/release_build.py --check`; confirm it derives and reports exactly one canonical wheel and one source distribution from `pyproject.toml`.
- [ ] Confirm the guarded script passes Twine and separate outside-the-tree wheel/sdist installation checks. Never upload `dist/*`.
- [ ] Confirm `.env`, databases, raw shadow traffic, caches, and credentials are absent from Git and both distributions.
- [ ] Confirm committed benchmark summaries match README numbers; do not rerun paid evaluations merely to release.
- [ ] Validate representative application traffic in shadow mode and document remaining risk.
- [ ] Configure and verify PyPI trusted publishing and GitHub private vulnerability reporting manually.
- [ ] Review the prepared copy and manual commands in `docs/launch.md`.
- [ ] Create a reviewed tag/release, then publish; no release automation is enabled in this candidate.

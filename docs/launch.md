# AdaptKit 0.1.0 launch packet

This file prepares copy and commands for manual review. Nothing here runs automatically.

## Positioning

AdaptKit 0.1.0 is a public-alpha, provider-independent Python SDK that learns per-user agent behavior preferences using contextual bandits.

Public alpha: package-tested and suitable for evaluation. Production integrations should begin in shadow mode and calibrate thresholds on representative traffic.

## GitHub release

Title: **AdaptKit 0.1.0 — Public Alpha**

Use `docs/github-release-notes.md` as the release body.

## X draft

AdaptKit 0.1.0 is now a public alpha: a provider-independent Python SDK that learns per-user agent behavior preferences with contextual bandits. It returns action keys you map to prompts, workflows, or tools. Start in shadow mode. https://github.com/Justee21/AdaptKit

## Blog-post outline

1. **Why agent memory is not enough**
   - Retrieval recalls facts and text; it does not learn which response behavior works.
   - Introduce a small action space such as code-first versus explanation-first.
2. **The AdaptKit loop**
   - Choose an action, map it to visible application behavior, observe an outcome, update the posterior.
   - Explain per-user, per-context state and immutable decisions.
3. **Why Beta–Bernoulli Thompson Sampling**
   - Show \(r\in\{0,1\}\), \(\alpha\leftarrow\alpha+r\), and \(\beta\leftarrow\beta+(1-r)\).
   - Explain exploration, exploitation, binary feedback, and confidence gating.
4. **How an action changes an actual agent**
   - Map `Decision.action` to a system instruction, workflow branch, or tool allowlist.
   - Emphasize that AdaptKit remains provider-independent and does not own model calls.
5. **Four feedback paths**
   - Explicit likes/dislikes, implicit conversation classification, deliberately mapped passive signals, and pairwise preference.
6. **Cold start and memory systems**
   - Describe modest per-action priors and the optional one-time Mem0/Letta examples.
   - Explain why raw memory text is never trusted as policy state.
7. **Evaluation without overclaiming**
   - Separate deterministic smoke tests, hand-authored real-model benchmarks, and representative shadow traffic.
   - Report the recorded metrics with confidence intervals and their limitations.
8. **Public-alpha boundaries and next steps**
   - Cover discrete contexts, immediate rewards, single-host SQLite, provider-specific calibration, and missing production evidence.
   - Invite developers to evaluate in shadow mode and report integration feedback.

## Manual launch sequence

Run from the repository root after reviewing every local change:

```bash
git status --short
git diff --check
python -m pip install -e '.[dev,evaluation]'
python -m pytest
mypy adaptkit examples benchmarks tests
pyright adaptkit examples benchmarks
python scripts/release_build.py --check
```

Stage only the files you reviewed and intend to release; no blanket staging command is provided because that scope is a human decision. Then inspect the staged snapshot before committing:

```bash
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Prepare AdaptKit 0.1.0 public alpha"
git push origin main
```

Wait for the hosted Python 3.11–3.13 test matrix to pass. Then create and push the reviewed tag:

```bash
git tag -a v0.1.0 -m "AdaptKit 0.1.0 public alpha"
git push origin v0.1.0
```

Publish only the two exact artifacts reported by the guarded build:

The command below assumes Twine authentication is configured securely through a keyring or another reviewed credential mechanism. Never place a PyPI token in the repository or command history. If you configure PyPI Trusted Publishing, use its reviewed workflow instead.

```bash
python -m twine upload \
  dist/adaptkit-0.1.0-py3-none-any.whl \
  dist/adaptkit-0.1.0.tar.gz
```

Create the GitHub release only after the package page resolves:

```bash
gh release create v0.1.0 \
  --title "AdaptKit 0.1.0 — Public Alpha" \
  --notes-file docs/github-release-notes.md
```

## Final manual checks

- [ ] Review `git diff --cached --stat` and `git diff --cached`; confirm `.env`, databases, caches, generated traffic, and unrelated files are absent.
- [ ] Confirm hosted CI passes on Python 3.11, 3.12, and 3.13.
- [ ] Confirm the Git tag points to the reviewed commit.
- [ ] Confirm the PyPI project shows version 0.1.0, Python requirement, MIT license, project URLs, and rendered README.
- [ ] Install `adaptkit==0.1.0` into a new environment from PyPI and import it outside the source tree.
- [ ] Confirm the GitHub release title, notes, tag, and PyPI link.
- [ ] Verify every README link from the public GitHub page and PyPI page.
- [ ] Enable or verify GitHub private vulnerability reporting.
- [ ] Post the announcement only after the repository, package, and release links resolve publicly.

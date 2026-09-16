# AdaptKit

AdaptKit 0.1.0 is a public-alpha, provider-independent Python SDK that learns per-user agent behavior preferences using contextual bandits. Its built-in learner applies Thompson Sampling independently for each user and named context to choose among mutually exclusive actions.

> **Public alpha:** Package-tested and suitable for evaluation. Production integrations should begin in shadow mode and calibrate thresholds on representative traffic.

```text
choose action → map action to an instruction → agent responds → observe feedback → update policy
```

AdaptKit chooses an action key. Your application decides what that key means and sends the mapped instruction to its model. AdaptKit does not own the agent, rewrite prompts, or automatically send `Decision.action` to an LLM.

Ordinary memory retrieval brings stored facts or text back into a model's context. AdaptKit instead maintains a small probability distribution over application-defined behaviors and updates it from outcomes. The two can complement each other: durable memory can seed modest cold-start priors, while AdaptKit learns which behavior works from subsequent feedback.

## Installation

Install the public alpha from PyPI after release:

```bash
python -m pip install adaptkit
```

For an unreleased checkout or contributor install:

```bash
git clone https://github.com/Justee21/AdaptKit.git
cd AdaptKit
python -m pip install -e .
```

Install the optional OpenAI integration dependencies alongside the package with:

```bash
python -m pip install 'adaptkit[evaluation]'
```

Runnable examples, benchmarks, and `.env.example` live in the repository rather than the importable wheel. From a checkout, use:

```bash
python -m pip install -e '.[evaluation]'
cp .env.example .env
```

Set these values in the ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_JUDGE_MODEL=your-structured-output-capable-model
OPENAI_AGENT_MODEL=your-agent-model
ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD=your-calibrated-threshold
ADAPTKIT_PROMPT_CONFIDENCE_THRESHOLD=your-calibrated-threshold
```

`OPENAI_AGENT_MODEL` is optional and falls back to `OPENAI_JUDGE_MODEL`. Both thresholds are required by the corresponding real-model examples and must be calibrated for the exact judge model, classifier prompt, and traffic. The recorded Terra values of \(0.95\) for feedback and \(0.70\) for prompt routing are benchmark results, not universal defaults. Never commit `.env`; only `.env.example` belongs in Git.

Memory SDKs are independent extras; install only the provider you use. The runnable integration examples remain in the repository checkout:

```bash
python -m pip install 'adaptkit[memory-mem0]'
# or
python -m pip install 'adaptkit[memory-letta]'
```

## Quick start

```python
from adaptkit import Profile, SQLiteStore

instructions = {
    "patch_first": "Show the code fix first, then explain briefly.",
    "explanation_first": "Explain the cause first, then show the fix.",
}

profile = Profile(
    user_id="user-123",
    actions=tuple(instructions),
    store=SQLiteStore("adaptkit.db"),
)

decision = profile.choose("debugging")

# `decision.action` is an app-owned lookup key. Its mapped value—not the key—
# is normally placed in the model's system/developer/instructions channel.
response = agent.run(
    prompt=user_prompt,
    system_instruction=instructions[decision.action],
)

# A durable event ID from your application is the best idempotency key.
profile.like(decision, idempotency_key="thumbs-up-event-456")
```

The returned immutable `Decision` records its ID, user, context, selected action, creation time, and the policy version that produced it. `policy_version` begins at \(0\) and increments once whenever that user's policy for that context changes.

## Learning and reward semantics

Each `(user, context, action)` starts with a posterior of \(\operatorname{Beta}(1,1)\). Every accepted observation has a binary selected-action-relative reward:

\[
r\in\{0,1\},\qquad
\alpha\leftarrow\alpha+r,\qquad
\beta\leftarrow\beta+(1-r)
\]

This Beta–Bernoulli model is intentional: positive evidence reinforces the selected action and negative evidence moves away from it. Confidence controls whether implicit or passive evidence is accepted; it is not the reward magnitude and does not fractionalize updates in release 0.1.0. Passive telemetry must be deliberately translated by the application into positive or negative evidence for the selected action.

Continuous rewards, delayed or multi-step credit, Gaussian or linear bandits, and contextual-feature learners are possible future learner types. They are not part of 0.1.0. AdaptKit is a contextual-bandit personalization SDK, not a general reinforcement-learning framework.

## Current-prompt personalization

An optional prompt extractor can honor an explicit request on the first turn, before any behavioral history exists:

```python
from adaptkit import LLMPromptPreferenceExtractor, Profile

def prompt_judge(messages):
    return {
        "action": "guided_learning",  # None means no clear cue
        "confidence": 0.96,
        "reason": "User explicitly requested guided reasoning.",
    }

descriptions = {
    "direct_solution": "Give the complete solution immediately.",
    "guided_learning": "Give useful hints before the complete solution.",
}
profile = Profile(
    user_id="user-123",
    actions=tuple(descriptions),
    prompt_evaluator=LLMPromptPreferenceExtractor(
        judge=prompt_judge,
        action_descriptions=descriptions,
    ),
    prompt_confidence_threshold=0.90,
)
decision = profile.choose("algorithms", prompt=user_prompt)
```

The extractor receives the context, current prompt, available actions, and their descriptions. A valid cue at or above the threshold overrides sampling for that turn without updating the posterior. No cue, an ambiguous cue, a weak cue, or a runtime evaluator failure falls back to Thompson Sampling. An empty or whitespace-only prompt is valid and means “no behavioral cue”; a supplied non-string prompt raises `ValidationError` before evaluator fallback. `PromptPreference.reason` is transient; the prompt, exception message, and reason are never persisted or emitted through lifecycle events. `achoose()` supports asynchronous extractors. Passing a nonempty prompt without configuring `prompt_evaluator` raises `ConfigurationError`.

See [`examples/openai_prompt_router.py`](https://github.com/Justee21/AdaptKit/blob/main/examples/openai_prompt_router.py) for an optional OpenAI Responses API and Structured Outputs implementation using `store=False`.

## Per-action cold-start priors

```python
from adaptkit import BetaPrior, Profile

profile = Profile(
    user_id="user-123",
    actions=["direct_solution", "guided_learning"],
    action_priors={
        "direct_solution": BetaPrior(alpha=1, beta=3),
        "guided_learning": BetaPrior.from_mean_strength(0.8, 5),
    },
)
```

Unspecified actions use `prior_alpha` and `prior_beta`. Priors are materialized the first time a user/context policy is used, remain at policy version \(0\), and cannot be changed by reopening the same stored policy with different constructor values. Existing SQLite state always wins. Applications may derive these priors from their own structured account memory or another profile's exported posterior; AdaptKit does not generalize across contexts automatically.

For external memory, keep translation application-owned:

```python
from adaptkit import Profile, cold_start_priors

# Your application retrieves memory, validates that it is durable, and maps it
# to one known action. AdaptKit never accepts or stores the raw memory text.
preferred_action = app_map_memory_to_action(memory_record)
profile = Profile(
    user_id="user-123",
    actions=["direct_solution", "guided_learning"],
    action_priors=cold_start_priors(
        actions=["direct_solution", "guided_learning"],
        preferred_action=preferred_action,
    ),
)
```

The default translated preference is deliberately modest: `guided_learning` becomes \(\operatorname{Beta}(4,2)\), while unspecified actions keep the profile default. “For this answer, be concise” is a temporary routing cue; “I generally prefer concise answers” may be a durable memory. Contradictory, ambiguous, or non-allowlisted memories yield neutral priors. See the optional [Mem0 example](https://github.com/Justee21/AdaptKit/blob/main/examples/mem0_cold_start.py), [Letta example](https://github.com/Justee21/AdaptKit/blob/main/examples/letta_cold_start.py), and [architecture guide](https://github.com/Justee21/AdaptKit/blob/main/docs/architecture.md). These are one-time cold-start reads, not continuous two-way memory synchronization. The application must not convert arbitrary memory text into trusted numeric priors.

## Implicit conversational feedback

An evaluator classifies the newest user message into a target and selected-action-relative sentiment:

- `behavior` with `positive` or `negative` may update the selected action.
- `answer_content`, `task_continuation`, `quoted_or_meta`, and `unrelated` use `none` and never update the policy.

```python
from adaptkit import LLMFeedbackExtractor, Profile

def judge(messages):
    # Call any provider and return validated structured output.
    return {"target": "behavior", "sentiment": "negative", "confidence": 0.95}

profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    evaluator=LLMFeedbackExtractor(judge=judge),
)

decision = profile.choose("debugging")
result = profile.observe(
    decision=decision,
    idempotency_key="conversation-turn-18",
    previous_prompt=user_prompt,
    previous_response=agent_response,
    user_message=next_user_message,
)
```

Sentiment is always about the action that produced the prior response. If `explanation_first` was selected and the user politely asks to see the patch first, that is negative evidence for `explanation_first`, not positive evidence for the requested replacement.

The default implicit confidence threshold is \(0.70\). Evaluator failures, malformed output, weak signals, and non-behavior targets fail closed without learning. `aobserve()` provides the equivalent asynchronous path.

`LLMFeedbackExtractor` always constructs `source="implicit"` internally. A legacy judge result may include `source="implicit"`, but any attempt to claim `explicit`, `passive`, or another trust source is rejected and cannot learn. Application-owned `FeedbackExtractor` implementations may deliberately construct other valid event sources when the application—not an LLM—owns that trust decision.

## Idempotency, learning modes, and direct feedback

One decision may receive multiple distinct signals, such as implicit feedback followed by a thumbs-up. The invariant is:

\[
\boxed{\text{one valid }(\text{decision\_id},\text{idempotency\_key})\rightarrow\text{at most one policy update}}
\]

```python
profile.like(decision, idempotency_key="thumb-up-18")
profile.dislike(decision, idempotency_key="thumb-down-18")

profile.prefer(
    "debugging",
    preferred="patch_first",
    rejected="explanation_first",
    idempotency_key="settings-save-91",
)
```

`like()`, `dislike()`, and `observe()` accept `apply=False` for a one-call shadow override. A profile-level mode is clearer for a rollout:

```python
profile = Profile(..., learning_mode="shadow")
profile.set_learning_mode("active")
```

Shadow observations are recorded but do not mutate the policy. `apply=True` can explicitly override shadow mode. Pairwise `prefer()` follows the same profile mode and accepts the same override.

Exact retries of `like()`, `dislike()`, `signal()`, and `prefer()` deduplicate. Reusing a key with a different direct sentiment, passive payload, learning mode, or preferred/rejected pair raises `IdempotencyConflictError`. `observe()` is the documented exception: once a `(decision_id, idempotency_key)` exists, a replay returns the stored result without evaluating or comparing conversation text. This privacy-first behavior avoids retaining prompts, responses, user messages, or hashes of them; applications must bind each conversational event to one stable key.

## Passive application signals

Applications can deliberately translate product telemetry into selected-action-relative feedback:

```python
result = profile.signal(
    decision,
    signal_type="ui.regeneration",
    sentiment="negative",
    confidence=0.92,
    metadata={"attempt": 2, "surface": "answer"},
    idempotency_key="regeneration-event-123",
)
```

`signal_type` is an open lowercase namespaced identifier rather than a closed enum, allowing values such as `ui.regeneration`, `agent.tool_approval`, or `task.completion`. Acceptance, completion, dwell time, approval, and abandonment are not inherently behavior preferences: the application must intentionally map each signal to positive or negative evidence for the selected action.

Passive confidence is gated by `passive_confidence_threshold`, which defaults to the implicit threshold. Accepted updates remain binary. AdaptKit persists `source="passive"`, signal type, confidence, and JSON metadata. Metadata is limited to \(8\,\mathrm{KiB}\), four levels of nesting, and finite numbers. Do not include prompts, responses, secrets, or sensitive user data. Reusing an idempotency key with the same payload returns `duplicate`; reusing it with a different payload raises `IdempotencyConflictError`.

## SQLite persistence

```python
from adaptkit import Profile, SQLiteStore

store = SQLiteStore("adaptkit.db", busy_timeout=1.0)
profile = Profile(user_id="user-123", actions=["a", "b"], store=store)
```

`SQLiteStore` is designed for a single host. Observation insertion and its posterior update occur in one transaction, so a crash cannot leave a partial learning update. Threads and local processes can share a database; lock waits default to one second and then raise `StorageBusyError`.

Storage schema version \(2\) performs one transactional upgrade from schema \(1\); other unsupported versions are rejected. The order-independent action-set fingerprint is bound on first use. Reopening the same user/context with added, removed, or renamed actions raises `ActionSetMismatchError` instead of silently introducing new priors. A migrated schema-\(1\) policy binds to the first compatible complete action set supplied after upgrade; its previously observed action keys must be a subset.

`InMemoryStore` remains the zero-configuration default and provides atomic thread-level updates, but it does not survive a restart or coordinate multiple processes.

`SQLiteStore` requires a real file path; use `InMemoryStore` instead of SQLite's `:memory:` pseudopath. `delete_user()` removes live rows transactionally, but SQLite files, filesystem snapshots, and backups may retain recoverable historical bytes. Applications needing physical erasure must manage database compaction and backup retention outside AdaptKit.

For account controls:

```python
structured_export = profile.export_user()
profile.delete_user()
```

## OpenAI Responses API judge

This integration uses the current OpenAI Python SDK, Responses API Structured Outputs, and `store=False`. OpenAI remains an optional example dependency rather than a core AdaptKit dependency.

```python
import math
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from adaptkit import LLMFeedbackExtractor, Profile


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal[
        "behavior", "answer_content", "task_continuation", "quoted_or_meta", "unrelated"
    ]
    sentiment: Literal["positive", "negative", "none"]
    confidence: float = Field(ge=0, le=1)


load_dotenv(Path(".env"))
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_retries=0,
    timeout=60.0,
)
model = os.environ["OPENAI_JUDGE_MODEL"]
threshold = float(os.environ["ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD"])
if not math.isfinite(threshold) or not 0 < threshold <= 1:
    raise RuntimeError("ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD must be within (0, 1]")


def judge(messages):
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=Judgment,
        store=False,
        max_output_tokens=800,
        reasoning={"effort": "low"},
    )
    if response.output_parsed is None:
        raise RuntimeError("judge returned no parsed output")
    result = response.output_parsed
    if (result.target == "behavior") != (result.sentiment != "none"):
        raise RuntimeError("inconsistent target and sentiment")
    return result.model_dump()


profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    implicit_confidence_threshold=threshold,
    evaluator=LLMFeedbackExtractor(
        judge=judge,
        action_descriptions={
            "patch_first": "Show the patch before the explanation.",
            "explanation_first": "Explain before showing the patch.",
        },
    ),
)
```

Pass `action_descriptions` whenever action keys alone do not completely describe their behavior. The descriptions are included in the transient evaluator request so sentiment can remain relative to the selected policy action; they are not persisted by AdaptKit.

See the runnable [`examples/openai_judge.py`](https://github.com/Justee21/AdaptKit/blob/main/examples/openai_judge.py).

For a callback with a completely different request format—or a local model with no SDK—see [`examples/generic_provider.py`](https://github.com/Justee21/AdaptKit/blob/main/examples/generic_provider.py). It shows the same structured contract and maps `Decision.action` independently to a model instruction, workflow branch, and tool allowlist.

## Try personalization end to end

The interactive playground isolates one preference dimension: `code_first` versus `explanation_first`. Both actions request comparable content and brevity; only the response ordering changes. The playground shows every selected action, mapped instruction, policy version, and Beta posterior.

```bash
python -m examples.interactive_personalization
```

Offline mode is deterministic and free. Try a prompt, then respond with “Please put the code first,” inspect `/state`, add `/feedback up`, switch `/mode shadow`, or use `/delete-user`. State persists in the ignored `.adaptkit-playground.db` file.

Real mode uses the models configured in `.env` and incurs API cost:

```bash
python -m examples.interactive_personalization --real
```

The first turn makes one generation request. Each later conversational turn judges the prior interaction and then generates a response, so it normally makes two requests. Both OpenAI calls use `store=False`.

The playground keeps up to four user/assistant turns in memory so follow-ups such as “re-explain that” retain conversational context. This text is sent to the configured agent model but is not written to AdaptKit's SQLite store. Type `/quit` before entering the launch command again; launch commands belong at the normal shell prompt, not inside the playground prompt.

Offline playground mode uses a deterministic \(0.90\) threshold. Real mode requires `ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD`; it never silently applies the Terra value to another model. AdaptKit's provider-independent `Profile` default remains \(0.70\), but production integrators must benchmark and explicitly configure thresholds for their own evaluator, prompt, model, and traffic.

### Expanded workflow dogfood

The `workflow` scenario more closely resembles a real coding assistant. It learns among four mutually exclusive presentation strategies: a concise direct solution, guided learning, a detailed deep dive, or examples first. Each chosen action is mapped by the application to a complete system instruction and visibly changes the response; the action key itself is not sent as an unexplained label.

```bash
python -m examples.interactive_personalization \
  --real \
  --scenario workflow \
  --user-id workflow-v1 \
  --context algorithms
```

Respond naturally after each answer—for example, “I learn better when you guide me through the reasoning before the complete answer,” or simply continue when the presentation was neutral. Use a new user ID when changing scenarios because the persisted policy's action set must remain stable. Switch contexts with `/context debugging` or `/context architecture` to see independent preferences for the same user.

With \(K=4\) actions, Thompson Sampling explores more alternatives and normally needs more observations than the focused \(K=2\) ordering test. The headings make adherence easy to inspect, but real personalization should be judged over several turns rather than a single selection.

## Privacy boundaries

AdaptKit's built-in stores persist only structured learning data: user IDs, contexts, action keys and fingerprints, policy versions, Beta parameters, decision-selection metadata, idempotency keys, classification labels, and application-supplied passive-signal metadata. They do not persist prompts, model responses, follow-up text, evaluator reasons, system instructions, or API keys. Conversation text exists in application memory while an evaluator call is made, and your own logs or surrounding application may store it.

Provider privacy is a separate boundary. A remote judge receives the interaction text you send. In the OpenAI examples, `store=False` disables Responses API application-state storage for those requests, but it is not a promise of zero provider retention; abuse-monitoring retention and organization-specific controls are governed by OpenAI's current data policy. Review the [Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) and [OpenAI API data controls](https://platform.openai.com/docs/guides/your-data) before sending sensitive content.

## Benchmarks

Run the deterministic offline checks:

```bash
PYTHONPATH=. python benchmarks/bandit_benchmark.py
PYTHONPATH=. python benchmarks/evaluator_benchmark.py
```

The bandit simulation checks whether Thompson Sampling learns a synthetic preference. The deterministic evaluator uses hand-written keyword rules over 28 labeled examples. It is a pipeline smoke test for schema validation and metric calculation—not evidence of real LLM accuracy.

The feedback benchmark evaluates \(120\) hand-authored labeled cases, and the initial-prompt benchmark evaluates \(80\). Each has a fixed, disjoint calibration/holdout split with a cross-split near-duplicate guard. Three deployment-shaped runs use calibration data only to freeze a threshold from \(\{0.70,0.80,0.90,0.95\}\); release gates are then applied independently to every run on holdout data at that frozen threshold.

```bash
PYTHONPATH=. python benchmarks/openai_evaluator_benchmark.py \
  --runs 3 --batch-size 1 --workers 8
PYTHONPATH=. python benchmarks/openai_prompt_routing_benchmark.py \
  --runs 3 --workers 8
```

Both summaries report per-run metrics, ranges, Wilson \(95\%\) confidence intervals, failure IDs, token usage, and whether every holdout run passed. Committed artifact checks bind the dataset, classifier prompt, gates, and README claims. Gates are point-estimate safety checks at the frozen threshold; even zero harmful errors on a small holdout is not strong statistical evidence by itself. These nondeterministic paid benchmarks remain outside CI and never write request text into artifacts.

The latest checked `gpt-5.6-terra` run used three independent runs of each dataset and passed every holdout gate:

| Benchmark | Data per run | Frozen threshold | Mean holdout result | Safety result |
| --- | --- | ---: | --- | --- |
| Feedback judge | \(40\) calibration + \(80\) holdout | \(0.95\) | \(95.42\%\) detection, \(99.17\%\) direction, \(8.33\%\) raw false positives | \(100\%\) applied precision, \(0\) harmful updates, \(0\) invalid outputs |
| Initial-prompt router | \(20\) calibration + \(60\) holdout | \(0.70\) | \(97.78\%\) cue routing, \(100\%\) routed precision | \(0\) wrong-action overrides, \(0\) no-cue overrides, \(0\) invalid outputs |

The raw feedback false-positive rate includes low-confidence classifications blocked by the frozen learning threshold. Full per-run metrics and confidence intervals are in [`artifacts/real_evaluator_summary.json`](https://github.com/Justee21/AdaptKit/blob/main/artifacts/real_evaluator_summary.json) and [`artifacts/real_prompt_router_summary.json`](https://github.com/Justee21/AdaptKit/blob/main/artifacts/real_prompt_router_summary.json).

For representative application traffic, keep learning in shadow mode, label structured outcomes outside AdaptKit, and run:

```bash
PYTHONPATH=. python benchmarks/shadow_traffic_report.py \
  --input local-structured-labels.jsonl \
  --output local-shadow-summary.json \
  --threshold 0.90
```

The reporter rejects unknown fields so raw prompts or responses cannot accidentally enter its artifact. Synthetic benchmarks are release checks, not production-accuracy claims.

## Production integration checklist

- Keep action keys, user IDs, and context keys stable across deployments.
- Map every action to behavior that is materially visible in the final output.
- Use durable, unique event IDs as idempotency keys; do not generate a new key when retrying the same event.
- Start implicit learning in `shadow` mode and inspect behavior precision and false-direction updates on representative data.
- Treat evaluator and `StorageBusyError` outcomes as expected operational states and monitor structured lifecycle events with `event_hook`.
- Set `max_decision_age` when delayed feedback attribution would be unsafe.
- Test your provider's timeout, retry, retention, and rate-limit behavior independently of AdaptKit.

## Development

```bash
python -m pip install -e '.[dev,evaluation]'
python -m pytest
mypy adaptkit examples benchmarks tests
pyright adaptkit examples benchmarks
python scripts/release_build.py --check
```

`scripts/release_build.py` derives the version from `pyproject.toml`, validates and removes only repository-local AdaptKit build outputs, and verifies that exactly one canonical wheel and one source distribution exist. It rejects unexpected archive members, compares both archives with the current tree, and excludes private, credential, cache, database, editor, bytecode, and raw JSONL files. `--check` also runs Twine and installs both distributions separately outside the source tree. Upload only the two reviewed filenames it reports; never publish with `dist/*`.

The normal test suite makes no provider or model requests and never requires an API key. Hosted CI installs declared dependencies from the package index, but its test and verification steps do not call provider APIs.

## 0.1 boundaries

- One mutually exclusive action set per profile.
- Discrete contexts with no embeddings, cross-context generalization, or preference decay.
- Single-host SQLite, not distributed coordination.
- Immediate feedback attribution to a persisted decision; no long-horizon rewards.
- No provider adapters in the core package and no automatic prompt rewriting.
- Schema version \(2\) supports only the targeted schema-\(1\) upgrade; there is no general migration framework.
- A migrated schema-\(1\) pairwise-operation key has no stored operation identity; its first schema-\(2\) replay binds that identity, after which conflicting reuse is rejected.

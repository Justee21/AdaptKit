# AdaptKit

AdaptKit is a compact, provider-independent Python SDK for per-user, per-context Thompson Sampling. It learns which mutually exclusive agent behavior works best for an individual user in a named context.

```text
choose action → map action to an instruction → agent responds → observe feedback → update policy
```

AdaptKit chooses an action key. Your application decides what that key means and sends the mapped instruction to its model. AdaptKit does not own the agent, rewrite prompts, or automatically send `Decision.action` to an LLM.

## Installation

AdaptKit is not published to PyPI yet. Install the repository locally:

```bash
git clone https://github.com/Justee21/AdaptKit.git
cd AdaptKit
python -m pip install -e .
```

For the optional OpenAI examples and evaluator benchmark:

```bash
python -m pip install -e '.[evaluation]'
cp .env.example .env
```

Set these values in the ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_JUDGE_MODEL=your-structured-output-capable-model
OPENAI_AGENT_MODEL=your-agent-model
```

`OPENAI_AGENT_MODEL` is optional and falls back to `OPENAI_JUDGE_MODEL`. Never commit `.env`; only `.env.example` belongs in Git.

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

Each `(user, context, action)` starts with a posterior of \(\operatorname{Beta}(1,1)\). Accepted positive feedback increments \(\alpha\) by \(1\); accepted negative feedback increments \(\beta\) by \(1\). Confidence gates implicit signals but does not fractionalize the update in V1.

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

## SQLite persistence

```python
from adaptkit import Profile, SQLiteStore

store = SQLiteStore("adaptkit.db", busy_timeout=1.0)
profile = Profile(user_id="user-123", actions=["a", "b"], store=store)
```

`SQLiteStore` is designed for a single host. Observation insertion and its posterior update occur in one transaction, so a crash cannot leave a partial learning update. Threads and local processes can share a database; lock waits default to one second and then raise `StorageBusyError`. The schema has an explicit version and unsupported versions are rejected instead of guessed at.

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
    return {**result.model_dump(), "source": "implicit"}


profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
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

See the runnable [`examples/openai_judge.py`](examples/openai_judge.py).

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

The playground uses an implicit-learning threshold of (0.90), calibrated conservatively for its Terra dogfood scenario. AdaptKit's provider-independent `Profile` default remains (0.70); production integrators must benchmark and choose a threshold for their own evaluator, prompt, model, and traffic.

## Privacy boundaries

AdaptKit's built-in stores persist only structured learning data: user IDs, contexts, action keys, policy versions, Beta parameters, decision metadata, idempotency keys, and classification labels. They do not persist prompts, model responses, follow-up text, evaluator reasons, system instructions, or API keys. Conversation text exists in application memory while an evaluator call is made, and your own logs or surrounding application may store it.

Provider privacy is a separate boundary. A remote judge receives the interaction text you send. In the OpenAI examples, `store=False` disables Responses API application-state storage for those requests, but it is not a promise of zero provider retention; abuse-monitoring retention and organization-specific controls are governed by OpenAI's current data policy. Review the [Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) and [OpenAI API data controls](https://platform.openai.com/docs/guides/your-data) before sending sensitive content.

## Benchmarks

Run the deterministic offline checks:

```bash
PYTHONPATH=. python benchmarks/bandit_benchmark.py
PYTHONPATH=. python benchmarks/evaluator_benchmark.py
```

The bandit simulation checks whether Thompson Sampling learns a synthetic preference. The deterministic evaluator uses hand-written keyword rules over 28 labeled examples. It is a pipeline smoke test for schema validation and metric calculation—not evidence of real LLM accuracy.

The opt-in real benchmark evaluates 28 base and 18 adversarial examples covering paraphrases, indirect feedback, quoted feedback, ambiguous continuation, polite corrections, praise followed by correction, historical references, and prompt injection:

```bash
PYTHONPATH=. python benchmarks/openai_evaluator_benchmark.py --runs 1 --batch-size 1
```

`--runs` is bounded to \(1\)–\(5\) and `--batch-size` to \(1\)–\(46\). Batch size \(1\) makes \(46\) paid calls but most closely matches production; the cheaper default of \(10\) makes five calls per run and is useful as a pipeline diagnostic. The separate `artifacts/real_evaluator_summary.json` reports the model, example count, detection accuracy, behavior precision/recall, direction accuracy, false-direction update rate, false-positive rate, invalid outputs, category metrics, and token usage. Its diagnostics contain labels and example IDs only, never request text. This nondeterministic benchmark is intentionally excluded from CI.

The recorded deployment-shaped `gpt-5-nano-2025-08-07` run produced:

| Metric | Result |
| --- | ---: |
| Examples | \(46\) |
| Feedback-detection accuracy | \(93.5\%\) |
| Behavior precision | \(91.7\%\) |
| Behavior recall | \(100.0\%\) |
| Direction accuracy | \(100.0\%\) |
| False-direction update rate | \(0.0\%\) |
| Raw false-positive rate | \(8.3\%\) |
| False-positive learning rate at \(0.70\) | \(0.0\%\) |
| Invalid outputs | \(1\) |
| Requests | \(46\) |
| Total tokens | \(34{,}322\) |

The two false behavioral classifications had confidences \(0.62\) and \(0.65\), so neither passes the default \(0.70\) learning threshold. The invalid output also fails closed. Fine-grained target accuracy was \(69.6\%\), but most of those errors were disagreements among non-behavior classes and could not change the learner. This small, prompt-tuned diagnostic set is not a production accuracy claim; begin with shadow learning on representative traffic.

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
python -m build
python -m twine check dist/*
```

Normal tests and CI are offline and never require an API key.

## V1 boundaries

- One mutually exclusive action set per profile.
- Discrete contexts with no embeddings, cross-context generalization, or preference decay.
- Single-host SQLite, not distributed coordination.
- Immediate feedback attribution to a persisted decision; no long-horizon rewards.
- No provider adapters and no automatic prompt rewriting.
- Schema version \(1\) rejects unsupported databases; there is no general migration framework yet.

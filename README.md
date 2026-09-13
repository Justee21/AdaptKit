# AdaptKit

AdaptKit is a small, provider-independent Python SDK for per-user, per-context Thompson Sampling. It learns which agent behavior an individual user prefers in a given context. An LLM can interpret conversational feedback, while the bandit alone updates the policy.

```text
choose behavior → agent responds → user reacts → evaluator extracts evidence → bandit updates
```

## Installation

Install the provider-independent core:

```bash
python -m pip install -e .
```

Install the optional OpenAI evaluation/example dependencies:

```bash
python -m pip install -e '.[evaluation]'
```

Copy the placeholder environment file and fill in your own values:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_JUDGE_MODEL=your-structured-output-capable-model
```

`.env` is ignored by Git. Do not commit it.

## Core usage

```python
from adaptkit import LLMFeedbackExtractor, Profile

def judge(messages):
    # Call any provider here and return its validated structured output.
    return {"target": "behavior", "sentiment": "negative", "confidence": 0.95}

profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    evaluator=LLMFeedbackExtractor(judge=judge),
)

strategy = profile.choose("debugging")
response = agent.run(prompt=user_prompt, strategy=strategy)

result = profile.observe(
    context="debugging",
    action=strategy,
    previous_prompt=user_prompt,
    previous_response=response,
    user_message=next_user_message,
)
```

Each `(user, context, action)` has its own Thompson Sampling posterior beginning at \(\operatorname{Beta}(1,1)\). Explicit feedback bypasses the evaluator:

```python
profile.like("debugging", "patch_first")
profile.dislike("debugging", "explanation_first")
profile.prefer("debugging", preferred="patch_first", rejected="explanation_first")
```

The evaluator separates a message's `target` from its `sentiment`. Targets are `behavior`, `answer_content`, `task_continuation`, `quoted_or_meta`, and `unrelated`. Only `behavior` may have `positive` or `negative` sentiment; every other target uses `none` and leaves the policy unchanged.

Behavioral feedback maps to a binary reward \(r \in \{-1,+1\}\). AdaptKit computes effective evidence \(e = r \times \text{confidence}\). If \(|e|\) reaches the configurable threshold (default \(0.35\)), only the observed action receives a success or failure. Weak evidence and non-behavior targets leave the policy unchanged.

`observe()` supports a synchronous judge. `aobserve()` supports an asynchronous judge, or runs a synchronous judge in a worker thread. Both use the same validated update path. Invalid output and evaluator failures return `evaluator_error` without changing the policy.

## OpenAI Responses API judge

This complete example uses the OpenAI Python SDK, Responses API, Pydantic Structured Outputs, and `store=False`. The SDK remains provider-independent because these imports live in an optional example.

```python
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from adaptkit import LLMFeedbackExtractor, Profile


class OpenAIJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal[
        "behavior", "answer_content", "task_continuation", "quoted_or_meta", "unrelated"
    ]
    sentiment: Literal["positive", "negative", "none"]
    confidence: float = Field(ge=0, le=1)
    reason: str | None


load_dotenv(Path(__file__).resolve().parent / ".env")
model = os.environ["OPENAI_JUDGE_MODEL"]
client = OpenAI(max_retries=0, timeout=60.0)


def judge(messages):
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=OpenAIJudgment,
        store=False,
        max_output_tokens=800,
        reasoning={"effort": "low"},
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI response contained no parsed event")
    judgment = response.output_parsed
    if (judgment.target == "behavior") != (judgment.sentiment != "none"):
        raise RuntimeError("OpenAI response contained an inconsistent target and sentiment")
    return {**judgment.model_dump(), "source": "implicit"}


profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    evaluator=LLMFeedbackExtractor(judge=judge),
)
```

The runnable version is [`examples/openai_judge.py`](examples/openai_judge.py).

## Privacy boundaries

AdaptKit's in-memory store retains only numeric learner state. It does not persist prompts, agent responses, user follow-ups, API keys, or evaluator reasoning. Your application still supplies conversation text to the configured judge and receives the returned event, so application logs and custom storage remain your responsibility.

A remote judge receives the interaction text. The OpenAI example sets `store=False`, which disables Responses API application-state storage for the request. This setting does not by itself guarantee zero provider retention: OpenAI's abuse-monitoring logs may retain API content for up to 30 days unless the organization has approved retention controls. Review the provider's current policy before sending sensitive data. See OpenAI's [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create) and [API data controls](https://platform.openai.com/docs/models/default-usage-policies-by-endpoint).

## Benchmarks

Run the deterministic offline benchmarks:

```bash
PYTHONPATH=. python benchmarks/bandit_benchmark.py
PYTHONPATH=. python benchmarks/evaluator_benchmark.py
```

The bandit benchmark compares Random and Thompson Sampling using average reward, cumulative reward, expected regret, and optimal-action selection rate.

The deterministic evaluator benchmark uses hand-written keyword rules over 28 labeled examples. It is a pipeline smoke test for schema validation and metric calculation. Its score is not evidence of real LLM evaluator accuracy.

To make exactly one bounded OpenAI request over the 28 base examples plus 10 adversarial examples, run:

```bash
PYTHONPATH=. python benchmarks/openai_evaluator_benchmark.py
```

The adversarial cases cover paraphrases, indirect feedback, quoted feedback, ambiguous continuation, and prompt-injection attempts. The summary contains the model, aggregate and category metrics, token usage, request count, example count, and label-only failure diagnostics. It never includes interaction text. It is saved separately to `artifacts/real_evaluator_summary.json`. This paid, nondeterministic benchmark is opt-in and is not part of CI.

The recorded `gpt-5-nano-2025-08-07` run produced:

| Metric | Result |
| --- | ---: |
| Examples | 38 |
| Feedback-detection accuracy | 97.4% |
| Target accuracy | 76.3% |
| Direction accuracy | 81.2% |
| False-positive rate | 0.0% |
| Invalid outputs | 0 |
| Input tokens | 2,545 |
| Output tokens | 2,235 |

This is a small diagnostic set, not a production accuracy claim. The target split eliminated false-positive learning updates in this run. Most fine-grained target errors were disagreements among non-behavior classes and therefore would not update the learner. The main remaining risk was preference correction: Nano missed one behavioral correction and labeled two requests for an alternative behavior as positive rather than negative toward the selected action. Applications should monitor evaluator quality on representative interactions and keep conservative update thresholds.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

CI and the normal test suite are offline and require no API key.

## MVP limitations

- Actions are one mutually exclusive set of strings.
- Contexts are discrete strings with no cross-context generalization or preference decay.
- State is in memory and is not retained across processes.
- Repeated observations are treated as new evidence; there is no interaction deduplication.
- Feedback is immediate and attributed to the supplied preceding interaction.
- Thread-level updates are atomic in `InMemoryStore`; coordination across processes is unsupported.
- Distributed learning, long-horizon rewards, and provider adapters are outside this MVP.

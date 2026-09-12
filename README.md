# AdaptKit

AdaptKit is a small, provider-independent Python SDK that learns which agent behavior an individual user prefers in a given context. An LLM interprets conversational feedback; a contextual bandit aggregates the resulting evidence. The evaluator never edits policy probabilities directly.

```text
choose behavior → agent responds → user reacts → evaluator extracts evidence → bandit updates
```

## Quickstart

```bash
python -m pip install -e .
```

```python
from adaptkit import LLMFeedbackExtractor, Profile

def judge(messages):
    # Call any provider here and return its validated structured output.
    return {"has_feedback": True, "reward": -0.9, "confidence": 0.95}

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

Conversation text is passed to the evaluator for that call and is not stored. `Profile.state()` contains only the bandit's numeric state.

## Learning behavior

Each `(user, context, action)` has an independent Thompson Sampling posterior beginning at \(\operatorname{Beta}(1,1)\). Explicit feedback bypasses the evaluator:

```python
profile.like("debugging", "patch_first")
profile.dislike("debugging", "explanation_first")
profile.prefer("debugging", preferred="patch_first", rejected="explanation_first")
```

For implicit feedback, AdaptKit computes \(e = \text{reward} \times \text{confidence}\). If \(e\) reaches the configurable threshold (default \(0.35\)), only the action actually observed receives a success or failure. Weak evidence and `has_feedback=False` leave the policy unchanged.

`observe()` supports a synchronous judge. `aobserve()` supports an asynchronous judge, or runs a synchronous judge in a worker thread. Both call the same validation and update code. Invalid output and evaluator failures return `evaluator_error` and leave the policy unchanged.

## Judge contract

`LLMFeedbackExtractor` accepts `judge(messages)` and/or `async_judge(messages)`. The callable returns a mapping:

```json
{
  "has_feedback": true,
  "reward": -0.8,
  "confidence": 0.9,
  "source": "implicit",
  "reason": "The user requested a more direct response."
}
```

Reward must be within `[-1, 1]`, confidence within `[0, 1]`, and no-feedback events omit reward. Configure provider timeouts in the judge callable. No provider package or API key is required by AdaptKit.

## Run the examples and benchmarks

```bash
PYTHONPATH=. python examples/explicit_feedback.py
PYTHONPATH=. python examples/implicit_feedback.py
PYTHONPATH=. python examples/async_feedback.py
PYTHONPATH=. python examples/coding_agent.py
PYTHONPATH=. python benchmarks/bandit_benchmark.py
PYTHONPATH=. python benchmarks/evaluator_benchmark.py
```

The seeded bandit benchmark compares Random and Thompson Sampling using average reward, cumulative reward, expected regret, and optimal-action selection rate. Its default run gives Thompson a success rate of `0.894` versus Random's `0.547`; generated SVGs are in `artifacts/`.

The default evaluator benchmark is explicitly an offline conservative-rule pipeline check over 28 labeled examples. It does not measure a real LLM. To evaluate a synchronous LLM judge, expose it as a callable and run:

```bash
PYTHONPATH=. python benchmarks/evaluator_benchmark.py --judge your_module:judge
```

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Tests are offline and require no API key.

## MVP limitations

- Actions are one mutually exclusive set of strings.
- Contexts are discrete strings with no cross-context generalization or preference decay.
- State is in memory and is not retained across processes.
- Repeated observations are treated as new evidence; there is no interaction deduplication.
- Feedback is immediate and attributed to the supplied preceding interaction.
- Distributed learning, long-horizon rewards, and provider adapters are outside this MVP.

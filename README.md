# AdaptKit
AdaptKit learns how the user wants the agent to behave, and iteratively improves based on their feedback. 
You define a few behaviors such as "explanation first", "concise responses", etc. AdaptKit only picks an action key. You decide what that key means, and your app sends the instruction to the model.

Under the hood it is per-user Thompson Sampling.

choose action → map to instruction → agent responds → observe feedback → update

This SDK doesn't wrap your agent or rewrite your prompts, and the built-in stores don't save prompts or responses.

## Why??
Many times users want different things/behaviors from the same agent. People are different and thus learn and work differently from one another, so I created an SDK to help an agent adapt to each user. 

## Install
 
```bash
pip install adaptkit
# with the OpenAI examples:
pip install 'adaptkit[evaluation]'
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
response = agent.run(prompt, system_instruction=instructions[decision.action])

 
profile.like(decision, idempotency_key="thumbs-up-456")
```

## How does it Learn?

Each (user, context, action) starts at $\mathrm{Beta}(1,1)$, and each piece of feedback is a binary reward $r \in \lbrace 0,1\rbrace$ for the chosen action.

$$\alpha \leftarrow \alpha + r, \qquad \beta \leftarrow \beta + (1-r)$$

In AdaptKit the contexts are independent - users can have different action preferences dependent on the context. Feedback is always about the chosen action.

## Feedback Sources
- **Explicit:** `like()`, `dislike()`, and pairwise `prefer()`.
- **Implicit:** `observe()` runs an LLM judge on the user's next message. Only comments about *behavior* update the policy. Comments about the answer's content, follow-up tasks, or unrelated messages are ignored, as are low-confidence judgments.
- **Passive:** `signal()` lets your app turn telemetry (regenerations, tool approvals) into positive or negative evidence.
- **Prompt cues:** an optional extractor can honor explicit requests like "walk me through it" on the first turn, before any history exists.

Every update takes an idempotency key, so retries never double-count.

Set `learning_mode="shadow"` to record feedback without changing the policy.

## Benchmarks

The LLM judge and prompt router were tested on hand-labeled datasets (120 and 80 cases) with fixed calibration/holdout splits, 3 runs each on `gpt-5.6-terra`:

- **Feedback judge:** the LLM classifier behind `observe()`. It reads the user's next message and decides whether it's feedback about the chosen behavior, and if so, whether it's positive or negative.
- **Prompt router:** the optional cue extractor. It spots explicit style requests in the user's prompt (like "walk me through it") and picks the matching action directly.
 
| | Threshold | Holdout result | Safety |
|---|---|---|---|
| Feedback judge | 0.95 | 95.4% detection, 99.2% direction | 0 harmful updates |
| Prompt router | 0.70 | 97.8% cue routing, 100% precision | 0 wrong overrides |
 
These are small synthetic sets, not production accuracy. Calibrate thresholds for your own model and traffic. Details are in [`artifacts/`](artifacts/) and [`benchmarks/`](benchmarks/).

## Try it
 
```bash
python -m examples.interactive_personalization          # offline, free
python -m examples.interactive_personalization --real   # uses OpenAI via .env
```
 
Tell it "please put the code first," then run `/state` to watch the posterior move.

## Limitations
 
- One action set per profile, discrete contexts, no generalization across contexts
- Binary rewards with immediate attribution only
- SQLite is single-host

More detail: [architecture guide](docs/architecture.md), [examples](examples/).

## Future Work
- Expand to allow for multiple action sets per profile
- Support a continuous reward function (instead of binary as is now)

MIT License

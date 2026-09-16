# Architecture

AdaptKit owns structured personalization state, not the model call or prompt text.

```text
durable external memory ──app validation──> per-action Beta priors
current user prompt ──transient router────> optional action override
stored user/context policy ───────────────> sampled action
sampled/overridden action ──app mapping───> system instruction ──> model response
explicit/implicit/passive outcome ────────> atomic policy update
```

## Boundaries

Memory retrieval and policy learning solve different problems. Retrieval supplies stored facts or text to a model. AdaptKit stores a Beta posterior over application-defined action keys and learns from selected-action-relative outcomes. External memory can seed a modest prior, but it does not replace subsequent policy evidence.

External memory providers may return raw text, but the application must map a validated durable preference to an action key. AdaptKit accepts only `Mapping[str, BetaPrior]`; it neither trusts arbitrary memory text as numeric state nor stores that text. A turn-specific request such as “for this answer, be concise” belongs in transient prompt routing. A durable statement such as “I generally prefer concise answers” may be stored by the application's chosen memory provider. The Mem0 and Letta examples perform a cold-start read only: they do not continuously synchronize AdaptKit state back to memory, and contradictory or ambiguous memory maps to neutral priors.

Prompt routing sees a nonempty current prompt transiently. It can override policy selection for one decision, but does not update the posterior. Empty prompts are treated as no cue without calling an evaluator; invalid non-string inputs raise before evaluator fallback. The application then maps the selected action key to a concrete instruction; AdaptKit does not rewrite prompts or call the response model.

Feedback is attributed to the action recorded by an immutable decision. LLM judge output can only become implicit feedback; the judge cannot claim an explicit or passive trust source. One observation insert and its posterior update occur atomically. Exact idempotent retries deduplicate; conflicting operation identities fail. Conversational `observe()` replays remain privacy-first because AdaptKit does not retain prompt text or hashes of it.

The built-in Thompson learner is Beta–Bernoulli. An accepted reward is \(r\in\{0,1\}\), with \(\alpha\leftarrow\alpha+r\) and \(\beta\leftarrow\beta+(1-r)\). Confidence gates whether implicit or passive evidence is accepted; it never becomes a fractional reward. Continuous, delayed, multi-step, or feature-based rewards require a different future learner and are outside 0.1.0.

Built-in persistence stores structured identifiers, action keys, posterior parameters, versions, classification labels, and explicitly supplied passive metadata. It excludes prompts, responses, evaluator reasoning, instructions, and provider credentials.

Public alpha: this package is release-tested and suitable for evaluation, not production-validated. Representative application traffic must pass deployment-specific shadow-mode gates before active implicit learning is enabled.

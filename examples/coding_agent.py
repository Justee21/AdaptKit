import json

from adaptkit import LLMFeedbackExtractor, Profile


def judge(messages):
    """A deterministic stand-in for an LLM judge so the example runs offline."""
    interaction = json.loads(messages[-1]["content"].split("\n", 1)[1])
    latest = interaction["latest_user_message"].lower()
    if "show me the fix" in latest:
        return {"target": "behavior", "sentiment": "negative", "confidence": 0.95}
    if "exactly" in latest or "thanks" in latest:
        return {"target": "behavior", "sentiment": "positive", "confidence": 0.9}
    return {"target": "task_continuation", "sentiment": "none"}


def coding_agent(prompt: str, strategy: str) -> str:
    if strategy == "patch_first":
        return "Patch: change `items[index]` to `items[index - 1]`. Explanation follows."
    return "The indexing invariant is off by one. Change `items[index]` to `items[index - 1]`."


profile = Profile(
    user_id="developer-42",
    actions=["patch_first", "explanation_first"],
    evaluator=LLMFeedbackExtractor(judge=judge),
    seed=4,
)

prompt = "Why does this loop fail?"
decision = profile.choose("debugging")
response = coding_agent(prompt, decision.action)
print(f"[{decision.action}] {response}")

# In a real agent this arrives on the next conversational turn.
reaction = "Just show me the fix."
result = profile.observe(
    decision=decision,
    idempotency_key="turn-1-implicit",
    previous_prompt=prompt,
    previous_response=response,
    user_message=reaction,
)
print("learning result:", result.status.value)
print("policy state:", profile.state())

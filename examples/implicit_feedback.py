import json

from adaptkit import LLMFeedbackExtractor, Profile


def demo_judge(messages):
    """Replace this deterministic demo with your provider's structured-output call."""
    interaction = json.loads(messages[-1]["content"].split("\n", 1)[1])
    if "just show me" in interaction["latest_user_message"].lower():
        return {
            "target": "behavior",
            "sentiment": "negative",
            "confidence": 0.95,
            "reason": "The user requested a more direct response.",
        }
    return {"target": "task_continuation", "sentiment": "none"}


profile = Profile(
    user_id="user-123",
    actions=["patch_first", "explanation_first"],
    evaluator=LLMFeedbackExtractor(judge=demo_judge),
)
result = profile.observe(
    context="debugging",
    action="explanation_first",
    previous_prompt="Why does this recursion fail?",
    previous_response="Here is a detailed explanation, followed by the patch...",
    user_message="Just show me the line I need to change.",
)
print("observation:", result)
print("state:", profile.state())

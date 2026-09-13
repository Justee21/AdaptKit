import asyncio

from adaptkit import LLMFeedbackExtractor, Profile


async def demo_async_judge(messages):
    """Replace this coroutine with an async structured-output provider call."""
    await asyncio.sleep(0)
    return {"target": "behavior", "sentiment": "positive", "confidence": 0.9}


async def main():
    profile = Profile(
        user_id="user-123",
        actions=["concise", "detailed"],
        evaluator=LLMFeedbackExtractor(async_judge=demo_async_judge),
    )
    result = await profile.aobserve(
        context="architecture",
        action="detailed",
        previous_prompt="Explain the design.",
        previous_response="Here is a detailed walkthrough...",
        user_message="Thanks, that explanation was helpful.",
    )
    print("observation:", result)
    print("state:", profile.state())


asyncio.run(main())

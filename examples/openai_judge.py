from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from adaptkit import LLMFeedbackExtractor, Profile
from examples.configuration import required_confidence_threshold


class OpenAIJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal[
        "behavior", "answer_content", "task_continuation", "quoted_or_meta", "unrelated"
    ] = Field(description="What the latest user message is primarily doing.")
    sentiment: Literal["positive", "negative", "none"] = Field(
        description=(
            "Sentiment toward the selected action, not toward a requested replacement."
        )
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence that target and sentiment are both correct."
    )
    reason: str | None


def build_openai_judge() -> tuple[Callable[[list[dict[str, str]]], dict[str, Any]], str]:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    model = os.getenv("OPENAI_JUDGE_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not model or model.startswith("insert-"):
        raise RuntimeError("set OPENAI_JUDGE_MODEL in .env")
    if not api_key or api_key.startswith("insert-"):
        raise RuntimeError("set OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)

    def judge(messages: list[dict[str, str]]) -> dict[str, Any]:
        response = client.responses.parse(
            model=model,
            input=cast(Any, messages),
            text_format=OpenAIJudgment,
            store=False,
            max_output_tokens=800,
            reasoning={"effort": "low"},
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI response did not contain a parsed preference event")
        judgment = response.output_parsed
        if (judgment.target == "behavior") != (judgment.sentiment != "none"):
            raise RuntimeError("OpenAI response contained an inconsistent target and sentiment")
        return judgment.model_dump()

    return judge, model


def main() -> None:
    judge, model = build_openai_judge()
    threshold = required_confidence_threshold(
        "ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD"
    )
    profile = Profile(
        user_id="user-123",
        actions=["patch_first", "explanation_first"],
        evaluator=LLMFeedbackExtractor(judge=judge),
        implicit_confidence_threshold=threshold,
    )
    decision = profile.choose("debugging")
    result = profile.observe(
        decision=decision,
        idempotency_key="turn-1-implicit",
        previous_prompt="Why does this function fail?",
        previous_response="Here is a detailed explanation followed by the change.",
        user_message="Can you lead with the fix next time?",
    )
    print(f"model={model} status={result.status.value} updated={result.updated}")


if __name__ == "__main__":
    main()

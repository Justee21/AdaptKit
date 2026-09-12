from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptkit import LLMFeedbackExtractor, Profile


class OpenAIPreferenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_feedback: bool
    reward: float | None
    confidence: float = Field(ge=0, le=1)
    source: Literal["implicit"]
    reason: str | None

    @model_validator(mode="after")
    def validate_reward(self) -> "OpenAIPreferenceEvent":
        if self.has_feedback and self.reward is None:
            raise ValueError("feedback requires a reward")
        if not self.has_feedback and self.reward is not None:
            raise ValueError("no-feedback events must use a null reward")
        if self.reward is not None and not -1 <= self.reward <= 1:
            raise ValueError("reward must be within [-1, 1]")
        return self


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
            text_format=OpenAIPreferenceEvent,
            store=False,
            max_output_tokens=500,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI response did not contain a parsed preference event")
        return response.output_parsed.model_dump()

    return judge, model


def main() -> None:
    judge, model = build_openai_judge()
    profile = Profile(
        user_id="user-123",
        actions=["patch_first", "explanation_first"],
        evaluator=LLMFeedbackExtractor(judge=judge),
    )
    result = profile.observe(
        context="debugging",
        action="explanation_first",
        previous_prompt="Why does this function fail?",
        previous_response="Here is a detailed explanation followed by the change.",
        user_message="Can you lead with the fix next time?",
    )
    print(f"model={model} status={result.status.value} updated={result.updated}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from adaptkit import LLMPromptPreferenceExtractor, Profile
from examples.configuration import required_confidence_threshold


class OpenAIPromptPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str | None = Field(
        description="An available action key, or null when the prompt has no clear cue."
    )
    confidence: float = Field(ge=0, le=1)
    reason: str | None = Field(
        description="A short explanation used transiently for debugging."
    )


def build_openai_prompt_judge() -> tuple[
    Callable[[list[dict[str, str]]], dict[str, Any]], str
]:
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
            text_format=OpenAIPromptPreference,
            store=False,
            max_output_tokens=800,
            reasoning={"effort": "low"},
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI response did not contain a parsed prompt preference")
        return response.output_parsed.model_dump()

    return judge, model


def main() -> None:
    actions = {
        "direct_solution": "Give the complete solution immediately and concisely.",
        "guided_learning": "Teach with a few hints before giving the complete solution.",
        "examples_first": "Begin with a concrete worked example.",
    }
    judge, model = build_openai_prompt_judge()
    threshold = required_confidence_threshold(
        "ADAPTKIT_PROMPT_CONFIDENCE_THRESHOLD"
    )
    profile = Profile(
        user_id="user-123",
        actions=tuple(actions),
        prompt_evaluator=LLMPromptPreferenceExtractor(
            judge=judge,
            action_descriptions=actions,
        ),
        prompt_confidence_threshold=threshold,
    )
    prompt = "Guide me with a few hints before showing the complete answer."
    decision = profile.choose("algorithms", prompt=prompt)
    print(
        f"model={model} action={decision.action} "
        f"selection_source={decision.selection_source.value}"
    )


if __name__ == "__main__":
    main()

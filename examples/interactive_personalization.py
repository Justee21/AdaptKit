from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from adaptkit import LLMFeedbackExtractor, Profile, SQLiteStore

ACTION_INSTRUCTIONS = {
    "code_first": "Present complete code first, then explain the approach briefly.",
    "explanation_first": "Explain the approach briefly, then present complete code.",
}
BASE_INSTRUCTIONS = (
    "You are a concise coding assistant. Answer the current request directly. "
    "Include comparable content and use similar brevity regardless of the selected "
    "behavior; only the ordering of code and explanation should change."
)

Generator = Callable[[str, str], tuple[str, dict[str, int | None]]]


def offline_judge(messages: list[dict[str, str]]) -> dict[str, Any]:
    interaction = json.loads(messages[-1]["content"].split("\n", 1)[1])
    message = interaction["latest_user_message"].lower()
    selected = interaction["selected_action"]
    code_phrases = ("fix first", "patch first", "code first", "too much explanation")
    explanation_phrases = ("explain first", "reasoning first", "too terse", "more detail")
    if any(phrase in message for phrase in code_phrases):
        sentiment = "positive" if selected == "code_first" else "negative"
        return {"target": "behavior", "sentiment": sentiment, "confidence": 0.95}
    if any(phrase in message for phrase in explanation_phrases):
        sentiment = "positive" if selected == "explanation_first" else "negative"
        return {"target": "behavior", "sentiment": sentiment, "confidence": 0.95}
    if any(phrase in message for phrase in ("helpful", "perfect", "exactly", "easier")):
        return {"target": "behavior", "sentiment": "positive", "confidence": 0.9}
    return {"target": "task_continuation", "sentiment": "none", "confidence": 0.9}


def offline_generate(prompt: str, action: str) -> tuple[str, dict[str, int | None]]:
    if action == "code_first":
        response = (
            f"Code first: apply the solution for `{prompt[:50]}`. "
            "Explanation: this ordering presents the implementation before its rationale."
        )
    else:
        response = (
            "Explanation first: establish the approach before its implementation. "
            f"Code: apply the solution for `{prompt[:50]}`."
        )
    return response, {"input_tokens": None, "output_tokens": None}


def real_components(repo_root: Path) -> tuple[LLMFeedbackExtractor, Generator, str, str]:
    from dotenv import load_dotenv
    from openai import OpenAI

    from examples.openai_judge import build_openai_judge

    load_dotenv(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("insert-"):
        raise RuntimeError("set OPENAI_API_KEY in .env")
    judge, judge_model = build_openai_judge()
    configured_agent_model = os.getenv("OPENAI_AGENT_MODEL")
    agent_model = (
        judge_model
        if not configured_agent_model or configured_agent_model.startswith("insert-")
        else configured_agent_model
    )
    client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)

    def generate(prompt: str, action: str) -> tuple[str, dict[str, int | None]]:
        response = client.responses.create(
            model=agent_model,
            instructions=f"{BASE_INSTRUCTIONS}\n\n{ACTION_INSTRUCTIONS[action]}",
            input=prompt,
            store=False,
            max_output_tokens=4000,
            reasoning={"effort": "low"},
        )
        usage = response.usage
        if not response.output_text.strip():
            incomplete_reason = (
                response.incomplete_details.reason
                if response.incomplete_details is not None
                else None
            )
            raise RuntimeError(
                "model returned no visible text "
                f"(status={response.status}, incomplete_reason={incomplete_reason})"
            )
        return response.output_text, {
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        }

    return LLMFeedbackExtractor(judge=judge), generate, agent_model, judge_model


def show_policy(profile: Profile, context: str) -> None:
    policy = profile.policy(context)
    print(f"\nPolicy version: {policy['version']}")
    actions = cast(dict[str, dict[str, float]], policy["actions"])
    for action, state in actions.items():
        print(
            f"  {action:22} Beta({state['alpha']:.0f}, {state['beta']:.0f}) "
            f"posterior mean={state['posterior_mean']:.1%}"
        )


def print_help() -> None:
    print(
        "Commands: /state, /context NAME, /user ID, /mode active|shadow, "
        "/feedback up|down, /export, /delete-user, /quit"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively dogfood AdaptKit end to end.")
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--context", default="debugging")
    parser.add_argument("--database", type=Path, default=Path(".adaptkit-playground.db"))
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use configured real generation and judge models; incurs API cost.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.real:
        evaluator, generate, agent_model, judge_model = real_components(repo_root)
        print(f"Real mode: agent={agent_model} judge={judge_model} store=False")
    else:
        evaluator = LLMFeedbackExtractor(judge=offline_judge)
        generate = offline_generate
        print("Offline mode: deterministic local agent and evaluator")

    store = SQLiteStore(args.database)
    current_user = args.user_id
    current_context = args.context

    def new_profile(user_id: str) -> Profile:
        return Profile(
            user_id=user_id,
            actions=tuple(ACTION_INSTRUCTIONS),
            evaluator=evaluator,
            store=store,
            seed=7,
        )

    profile = new_profile(current_user)
    previous_decision = None
    previous_prompt = None
    previous_response = None
    turn = 0
    print_help()

    while True:
        try:
            raw = input(f"\n{current_user}@{current_context}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/state":
            show_policy(profile, current_context)
            continue
        if raw.startswith("/context "):
            current_context = raw.split(maxsplit=1)[1].strip()
            previous_decision = previous_prompt = previous_response = None
            show_policy(profile, current_context)
            continue
        if raw.startswith("/user "):
            current_user = raw.split(maxsplit=1)[1].strip()
            profile = new_profile(current_user)
            previous_decision = previous_prompt = previous_response = None
            show_policy(profile, current_context)
            continue
        if raw.startswith("/mode "):
            profile.set_learning_mode(raw.split(maxsplit=1)[1].strip())
            print(f"Learning mode: {profile.learning_mode.value}")
            continue
        if raw.startswith("/feedback "):
            if previous_decision is None:
                print("No previous decision is available.")
                continue
            direction = raw.split(maxsplit=1)[1].strip()
            key = f"explicit-{uuid4()}"
            if direction == "up":
                result = profile.like(previous_decision, idempotency_key=key)
            elif direction == "down":
                result = profile.dislike(previous_decision, idempotency_key=key)
            else:
                print("Use /feedback up or /feedback down.")
                continue
            print(
                f"Explicit feedback: {result.status.value}; "
                f"policy {result.policy_version_before} → {result.policy_version_after}"
            )
            show_policy(profile, current_context)
            continue
        if raw == "/export":
            print(json.dumps(profile.export_user(), indent=2))
            continue
        if raw == "/delete-user":
            profile.delete_user()
            previous_decision = previous_prompt = previous_response = None
            print("Deleted this user's structured AdaptKit data.")
            continue
        if raw.startswith("/"):
            print_help()
            continue

        if previous_decision is not None:
            assert previous_prompt is not None and previous_response is not None
            result = profile.observe(
                decision=previous_decision,
                idempotency_key=f"implicit-turn-{turn}",
                previous_prompt=previous_prompt,
                previous_response=previous_response,
                user_message=raw,
            )
            event = result.event
            if event is None:
                print(f"Evaluator: {result.status.value}")
            else:
                print(
                    f"Evaluator: {event.target.value}/{event.sentiment.value} "
                    f"confidence={event.confidence:.2f}; {result.status.value}; "
                    f"policy {result.policy_version_before} → {result.policy_version_after}"
                )

        decision = profile.choose(current_context)
        instruction = ACTION_INSTRUCTIONS[decision.action]
        print(f"\nSelected action: {decision.action}")
        print(f"Applied instruction: {instruction}")
        try:
            response, usage = generate(raw, decision.action)
        except Exception as exc:
            print(f"\nAgent generation failed safely: {type(exc).__name__}: {exc}")
            print("This decision will not be used as a feedback-learning example.")
            previous_decision = previous_prompt = previous_response = None
            continue
        print(f"\nAgent:\n{response}")
        if args.real:
            print(
                f"Usage: input={usage['input_tokens']} output={usage['output_tokens']}"
            )
        show_policy(profile, current_context)
        previous_decision = decision
        previous_prompt = raw
        previous_response = response
        turn += 1


if __name__ == "__main__":
    main()

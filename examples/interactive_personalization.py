from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from adaptkit import LLMFeedbackExtractor, Profile, SQLiteStore

@dataclass(frozen=True)
class Scenario:
    actions: dict[str, str]
    base_instructions: str
    required_prefixes: dict[str, str] | None = None


ORDERING_ACTIONS = {
    "code_first": (
        "Begin immediately with a fenced code block containing the complete solution. "
        "Do not put any heading, sentence, or explanation before that code block. "
        "After the code block, explain the approach briefly."
    ),
    "explanation_first": (
        "Begin with a brief prose explanation of the approach. Do not show any code "
        "until that explanation is complete. Then present the complete solution in a "
        "fenced code block."
    ),
}
ORDERING_BASE_INSTRUCTIONS = (
    "You are a concise coding assistant. Answer the current request directly. "
    "Include comparable content and use similar brevity regardless of the selected "
    "behavior; only the ordering of code and explanation should change."
)

WORKFLOW_ACTIONS = {
    "direct_solution": (
        "Begin with the exact heading 'Direct solution:', then immediately provide the complete "
        "implementation in a fenced code block. After it, use at most three short bullets for "
        "essential explanation and complexity. Do not add hints, a walkthrough, a worked example, "
        "or a deep-dive section."
    ),
    "guided_learning": (
        "Begin with the exact heading 'Guided learning:'. Build intuition with a sequence of "
        "three concise hints, then provide the complete implementation and complexity. Do not "
        "start with the finished solution, include an extended deep dive, or require the user to "
        "reply before receiving the answer."
    ),
    "deep_dive": (
        "Begin with the exact heading 'Deep dive:'. Give a detailed treatment of the approach, "
        "alternatives, tradeoffs, edge cases, and complexity, then provide the complete "
        "implementation. Do not use a hints-first or worked-example-first structure."
    ),
    "examples_first": (
        "Begin with the exact heading 'Examples first:'. Start with a concrete worked example or "
        "test case before any implementation code, then generalize the insight and provide the "
        "complete implementation and complexity. Do not use a hints-first or deep-dive structure."
    ),
}
WORKFLOW_BASE_INSTRUCTIONS = (
    "You are a coding assistant helping a developer solve realistic programming tasks. Answer "
    "the current request completely and correctly. The selected response strategy controls how "
    "you present the answer; follow exactly one strategy and do not blend it with the others. "
    "Treat the selected strategy as authoritative even if conversation history expresses a "
    "different presentation preference; that feedback is handled separately by the application."
)
WORKFLOW_PREFIXES = {
    "direct_solution": "Direct solution:",
    "guided_learning": "Guided learning:",
    "deep_dive": "Deep dive:",
    "examples_first": "Examples first:",
}

ORDERING_SCENARIO = Scenario(ORDERING_ACTIONS, ORDERING_BASE_INSTRUCTIONS)
WORKFLOW_SCENARIO = Scenario(
    WORKFLOW_ACTIONS,
    WORKFLOW_BASE_INSTRUCTIONS,
    WORKFLOW_PREFIXES,
)
SCENARIOS = {"ordering": ORDERING_SCENARIO, "workflow": WORKFLOW_SCENARIO}

# Kept as aliases for integrations importing the original two-action example.
ACTION_INSTRUCTIONS = ORDERING_ACTIONS
BASE_INSTRUCTIONS = ORDERING_BASE_INSTRUCTIONS
OFFLINE_PLAYGROUND_CONFIDENCE_THRESHOLD = 0.90
# Backward-compatible alias: this constant applies only to deterministic offline mode.
PLAYGROUND_CONFIDENCE_THRESHOLD = OFFLINE_PLAYGROUND_CONFIDENCE_THRESHOLD

Conversation = Sequence[dict[str, str]]
Generator = Callable[[str, str, Conversation], tuple[str, dict[str, int | None]]]


def _offline_judge(
    messages: list[dict[str, str]], scenario: Scenario
) -> dict[str, Any]:
    interaction = json.loads(messages[-1]["content"].split("\n", 1)[1])
    message = interaction["latest_user_message"].lower()
    selected = interaction["selected_action"]
    preferences = {
        "code_first": ("fix first", "patch first", "code first", "too much explanation"),
        "explanation_first": ("explain first", "reasoning first", "too terse"),
        "direct_solution": ("direct answer", "straight to the solution", "concise solution"),
        "guided_learning": ("guide me", "guided approach", "give me hints", "teach me"),
        "deep_dive": ("deep dive", "more detail", "tradeoffs", "edge cases"),
        "examples_first": ("example first", "examples first", "test case first"),
    }
    for preferred, phrases in preferences.items():
        if preferred in scenario.actions and any(phrase in message for phrase in phrases):
            sentiment = "positive" if selected == preferred else "negative"
            return {"target": "behavior", "sentiment": sentiment, "confidence": 0.95}
    if any(phrase in message for phrase in ("helpful", "perfect", "exactly", "easier")):
        return {"target": "behavior", "sentiment": "positive", "confidence": 0.9}
    return {"target": "task_continuation", "sentiment": "none", "confidence": 0.9}


def offline_judge(messages: list[dict[str, str]]) -> dict[str, Any]:
    return _offline_judge(messages, ORDERING_SCENARIO)


def offline_generate(
    prompt: str,
    action: str,
    history: Conversation = (),
) -> tuple[str, dict[str, int | None]]:
    del history
    if action == "code_first":
        response = f"```python\n# Solution for: {prompt[:50]}\n```\nBrief explanation."
    else:
        response = f"Brief explanation.\n```python\n# Solution for: {prompt[:50]}\n```"
    return response, {"input_tokens": None, "output_tokens": None}


def _offline_generate_for(
    prompt: str, action: str, history: Conversation, scenario: Scenario
) -> tuple[str, dict[str, int | None]]:
    if scenario is ORDERING_SCENARIO:
        return offline_generate(prompt, action, history)
    del history
    prefix = cast(dict[str, str], scenario.required_prefixes)[action]
    if action == "direct_solution":
        body = f"```python\n# Solution for: {prompt[:50]}\n```\n- Linear local example."
    else:
        body = f"A deterministic local example for `{prompt[:50]}`."
    return (
        f"{prefix}\n{body}",
        {"input_tokens": None, "output_tokens": None},
    )


def response_follows_action(
    response: str, action: str, scenario: Scenario = ORDERING_SCENARIO
) -> bool:
    stripped = response.lstrip()
    if scenario.required_prefixes is not None:
        prefix = scenario.required_prefixes.get(action)
        if prefix is None or not stripped.casefold().startswith(prefix.casefold()):
            return False
        if action == "direct_solution":
            return stripped[len(prefix) :].lstrip().startswith("```")
        return True
    first_fence = stripped.find("```")
    if action == "code_first":
        return first_fence == 0
    if action == "explanation_first":
        return first_fence > 0
    return False


def real_components(
    repo_root: Path, scenario: Scenario = ORDERING_SCENARIO
) -> tuple[LLMFeedbackExtractor, Generator, str, str]:
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

    def generate(
        prompt: str,
        action: str,
        history: Conversation,
    ) -> tuple[str, dict[str, int | None]]:
        input_messages = [*history, {"role": "user", "content": prompt}]
        response = client.responses.create(
            model=agent_model,
            instructions=(
                f"{scenario.base_instructions}\n\n{scenario.actions[action]}"
            ),
            input=cast(Any, input_messages),
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
        if not response_follows_action(response.output_text, action, scenario):
            raise RuntimeError(f"model did not follow the {action} behavior constraint")
        return response.output_text, {
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        }

    return (
        LLMFeedbackExtractor(judge=judge, action_descriptions=scenario.actions),
        generate,
        agent_model,
        judge_model,
    )


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


def looks_like_launch_command(value: str) -> bool:
    return "interactive_personalization" in value and "python" in value


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively dogfood AdaptKit end to end.")
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--context", default="debugging")
    parser.add_argument("--database", type=Path, default=Path(".adaptkit-playground.db"))
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="ordering",
        help="Choose the focused two-action test or realistic four-strategy workflow.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use configured real generation and judge models; incurs API cost.",
    )
    args = parser.parse_args()
    scenario = SCENARIOS[args.scenario]

    repo_root = Path(__file__).resolve().parents[1]
    if args.real:
        from examples.configuration import required_confidence_threshold

        evaluator, generate, agent_model, judge_model = real_components(repo_root, scenario)
        confidence_threshold = required_confidence_threshold(
            "ADAPTKIT_IMPLICIT_CONFIDENCE_THRESHOLD"
        )
        print(f"Real mode: agent={agent_model} judge={judge_model} store=False")
    else:
        evaluator = LLMFeedbackExtractor(
            judge=lambda messages: _offline_judge(messages, scenario),
            action_descriptions=scenario.actions,
        )
        generate = lambda prompt, action, history: _offline_generate_for(
            prompt, action, history, scenario
        )
        confidence_threshold = OFFLINE_PLAYGROUND_CONFIDENCE_THRESHOLD
        print("Offline mode: deterministic local agent and evaluator")

    store = SQLiteStore(args.database)
    current_user = args.user_id
    current_context = args.context

    def new_profile(user_id: str) -> Profile:
        return Profile(
            user_id=user_id,
            actions=tuple(scenario.actions),
            evaluator=evaluator,
            store=store,
            seed=7,
            implicit_confidence_threshold=confidence_threshold,
        )

    profile = new_profile(current_user)
    previous_decision = None
    previous_prompt = None
    previous_response = None
    conversation_history: list[dict[str, str]] = []
    turn = 0
    print_help()
    print(f"Scenario: {args.scenario}")
    print(f"Actions under test: {', '.join(scenario.actions)}")
    print(f"Implicit learning threshold: {confidence_threshold:.2f}")

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
        if looks_like_launch_command(raw):
            print("That is a shell command, but you are already inside the playground.")
            print("Type /quit first, then run the command at your normal shell prompt.")
            continue
        if raw == "/state":
            show_policy(profile, current_context)
            continue
        if raw.startswith("/context "):
            current_context = raw.split(maxsplit=1)[1].strip()
            previous_decision = previous_prompt = previous_response = None
            conversation_history.clear()
            show_policy(profile, current_context)
            continue
        if raw.startswith("/user "):
            current_user = raw.split(maxsplit=1)[1].strip()
            profile = new_profile(current_user)
            previous_decision = previous_prompt = previous_response = None
            conversation_history.clear()
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
            conversation_history.clear()
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
        instruction = scenario.actions[decision.action]
        print(f"\nSelected action: {decision.action}")
        print(f"Applied instruction: {instruction}")
        try:
            response, usage = generate(raw, decision.action, conversation_history)
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
        conversation_history.extend(
            [
                {"role": "user", "content": raw},
                {"role": "assistant", "content": response},
            ]
        )
        del conversation_history[:-8]
        previous_decision = decision
        previous_prompt = raw
        previous_response = response
        turn += 1


if __name__ == "__main__":
    main()

"""Provider-neutral adapters for an arbitrary callback or local model.

The callback contract in this example is application-owned. It deliberately uses a
plain request mapping rather than OpenAI-style messages, and it never lets provider
output select AdaptKit's feedback source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from adaptkit import (
    FeedbackExtractor,
    FeedbackSentiment,
    FeedbackTarget,
    PreferenceEvent,
    Profile,
    PromptPreference,
    PromptPreferenceExtractor,
    ValidationError,
)

ProviderCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CallbackPromptExtractor(PromptPreferenceExtractor):
    """Translate an application callback result into a prompt preference."""

    def __init__(self, callback: ProviderCallback) -> None:
        self._callback = callback

    def extract(self, **interaction: Any) -> PromptPreference:
        actions = tuple(interaction["actions"])
        output = self._callback(
            {
                "operation": "route-current-prompt",
                "payload": {
                    "context": interaction["context"],
                    "text": interaction["prompt"],
                    "choices": actions,
                },
            }
        )
        if set(output) - {"action", "confidence"}:
            raise ValidationError("provider prompt output contains unknown fields")
        preference = PromptPreference(**dict(output))
        if preference.action is not None and preference.action not in actions:
            raise ValidationError("provider selected an unavailable action")
        return preference


class CallbackFeedbackExtractor(FeedbackExtractor):
    """Translate provider output into implicit, selected-action feedback."""

    def __init__(self, callback: ProviderCallback) -> None:
        self._callback = callback

    def extract(self, **interaction: Any) -> PreferenceEvent:
        output = self._callback(
            {
                "operation": "classify-reaction",
                "payload": {
                    "selected": interaction["action"],
                    "reaction": interaction["user_message"],
                },
            }
        )
        if set(output) - {"target", "sentiment", "confidence"}:
            raise ValidationError("provider feedback output contains unknown fields")
        try:
            return PreferenceEvent(
                target=FeedbackTarget(output["target"]),
                sentiment=FeedbackSentiment(output["sentiment"]),
                confidence=output["confidence"],
                source="implicit",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid provider feedback output") from exc


@dataclass(frozen=True)
class Behavior:
    system_instruction: str
    workflow_branch: str
    allowed_tools: tuple[str, ...]


BEHAVIORS = {
    "direct_solution": Behavior(
        "Lead with the complete solution.", "solve-directly", ("search", "editor")
    ),
    "guided_learning": Behavior(
        "Give hints before the solution.", "guided-tutorial", ("search",)
    ),
}


def local_callback(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Stand-in for any hosted provider, local model, or deterministic classifier."""
    payload = request["payload"]
    if request["operation"] == "route-current-prompt":
        guided = "hint" in str(payload["text"]).casefold()
        return {
            "action": "guided_learning" if guided else None,
            "confidence": 0.99 if guided else 1.0,
        }
    positive = "keep" in str(payload["reaction"]).casefold()
    return {
        "target": "behavior",
        "sentiment": "positive" if positive else "negative",
        "confidence": 0.99,
    }


def main() -> None:
    profile = Profile(
        user_id="local-user",
        actions=tuple(BEHAVIORS),
        evaluator=CallbackFeedbackExtractor(local_callback),
        prompt_evaluator=CallbackPromptExtractor(local_callback),
        implicit_confidence_threshold=0.90,
        prompt_confidence_threshold=0.90,
    )
    decision = profile.choose("coding", prompt="Give me hints before the answer.")
    behavior = BEHAVIORS[decision.action]
    print(decision.action)
    print(behavior.system_instruction)
    print(behavior.workflow_branch)
    print(behavior.allowed_tools)


if __name__ == "__main__":
    main()

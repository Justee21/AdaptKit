from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from adaptkit import (
    FeedbackExtractor,
    FeedbackSentiment,
    FeedbackTarget,
    InMemoryStore,
    LearningMode,
    ObservationStatus,
    PreferenceEvent,
    Profile,
    PromptPreference,
    PromptPreferenceExtractor,
    SQLiteStore,
)
from examples.generic_provider import (
    BEHAVIORS,
    CallbackFeedbackExtractor,
    CallbackPromptExtractor,
    local_callback,
)


class SyncPromptExtractor(PromptPreferenceExtractor):
    def extract(self, **interaction: Any) -> PromptPreference:
        return PromptPreference(action="b", confidence=0.99)


class AsyncPromptExtractor(PromptPreferenceExtractor):
    async def aextract(self, **interaction: Any) -> PromptPreference:
        return PromptPreference(action="b", confidence=0.99)


class SyncFeedbackExtractor(FeedbackExtractor):
    def extract(self, **interaction: Any) -> PreferenceEvent:
        return PreferenceEvent(
            FeedbackTarget.BEHAVIOR,
            FeedbackSentiment.POSITIVE,
            confidence=0.99,
        )


class AsyncFeedbackExtractor(FeedbackExtractor):
    async def aextract(self, **interaction: Any) -> PreferenceEvent:
        return PreferenceEvent(
            FeedbackTarget.BEHAVIOR,
            FeedbackSentiment.POSITIVE,
            confidence=0.99,
        )


def interaction(decision: Any, key: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "idempotency_key": key,
        "previous_prompt": "private prompt",
        "previous_response": "private response",
        "user_message": "keep this format",
    }


@pytest.mark.parametrize("extractor", [SyncPromptExtractor(), CallbackPromptExtractor(local_callback)])
def test_custom_synchronous_prompt_pathways(extractor: PromptPreferenceExtractor) -> None:
    actions = ["a", "b"] if isinstance(extractor, SyncPromptExtractor) else list(BEHAVIORS)
    prompt = "Use B" if isinstance(extractor, SyncPromptExtractor) else "Give me hints"
    expected = "b" if isinstance(extractor, SyncPromptExtractor) else "guided_learning"
    profile = Profile(user_id="u", actions=actions, prompt_evaluator=extractor)
    decision = profile.choose("ctx", prompt=prompt)
    assert decision.action == expected
    assert profile.policy("ctx")["version"] == 0


def test_custom_asynchronous_prompt_pathway() -> None:
    profile = Profile(
        user_id="u", actions=["a", "b"], prompt_evaluator=AsyncPromptExtractor()
    )
    assert asyncio.run(profile.achoose("ctx", prompt="Use B")).action == "b"


def test_custom_synchronous_and_asynchronous_feedback_pathways() -> None:
    sync = Profile(user_id="sync", actions=["a"], evaluator=SyncFeedbackExtractor())
    sync_result = sync.observe(**interaction(sync.choose("ctx"), "sync"))
    asynchronous = Profile(
        user_id="async", actions=["a"], evaluator=AsyncFeedbackExtractor()
    )
    async_result = asyncio.run(
        asynchronous.aobserve(**interaction(asynchronous.choose("ctx"), "async"))
    )
    assert sync_result.status == async_result.status == ObservationStatus.UPDATED
    assert sync.state()["ctx"]["a"] == asynchronous.state()["ctx"]["a"]


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_direct_feedback_is_evaluator_free_across_stores(
    kind: str, tmp_path: Path
) -> None:
    store = InMemoryStore() if kind == "memory" else SQLiteStore(tmp_path / "state.db")
    profile = Profile(user_id="u", actions=["a", "b"], store=store)
    decision = profile.choose("ctx")
    assert profile.like(decision, idempotency_key="like").updated
    assert profile.dislike(decision, idempotency_key="dislike").updated
    passive = profile.signal(
        decision,
        signal_type="ui.regeneration",
        sentiment="negative",
        confidence=1,
        idempotency_key="passive",
    )
    pairwise = profile.prefer(
        "ctx", preferred="a", rejected="b", idempotency_key="pairwise"
    )
    assert passive.updated and pairwise.updated
    assert profile.policy("ctx")["version"] == 4


def test_action_controls_instruction_workflow_and_tools() -> None:
    profile = Profile(
        user_id="u",
        actions=tuple(BEHAVIORS),
        prompt_evaluator=CallbackPromptExtractor(local_callback),
    )
    decision = profile.choose("ctx", prompt="Give me hints")
    applied = BEHAVIORS[decision.action]
    assert applied.system_instruction == "Give hints before the solution."
    assert applied.workflow_branch == "guided-tutorial"
    assert applied.allowed_tools == ("search",)


def test_active_and_shadow_modes_are_provider_neutral() -> None:
    active = Profile(user_id="active", actions=["a"], evaluator=SyncFeedbackExtractor())
    shadow = Profile(
        user_id="shadow",
        actions=["a"],
        evaluator=SyncFeedbackExtractor(),
        learning_mode=LearningMode.SHADOW,
    )
    active_result = active.observe(**interaction(active.choose("ctx"), "active"))
    shadow_result = shadow.observe(**interaction(shadow.choose("ctx"), "shadow"))
    assert active_result.status == ObservationStatus.UPDATED
    assert shadow_result.status == ObservationStatus.SHADOW
    assert active.policy("ctx")["version"] == 1
    assert shadow.policy("ctx")["version"] == 0


def test_generic_feedback_callback_uses_implicit_trust() -> None:
    extractor = CallbackFeedbackExtractor(local_callback)
    profile = Profile(user_id="u", actions=["a"], evaluator=extractor)
    result = profile.observe(**interaction(profile.choose("ctx"), "callback"))
    assert result.event is not None
    assert result.event.source == "implicit"


def test_provider_timeout_and_malformed_output_fail_closed() -> None:
    class TimeoutPrompt(PromptPreferenceExtractor):
        def extract(self, **interaction: Any) -> PromptPreference:
            raise TimeoutError("private provider details")

    class MalformedFeedback(FeedbackExtractor):
        def extract(self, **interaction: Any) -> PreferenceEvent:
            return cast(PreferenceEvent, {"invalid": True})

    routed = Profile(
        user_id="route", actions=["a"], prompt_evaluator=TimeoutPrompt()
    ).choose("ctx", prompt="private prompt")
    assert routed.action == "a"
    feedback = Profile(
        user_id="feedback", actions=["a"], evaluator=MalformedFeedback()
    )
    result = feedback.observe(**interaction(feedback.choose("ctx"), "malformed"))
    assert result.status == ObservationStatus.EVALUATOR_ERROR
    assert "private" not in repr(result)


def test_provider_retry_deduplicates_before_second_evaluation() -> None:
    calls = 0

    class CountingFeedback(SyncFeedbackExtractor):
        def extract(self, **interaction: Any) -> PreferenceEvent:
            nonlocal calls
            calls += 1
            return super().extract(**interaction)

    profile = Profile(user_id="u", actions=["a"], evaluator=CountingFeedback())
    decision = profile.choose("ctx")
    first = profile.observe(**interaction(decision, "stable-event"))
    retry = profile.observe(**interaction(decision, "stable-event"))
    assert first.updated and retry.duplicate
    assert calls == 1
    assert profile.policy("ctx")["version"] == 1


def test_provider_async_cancellation_propagates() -> None:
    class CancelledFeedback(FeedbackExtractor):
        async def aextract(self, **interaction: Any) -> PreferenceEvent:
            raise asyncio.CancelledError

    profile = Profile(user_id="u", actions=["a"], evaluator=CancelledFeedback())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            profile.aobserve(**interaction(profile.choose("ctx"), "cancelled"))
        )
    assert profile.policy("ctx")["version"] == 0

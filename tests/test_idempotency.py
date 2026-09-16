from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from adaptkit import (
    FeedbackExtractor,
    IdempotencyConflictError,
    InMemoryStore,
    ObservationStatus,
    Profile,
    SQLiteStore,
)


@pytest.fixture(params=("memory", "sqlite"))
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "adaptkit.db", busy_timeout=5)


def test_direct_feedback_exact_retry_and_conflict(store) -> None:
    profile = Profile(user_id="u", actions=["a"], store=store)
    decision = profile.choose("ctx")

    first = profile.like(decision, idempotency_key="event")
    duplicate = profile.like(decision, idempotency_key="event")
    assert first.updated
    assert duplicate.status is ObservationStatus.DUPLICATE

    with pytest.raises(IdempotencyConflictError):
        profile.dislike(decision, idempotency_key="event")
    assert profile.policy("ctx")["version"] == 1


def test_learning_mode_is_part_of_direct_operation_identity(store) -> None:
    profile = Profile(user_id="u", actions=["a"], store=store)
    decision = profile.choose("ctx")

    shadow = profile.like(decision, idempotency_key="event", apply=False)
    duplicate = profile.like(decision, idempotency_key="event", apply=False)
    assert shadow.status is ObservationStatus.SHADOW
    assert duplicate.duplicate

    with pytest.raises(IdempotencyConflictError):
        profile.like(decision, idempotency_key="event", apply=True)
    assert profile.policy("ctx")["version"] == 0


def test_pairwise_feedback_exact_retry_and_conflict(store) -> None:
    profile = Profile(user_id="u", actions=["a", "b"], store=store)

    first = profile.prefer(
        "ctx", preferred="a", rejected="b", idempotency_key="pair"
    )
    duplicate = profile.prefer(
        "ctx", preferred="a", rejected="b", idempotency_key="pair"
    )
    assert first.updated
    assert duplicate.duplicate

    with pytest.raises(IdempotencyConflictError):
        profile.prefer(
            "ctx", preferred="b", rejected="a", idempotency_key="pair"
        )
    assert profile.policy("ctx")["version"] == 1


def test_concurrent_conflicting_direct_feedback_updates_once(store) -> None:
    profile = Profile(user_id="u", actions=["a"], store=store)
    decision = profile.choose("ctx")
    barrier = threading.Barrier(3)

    def submit(positive: bool):
        barrier.wait(timeout=5)
        method = profile.like if positive else profile.dislike
        return method(decision, idempotency_key="same-event")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, value) for value in (True, False)]
        barrier.wait(timeout=5)
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except IdempotencyConflictError as exc:
                outcomes.append(exc)

    assert sum(isinstance(item, IdempotencyConflictError) for item in outcomes) == 1
    state = profile.state()["ctx"]["a"]
    assert state["alpha"] + state["beta"] == 3
    assert profile.policy("ctx")["version"] == 1


def test_nested_metadata_is_detached_at_every_boundary(store) -> None:
    profile = Profile(user_id="u", actions=["a"], store=store)
    decision = profile.choose("ctx")
    original: dict[str, Any] = {"nested": {"items": [{"value": "kept"}]}}

    result = profile.signal(
        decision,
        signal_type="ui.regeneration",
        sentiment="negative",
        metadata=original,
        idempotency_key="signal",
    )
    original["nested"]["items"][0]["value"] = "input-mutated"
    assert result.event is not None
    cast(Any, result.event.metadata)["nested"]["items"][0][
        "value"
    ] = "result-mutated"

    exact_payload = {"nested": {"items": [{"value": "kept"}]}}
    duplicate = profile.signal(
        decision,
        signal_type="ui.regeneration",
        sentiment="negative",
        metadata=exact_payload,
        idempotency_key="signal",
    )
    assert duplicate.event is not None
    cast(Any, duplicate.event.metadata)["nested"]["items"][0][
        "value"
    ] = "duplicate-mutated"

    exported = profile.export_user()
    cast(Any, exported)["observations"][0]["metadata"]["nested"]["items"][0][
        "value"
    ] = "export-mutated"

    later = profile.export_user()
    assert later["observations"][0]["metadata"] == exact_payload
    assert profile.signal(
        decision,
        signal_type="ui.regeneration",
        sentiment="negative",
        metadata=exact_payload,
        idempotency_key="signal",
    ).duplicate


def test_observe_replay_is_privacy_first(store) -> None:
    calls = 0

    class Evaluator(FeedbackExtractor):
        def extract(self, **_interaction):
            nonlocal calls
            calls += 1
            from adaptkit import FeedbackSentiment, FeedbackTarget, PreferenceEvent

            return PreferenceEvent(
                FeedbackTarget.BEHAVIOR,
                FeedbackSentiment.POSITIVE,
                source="implicit",
            )

    profile = Profile(user_id="u", actions=["a"], store=store, evaluator=Evaluator())
    decision = profile.choose("ctx")
    profile.observe(
        decision=decision,
        idempotency_key="turn",
        previous_prompt="first prompt",
        previous_response="first response",
        user_message="first message",
    )
    assert profile.observe(
        decision=decision,
        idempotency_key="turn",
        previous_prompt="different prompt",
        previous_response="different response",
        user_message="different replay content",
    ).duplicate
    assert calls == 1


def test_pairwise_update_rolls_back_without_recording_operation(store) -> None:
    profile = Profile(user_id="u", actions=["a", "b"], store=store)
    profile.policy("ctx")

    def increment(state):
        state["alpha"] += 1

    def fail(state):
        state["beta"] += 1
        raise RuntimeError("injected failure")

    arguments = {
        "user_id": "u",
        "context": "ctx",
        "idempotency_key": "pair",
        "operation_identity": {
            "operation": "prefer",
            "preferred": "a",
            "rejected": "b",
            "learning_mode": "active",
        },
        "initial_states": {
            "a": {"alpha": 1.0, "beta": 1.0},
            "b": {"alpha": 1.0, "beta": 1.0},
        },
    }
    with pytest.raises(RuntimeError, match="injected failure"):
        store.atomic_policy_update(
            **arguments,
            updaters={"a": increment, "b": fail},
        )
    assert profile.state()["ctx"] == {
        "a": {"alpha": 1.0, "beta": 1.0},
        "b": {"alpha": 1.0, "beta": 1.0},
    }
    retry = store.atomic_policy_update(
        **arguments,
        updaters={"a": increment, "b": increment},
    )
    assert retry.updated
    assert not retry.duplicate

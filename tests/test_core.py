import asyncio
import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from typing import cast

from adaptkit import (
    ConfigurationError,
    Decision,
    DecisionMismatchError,
    FeedbackExtractor,
    FeedbackSentiment,
    FeedbackTarget,
    InMemoryStore,
    LearningMode,
    LLMFeedbackExtractor,
    ObservationStatus,
    PreferenceEvent,
    Profile,
    ValidationError,
)


def interaction(decision: Decision, key: str) -> dict[str, object]:
    return {
        "decision": decision,
        "idempotency_key": key,
        "previous_prompt": "Why does this fail?",
        "previous_response": "Here is a detailed explanation.",
        "user_message": "Just show me the fix.",
    }


class CoreTests(unittest.TestCase):
    def test_profile_validates_configuration(self):
        self.assertEqual(
            Profile(user_id="u", actions=["a"]).implicit_confidence_threshold,
            0.70,
        )
        with self.assertRaises(ValidationError):
            Profile(user_id="", actions=["a"])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=[])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a", "a"])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a"], implicit_confidence_threshold=0)
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a"], prior_alpha=0)
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a"], learning_mode="invalid")
        with self.assertRaises(ConfigurationError):
            Profile(user_id="u", actions=["a"], learner="missing")

    def test_choose_returns_immutable_decision(self):
        profile = Profile(user_id="u", actions=["a", "b"], seed=3)
        decision = profile.choose("ctx")
        self.assertIn(decision.action, profile.actions)
        self.assertEqual(decision.user_id, "u")
        self.assertEqual(decision.context, "ctx")
        self.assertEqual(decision.policy_version, 0)
        self.assertIsNotNone(decision.created_at.tzinfo)
        with self.assertRaises(FrozenInstanceError):
            decision.action = "b"  # type: ignore[misc]

    def test_random_always_returns_allowed_action(self):
        profile = Profile(user_id="u", actions=["a", "b"], learner="random", seed=3)
        self.assertTrue(
            all(profile.choose("ctx").action in {"a", "b"} for _ in range(30))
        )

    def test_application_maps_action_to_visible_behavior(self):
        instructions = {
            "patch_first": "PATCH FIRST",
            "explanation_first": "EXPLANATION FIRST",
        }
        profile = Profile(user_id="u", actions=tuple(instructions), seed=4)
        decision = profile.choose("debugging")
        generated = f"system={instructions[decision.action]}"
        self.assertIn(instructions[decision.action], generated)

    def test_distinct_explicit_signals_can_update_one_decision(self):
        profile = Profile(user_id="u", actions=["a"])
        decision = profile.choose("ctx")
        first = profile.like(decision, idempotency_key="thumb-up")
        second = profile.like(decision, idempotency_key="later-confirmation")
        self.assertTrue(first.updated)
        self.assertTrue(second.updated)
        self.assertEqual(first.policy_version_after, 1)
        self.assertEqual(second.policy_version_after, 2)
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 3.0, "beta": 1.0})

    def test_duplicate_observation_updates_at_most_once(self):
        profile = Profile(user_id="u", actions=["a"])
        decision = profile.choose("ctx")
        first = profile.like(decision, idempotency_key="thumb-up")
        duplicate = profile.like(decision, idempotency_key="thumb-up")
        self.assertEqual(first.status, ObservationStatus.UPDATED)
        self.assertEqual(duplicate.status, ObservationStatus.DUPLICATE)
        self.assertTrue(duplicate.duplicate)
        self.assertFalse(duplicate.updated)
        self.assertEqual(first.observation_id, duplicate.observation_id)
        self.assertEqual(profile.state()["ctx"]["a"]["alpha"], 2)

    def test_pairwise_update_is_one_versioned_idempotent_operation(self):
        profile = Profile(user_id="u", actions=["a", "b"])
        first = profile.prefer(
            "ctx", preferred="a", rejected="b", idempotency_key="pair-1"
        )
        duplicate = profile.prefer(
            "ctx", preferred="a", rejected="b", idempotency_key="pair-1"
        )
        self.assertEqual(first.policy_version_before, 0)
        self.assertEqual(first.policy_version_after, 1)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.policy_version_after, 1)
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 2.0, "beta": 1.0})
        self.assertEqual(profile.state()["ctx"]["b"], {"alpha": 1.0, "beta": 2.0})
        self.assertEqual(profile.choose("ctx").policy_version, 1)

    def test_pairwise_update_respects_shadow_mode_and_override(self):
        profile = Profile(
            user_id="u",
            actions=["a", "b"],
            learning_mode=LearningMode.SHADOW,
        )
        shadow = profile.prefer(
            "ctx", preferred="a", rejected="b", idempotency_key="shadow"
        )
        active = profile.prefer(
            "ctx",
            preferred="a",
            rejected="b",
            idempotency_key="active",
            apply=True,
        )
        self.assertFalse(shadow.updated)
        self.assertTrue(active.updated)
        self.assertEqual(profile.policy("ctx")["version"], 1)

    def test_users_and_contexts_are_independent(self):
        store = InMemoryStore()
        first = Profile(user_id="first", actions=["a", "b"], store=store)
        second = Profile(user_id="second", actions=["a", "b"], store=store)
        first.like(first.choose("debugging"), idempotency_key="first-debug")
        first.dislike(first.choose("writing"), idempotency_key="first-writing")
        second.like(second.choose("debugging"), idempotency_key="second-debug")
        self.assertEqual(first.policy("debugging")["version"], 1)
        self.assertEqual(first.policy("writing")["version"], 1)
        self.assertEqual(second.policy("debugging")["version"], 1)

    def test_simultaneous_distinct_updates_are_atomic(self):
        store = InMemoryStore()
        profile = Profile(user_id="u", actions=["a"], store=store)
        decision = profile.choose("ctx")
        simultaneous_start = threading.Barrier(3)

        def positive_update(key: str) -> None:
            simultaneous_start.wait(timeout=5)
            profile.like(decision, idempotency_key=key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(positive_update, f"signal-{index}") for index in range(2)]
            simultaneous_start.wait(timeout=5)
            for future in futures:
                future.result(timeout=5)

        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 3.0, "beta": 1.0})
        self.assertEqual(profile.policy("ctx")["version"], 2)

    def test_signal_and_threshold_statuses(self):
        outputs = iter(
            [
                {"target": "task_continuation", "sentiment": "none"},
                {"target": "behavior", "sentiment": "positive", "confidence": 0.34},
                {"target": "behavior", "sentiment": "positive", "confidence": 0.35},
                {"target": "behavior", "sentiment": "negative", "confidence": 0.35},
            ]
        )
        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(judge=lambda _: next(outputs)),
            implicit_confidence_threshold=0.35,
        )
        decision = profile.choose("debugging")
        results = [
            profile.observe(**interaction(decision, f"signal-{index}"))
            for index in range(4)
        ]
        self.assertEqual(
            [result.status for result in results],
            [
                ObservationStatus.NO_PREFERENCE_SIGNAL,
                ObservationStatus.BELOW_THRESHOLD,
                ObservationStatus.UPDATED,
                ObservationStatus.UPDATED,
            ],
        )
        self.assertEqual(profile.state()["debugging"]["a"], {"alpha": 2.0, "beta": 2.0})

    def test_shadow_mode_and_apply_overrides(self):
        evaluator = LLMFeedbackExtractor(
            judge=lambda _: {
                "target": "behavior",
                "sentiment": "positive",
                "confidence": 1,
            }
        )
        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=evaluator,
            learning_mode=LearningMode.SHADOW,
        )
        decision = profile.choose("ctx")
        shadow = profile.observe(**interaction(decision, "shadow"))
        active = profile.observe(**interaction(decision, "forced-active"), apply=True)
        disabled = profile.observe(**interaction(decision, "forced-shadow"), apply=False)
        self.assertEqual(shadow.status, ObservationStatus.SHADOW)
        self.assertEqual(active.status, ObservationStatus.UPDATED)
        self.assertEqual(disabled.status, ObservationStatus.SHADOW)
        self.assertEqual(profile.policy("ctx")["version"], 1)

    def test_implicit_learning_can_be_disabled(self):
        profile = Profile(
            user_id="u",
            actions=["a"],
            implicit_learning_enabled=False,
            evaluator=LLMFeedbackExtractor(
                judge=lambda _: {
                    "target": "behavior",
                    "sentiment": "positive",
                    "confidence": 1,
                }
            ),
        )
        result = profile.observe(**interaction(profile.choose("ctx"), "implicit"))
        self.assertEqual(result.status, ObservationStatus.LEARNER_IGNORED)
        self.assertEqual(profile.state(), {})

    def test_malformed_output_fails_closed(self):
        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(
                judge=lambda _: {"target": "answer_content", "sentiment": "positive"}
            ),
        )
        result = profile.observe(**interaction(profile.choose("ctx"), "malformed"))
        self.assertEqual(result.status, ObservationStatus.EVALUATOR_ERROR)
        self.assertFalse(result.recorded)
        self.assertEqual(profile.state(), {})

    def test_pre_v1_feedback_payload_fails_closed(self):
        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(
                judge=lambda _: {"has_feedback": True, "reward": 1}
            ),
        )
        result = profile.observe(**interaction(profile.choose("ctx"), "legacy"))
        self.assertEqual(result.status, ObservationStatus.EVALUATOR_ERROR)

    def test_custom_extractor_contract_violation_fails_closed(self):
        class InvalidExtractor(FeedbackExtractor):
            def extract(self, **interaction) -> PreferenceEvent:
                return cast(
                    PreferenceEvent,
                    {"target": "behavior", "sentiment": "positive"},
                )

        profile = Profile(user_id="u", actions=["a"], evaluator=InvalidExtractor())
        result = profile.observe(**interaction(profile.choose("ctx"), "invalid"))
        self.assertEqual(result.status, ObservationStatus.EVALUATOR_ERROR)

    def test_evaluator_exception_fails_closed_without_sensitive_message(self):
        def broken(_):
            raise TimeoutError("SENTINEL-SENSITIVE-ERROR")

        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(judge=broken),
        )
        result = profile.observe(**interaction(profile.choose("ctx"), "timeout"))
        self.assertEqual(result.error, "TimeoutError")
        self.assertNotIn("SENTINEL", repr(result))
        self.assertEqual(profile.state(), {})

    def test_conversation_text_is_not_exported(self):
        sentinel = "SENTINEL-CONVERSATION-TEXT-92"
        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(
                judge=lambda _: {
                    "target": "behavior",
                    "sentiment": "positive",
                    "confidence": 1,
                    "reason": sentinel,
                }
            ),
        )
        decision = profile.choose("ctx")
        profile.observe(
            decision=decision,
            idempotency_key="privacy",
            previous_prompt=sentinel,
            previous_response=sentinel,
            user_message=sentinel,
        )
        self.assertNotIn(sentinel, repr(profile.export_user()))

    def test_sync_and_async_share_update_semantics(self):
        event = {"target": "behavior", "sentiment": "negative", "confidence": 0.9}
        sync_profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(judge=lambda _: event),
        )

        async def judge(_):
            return event

        async_profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(async_judge=judge),
        )
        sync_result = sync_profile.observe(
            **interaction(sync_profile.choose("ctx"), "sync")
        )
        async_result = asyncio.run(
            async_profile.aobserve(**interaction(async_profile.choose("ctx"), "async"))
        )
        self.assertEqual(sync_result.status, async_result.status)
        self.assertEqual(sync_profile.state(), async_profile.state())

    def test_async_cancellation_propagates_without_update(self):
        async def cancelled(_):
            raise asyncio.CancelledError

        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(async_judge=cancelled),
        )
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(profile.aobserve(**interaction(profile.choose("ctx"), "cancel")))
        self.assertEqual(profile.state(), {})

    def test_event_validation_and_derived_reward(self):
        positive = PreferenceEvent(
            FeedbackTarget.BEHAVIOR, FeedbackSentiment.POSITIVE, confidence=0.8
        )
        self.assertTrue(positive.has_preference_signal)
        self.assertEqual(positive.reward, 1)
        with self.assertRaises(ValidationError):
            PreferenceEvent(FeedbackTarget.BEHAVIOR, FeedbackSentiment.NONE)
        with self.assertRaises(ValidationError):
            PreferenceEvent(FeedbackTarget.ANSWER_CONTENT, FeedbackSentiment.POSITIVE)
        with self.assertRaises(ValidationError):
            PreferenceEvent(
                FeedbackTarget.BEHAVIOR,
                FeedbackSentiment.POSITIVE,
                confidence=math.inf,
            )

    def test_observation_requires_evaluator(self):
        profile = Profile(user_id="u", actions=["a"])
        decision = profile.choose("ctx")
        with self.assertRaises(ConfigurationError):
            profile.observe(**interaction(decision, "missing-evaluator"))

    def test_sync_observation_with_async_only_judge_is_configuration_error(self):
        async def judge(_):
            return {"target": "task_continuation", "sentiment": "none"}

        profile = Profile(
            user_id="u",
            actions=["a"],
            evaluator=LLMFeedbackExtractor(async_judge=judge),
        )
        with self.assertRaises(ConfigurationError):
            profile.observe(**interaction(profile.choose("ctx"), "async-only"))

    def test_decision_must_match_store_and_user(self):
        first = Profile(user_id="first", actions=["a"])
        second = Profile(user_id="second", actions=["a"], store=first.store)
        decision = first.choose("ctx")
        with self.assertRaises(DecisionMismatchError):
            second.like(decision, idempotency_key="wrong-user")
        forged = replace(decision, action="other")
        with self.assertRaises(ValidationError):
            first.like(forged, idempotency_key="forged")

    def test_max_decision_age(self):
        profile = Profile(
            user_id="u", actions=["a"], max_decision_age=timedelta(microseconds=1)
        )
        decision = profile.choose("ctx")
        with self.assertRaises(ValidationError):
            profile.like(decision, idempotency_key="expired")

    def test_expired_duplicate_returns_the_committed_result(self):
        profile = Profile(
            user_id="u", actions=["a"], max_decision_age=timedelta(days=1)
        )
        decision = profile.choose("ctx")
        first = profile.like(decision, idempotency_key="committed")
        profile.max_decision_age = timedelta(microseconds=1)
        duplicate = profile.like(decision, idempotency_key="committed")
        self.assertEqual(duplicate.status, ObservationStatus.DUPLICATE)
        self.assertEqual(duplicate.observation_id, first.observation_id)

    def test_event_hook_is_structured_and_cannot_break_learning(self):
        events = []

        def hook(event):
            events.append(event)
            if event.name == "observation_recorded":
                raise RuntimeError("hook failed")

        profile = Profile(user_id="u", actions=["a"], event_hook=hook)
        decision = profile.choose("ctx")
        result = profile.like(decision, idempotency_key="hook")
        self.assertTrue(result.updated)
        self.assertEqual([event.name for event in events], ["decision_created", "observation_recorded"])
        self.assertFalse(any(hasattr(event, "previous_prompt") for event in events))

    def test_export_and_delete_user(self):
        profile = Profile(user_id="u", actions=["a"])
        profile.like(profile.choose("ctx"), idempotency_key="export")
        exported = profile.export_user()
        self.assertEqual(exported["schema_version"], 1)
        self.assertEqual(len(cast(list, exported["decisions"])), 1)
        profile.delete_user()
        self.assertEqual(profile.state(), {})
        self.assertEqual(cast(list, profile.export_user()["decisions"]), [])


if __name__ == "__main__":
    unittest.main()

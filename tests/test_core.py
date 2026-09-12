import asyncio
import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from adaptkit import (
    ConfigurationError,
    FeedbackExtractor,
    InMemoryStore,
    LLMFeedbackExtractor,
    ObservationStatus,
    PreferenceEvent,
    Profile,
    ValidationError,
)


INTERACTION = {
    "context": "debugging",
    "action": "explanation_first",
    "previous_prompt": "Why does this fail?",
    "previous_response": "Here is a detailed explanation.",
    "user_message": "Just show me the fix.",
}


class CoreTests(unittest.TestCase):
    def test_profile_validates_configuration(self):
        with self.assertRaises(ValidationError):
            Profile(user_id="", actions=["a"])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=[])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a", "a"])
        with self.assertRaises(ValidationError):
            Profile(user_id="u", actions=["a"], implicit_threshold=0)
        with self.assertRaises(ConfigurationError):
            Profile(user_id="u", actions=["a"], learner="missing")

    def test_random_always_returns_allowed_action(self):
        profile = Profile(user_id="u", actions=["a", "b"], learner="random", seed=3)
        self.assertTrue(all(profile.choose("ctx") in {"a", "b"} for _ in range(30)))

    def test_explicit_and_pairwise_updates(self):
        profile = Profile(user_id="u", actions=["a", "b"])
        profile.like("ctx", "a")
        profile.dislike("ctx", "a")
        profile.prefer("ctx", preferred="a", rejected="b")
        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 3.0, "beta": 2.0})
        self.assertEqual(profile.state()["ctx"]["b"], {"alpha": 1.0, "beta": 2.0})

    def test_users_and_contexts_are_independent(self):
        store = InMemoryStore()
        first = Profile(user_id="first", actions=["a", "b"], store=store)
        second = Profile(user_id="second", actions=["a", "b"], store=store)
        first.like("debugging", "a")
        first.dislike("writing", "a")
        second.like("debugging", "b")
        self.assertEqual(first.state()["debugging"]["a"]["alpha"], 2)
        self.assertEqual(first.state()["writing"]["a"]["beta"], 2)
        self.assertNotIn("b", first.state()["debugging"])
        self.assertEqual(second.state()["debugging"]["b"]["alpha"], 2)

    def test_simultaneous_positive_updates_are_atomic(self):
        class RaceAmplifyingStore(InMemoryStore):
            """Makes the removed get-then-set implementation lose an update."""

            def __init__(self) -> None:
                super().__init__()
                self.concurrent_reads = threading.Barrier(2)

            def get(self, user_id: str, context: str, action: str) -> dict[str, float] | None:
                state = super().get(user_id, context, action)
                self.concurrent_reads.wait(timeout=5)
                return state

        store = RaceAmplifyingStore()
        profile = Profile(user_id="u", actions=["a", "b"], store=store)
        simultaneous_start = threading.Barrier(3)

        def positive_update() -> None:
            simultaneous_start.wait(timeout=5)
            profile.like("ctx", "a")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(positive_update) for _ in range(2)]
            simultaneous_start.wait(timeout=5)
            for future in futures:
                future.result(timeout=5)

        self.assertEqual(profile.state()["ctx"]["a"], {"alpha": 3.0, "beta": 1.0})

    def test_no_feedback_and_threshold_boundaries(self):
        outputs = iter(
            [
                {"has_feedback": False},
                {"has_feedback": True, "reward": 0.5, "confidence": 0.69},
                {"has_feedback": True, "reward": 0.5, "confidence": 0.7},
                {"has_feedback": True, "reward": -0.5, "confidence": 0.7},
            ]
        )
        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(judge=lambda _: next(outputs)),
        )
        self.assertEqual(profile.observe(**INTERACTION).status, ObservationStatus.NO_FEEDBACK)
        self.assertEqual(profile.observe(**INTERACTION).status, ObservationStatus.BELOW_THRESHOLD)
        self.assertEqual(profile.observe(**INTERACTION).status, ObservationStatus.UPDATED)
        self.assertEqual(profile.observe(**INTERACTION).status, ObservationStatus.UPDATED)
        self.assertEqual(
            profile.state()["debugging"]["explanation_first"],
            {"alpha": 2.0, "beta": 2.0},
        )
        self.assertNotIn("patch_first", profile.state()["debugging"])

    def test_malformed_output_fails_closed(self):
        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(judge=lambda _: {"has_feedback": True, "reward": 2}),
        )
        result = profile.observe(**INTERACTION)
        self.assertEqual(result.status, ObservationStatus.EVALUATOR_ERROR)
        self.assertFalse(result.updated)
        self.assertEqual(profile.state(), {})

    def test_custom_extractor_contract_violation_fails_closed(self):
        class InvalidExtractor(FeedbackExtractor):
            def extract(self, **interaction) -> PreferenceEvent:
                return cast(PreferenceEvent, {"has_feedback": True, "reward": 1})

        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=InvalidExtractor(),
        )
        result = profile.observe(**INTERACTION)
        self.assertEqual(result.status, ObservationStatus.EVALUATOR_ERROR)
        self.assertEqual(profile.state(), {})

    def test_evaluator_exception_fails_closed(self):
        def broken(_):
            raise TimeoutError("judge timed out")

        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(judge=broken),
        )
        result = profile.observe(**INTERACTION)
        self.assertIsNotNone(result.error)
        self.assertIn("TimeoutError", result.error or "")
        self.assertEqual(profile.state(), {})

    def test_conversation_text_is_not_stored_in_policy_state(self):
        sentinel = "SENTINEL-CONVERSATION-TEXT-92"
        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(
                judge=lambda _: {"has_feedback": True, "reward": 1, "confidence": 1}
            ),
        )
        profile.observe(
            context="debugging",
            action="patch_first",
            previous_prompt=sentinel,
            previous_response=sentinel,
            user_message=sentinel,
        )
        self.assertNotIn(sentinel, repr(profile.state()))

    def test_sync_and_async_share_update_semantics(self):
        event = {"has_feedback": True, "reward": -0.8, "confidence": 0.9}
        sync_profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(judge=lambda _: event),
        )

        async def judge(_):
            return event

        async_profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(async_judge=judge),
        )
        sync_result = sync_profile.observe(**INTERACTION)
        async_result = asyncio.run(async_profile.aobserve(**INTERACTION))
        self.assertEqual(sync_result.status, async_result.status)
        self.assertEqual(sync_profile.state(), async_profile.state())

    def test_async_cancellation_propagates_without_update(self):
        async def cancelled(_):
            raise asyncio.CancelledError

        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(async_judge=cancelled),
        )
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(profile.aobserve(**INTERACTION))
        self.assertEqual(profile.state(), {})

    def test_event_validation(self):
        with self.assertRaises(ValidationError):
            PreferenceEvent(has_feedback=False, reward=0)
        with self.assertRaises(ValidationError):
            PreferenceEvent(has_feedback=True, reward=math.nan)
        with self.assertRaises(ValidationError):
            PreferenceEvent(has_feedback=True, reward=1, confidence=math.inf)

    def test_observation_requires_evaluator(self):
        profile = Profile(user_id="u", actions=["a"])
        with self.assertRaises(ConfigurationError):
            profile.observe(
                context="ctx",
                action="a",
                previous_prompt="prompt",
                previous_response="response",
                user_message="message",
            )

    def test_sync_observation_with_async_only_judge_is_configuration_error(self):
        async def judge(_):
            return {"has_feedback": False}

        profile = Profile(
            user_id="u",
            actions=["patch_first", "explanation_first"],
            evaluator=LLMFeedbackExtractor(async_judge=judge),
        )
        with self.assertRaises(ConfigurationError):
            profile.observe(**INTERACTION)


if __name__ == "__main__":
    unittest.main()

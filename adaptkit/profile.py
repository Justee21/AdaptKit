from __future__ import annotations

import math
from collections.abc import Sequence

from .events import ObservationResult, ObservationStatus, PreferenceEvent
from .exceptions import ConfigurationError, ValidationError
from .feedback import FeedbackExtractor
from .learners import BaseLearner, RandomLearner, ThompsonLearner
from .storage import InMemoryStore, StateStore


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    return value


class Profile:
    def __init__(
        self,
        *,
        user_id: str,
        actions: Sequence[str],
        learner: str | BaseLearner = "thompson",
        evaluator: FeedbackExtractor | None = None,
        implicit_threshold: float = 0.35,
        seed: int | None = None,
        store: StateStore | None = None,
    ) -> None:
        self.user_id = _nonempty_string(user_id, "user_id")
        if isinstance(actions, (str, bytes)) or not actions:
            raise ValidationError("actions must be a non-empty sequence of strings")
        normalized = tuple(_nonempty_string(action, "action") for action in actions)
        if len(set(normalized)) != len(normalized):
            raise ValidationError("actions must be unique")
        if isinstance(implicit_threshold, bool) or not isinstance(implicit_threshold, (int, float)):
            raise ValidationError("implicit_threshold must be a number")
        if not math.isfinite(implicit_threshold) or not 0 < implicit_threshold <= 1:
            raise ValidationError("implicit_threshold must be finite and within (0, 1]")

        self.actions = normalized
        self.implicit_threshold = float(implicit_threshold)
        self.evaluator = evaluator
        self.store = (
            learner.store
            if isinstance(learner, BaseLearner) and store is None
            else store or InMemoryStore()
        )
        self.learner = self._build_learner(learner, seed)

    def _build_learner(self, learner: str | BaseLearner, seed: int | None) -> BaseLearner:
        if isinstance(learner, BaseLearner):
            if learner.store is not self.store:
                raise ConfigurationError("custom learner and profile must use the same store")
            return learner
        if learner == "thompson":
            return ThompsonLearner(self.store, seed=seed)
        if learner == "random":
            return RandomLearner(self.store, seed=seed)
        raise ConfigurationError(f"unknown learner: {learner!r}")

    def _context(self, context: str) -> str:
        return _nonempty_string(context, "context")

    def _action(self, action: str) -> str:
        action = _nonempty_string(action, "action")
        if action not in self.actions:
            raise ValidationError(f"unknown action: {action!r}")
        return action

    def choose(self, context: str) -> str:
        return self.learner.choose(self.user_id, self._context(context), self.actions)

    def like(self, context: str, action: str) -> None:
        self.learner.update(self.user_id, self._context(context), self._action(action), 1)

    def dislike(self, context: str, action: str) -> None:
        self.learner.update(self.user_id, self._context(context), self._action(action), -1)

    def prefer(self, context: str, preferred: str, rejected: str) -> None:
        context = self._context(context)
        preferred = self._action(preferred)
        rejected = self._action(rejected)
        if preferred == rejected:
            raise ValidationError("preferred and rejected actions must differ")
        self.learner.update(self.user_id, context, preferred, 1)
        self.learner.update(self.user_id, context, rejected, -1)

    def _interaction(
        self,
        *,
        context: str,
        action: str,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
    ) -> dict[str, object]:
        return {
            "context": self._context(context),
            "actions": self.actions,
            "action": self._action(action),
            "previous_prompt": _string(previous_prompt, "previous_prompt"),
            "previous_response": _string(previous_response, "previous_response"),
            "user_message": _string(user_message, "user_message"),
        }

    def _apply_event(self, context: str, action: str, event: PreferenceEvent) -> ObservationResult:
        if not event.has_feedback:
            return ObservationResult(ObservationStatus.NO_FEEDBACK, False, event)
        assert event.reward is not None
        effective = event.reward * event.confidence
        if effective >= self.implicit_threshold:
            self.learner.update(self.user_id, context, action, 1)
        elif effective <= -self.implicit_threshold:
            self.learner.update(self.user_id, context, action, -1)
        else:
            return ObservationResult(ObservationStatus.BELOW_THRESHOLD, False, event)
        return ObservationResult(ObservationStatus.UPDATED, True, event)

    @staticmethod
    def _validated_event(value: object) -> PreferenceEvent:
        if not isinstance(value, PreferenceEvent):
            raise ValidationError("extractor must return a PreferenceEvent")
        return value

    def observe(
        self,
        *,
        context: str,
        action: str,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
    ) -> ObservationResult:
        if self.evaluator is None:
            raise ConfigurationError("observe requires an evaluator")
        interaction = self._interaction(
            context=context,
            action=action,
            previous_prompt=previous_prompt,
            previous_response=previous_response,
            user_message=user_message,
        )
        try:
            event = self._validated_event(self.evaluator.extract(**interaction))
        except ConfigurationError:
            raise
        except Exception as exc:
            return ObservationResult(
                ObservationStatus.EVALUATOR_ERROR,
                False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._apply_event(str(interaction["context"]), str(interaction["action"]), event)

    async def aobserve(
        self,
        *,
        context: str,
        action: str,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
    ) -> ObservationResult:
        if self.evaluator is None:
            raise ConfigurationError("aobserve requires an evaluator")
        interaction = self._interaction(
            context=context,
            action=action,
            previous_prompt=previous_prompt,
            previous_response=previous_response,
            user_message=user_message,
        )
        try:
            event = self._validated_event(await self.evaluator.aextract(**interaction))
        except ConfigurationError:
            raise
        except Exception as exc:
            return ObservationResult(
                ObservationStatus.EVALUATOR_ERROR,
                False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._apply_event(str(interaction["context"]), str(interaction["action"]), event)

    def state(self) -> dict[str, dict[str, dict[str, float]]]:
        return self.store.snapshot(self.user_id)

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .events import (
    Decision,
    FeedbackSentiment,
    FeedbackTarget,
    LearningMode,
    LifecycleEvent,
    ObservationResult,
    ObservationStatus,
    PreferenceEvent,
    PromptPreference,
    SelectionSource,
)
from .exceptions import (
    ConfigurationError,
    DecisionMismatchError,
    DecisionNotFoundError,
    IdempotencyConflictError,
    ValidationError,
)
from .feedback import FeedbackExtractor
from .learners import BaseLearner, RandomLearner, ThompsonLearner
from .priors import BetaPrior
from .prompting import PromptPreferenceExtractor
from .storage import InMemoryStore, PolicyUpdateResult, StateStore, StoredObservation
from .storage.base import StateUpdater, same_observation_request

EventHook = Callable[[LifecycleEvent], None]


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be finite and positive")
    return float(value)


class Profile:
    """A per-user policy over mutually exclusive actions in discrete contexts."""

    def __init__(
        self,
        *,
        user_id: str,
        actions: Sequence[str],
        learner: str | BaseLearner = "thompson",
        evaluator: FeedbackExtractor | None = None,
        prompt_evaluator: PromptPreferenceExtractor | None = None,
        implicit_confidence_threshold: float = 0.70,
        passive_confidence_threshold: float | None = None,
        prompt_confidence_threshold: float = 0.90,
        implicit_learning_enabled: bool = True,
        learning_mode: str | LearningMode = LearningMode.ACTIVE,
        max_decision_age: timedelta | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        action_priors: Mapping[str, BetaPrior] | None = None,
        seed: int | None = None,
        store: StateStore | None = None,
        event_hook: EventHook | None = None,
    ) -> None:
        self.user_id = _nonempty_string(user_id, "user_id")
        if isinstance(actions, (str, bytes)) or not actions:
            raise ValidationError("actions must be a non-empty sequence of strings")
        normalized = tuple(_nonempty_string(action, "action") for action in actions)
        if len(set(normalized)) != len(normalized):
            raise ValidationError("actions must be unique")
        threshold = _positive_number(
            implicit_confidence_threshold, "implicit_confidence_threshold"
        )
        if threshold > 1:
            raise ValidationError("implicit_confidence_threshold must be within (0, 1]")
        passive_threshold = (
            threshold
            if passive_confidence_threshold is None
            else _positive_number(
                passive_confidence_threshold, "passive_confidence_threshold"
            )
        )
        if passive_threshold > 1:
            raise ValidationError("passive_confidence_threshold must be within (0, 1]")
        prompt_threshold = _positive_number(
            prompt_confidence_threshold, "prompt_confidence_threshold"
        )
        if prompt_threshold > 1:
            raise ValidationError("prompt_confidence_threshold must be within (0, 1]")
        if not isinstance(implicit_learning_enabled, bool):
            raise ValidationError("implicit_learning_enabled must be a boolean")
        try:
            normalized_mode = LearningMode(learning_mode)
        except (TypeError, ValueError) as exc:
            raise ValidationError("learning_mode must be active or shadow") from exc
        if max_decision_age is not None:
            if not isinstance(max_decision_age, timedelta) or max_decision_age <= timedelta(0):
                raise ValidationError("max_decision_age must be a positive timedelta or None")
        alpha = _positive_number(prior_alpha, "prior_alpha")
        beta = _positive_number(prior_beta, "prior_beta")
        normalized_priors = dict(action_priors or {})
        unknown_priors = set(normalized_priors) - set(normalized)
        if unknown_priors:
            raise ValidationError(f"action_priors contains unknown actions: {sorted(unknown_priors)}")
        if any(not isinstance(prior, BetaPrior) for prior in normalized_priors.values()):
            raise ValidationError("action_priors values must be BetaPrior instances")
        if normalized_priors and (
            isinstance(learner, BaseLearner) or learner != "thompson"
        ):
            raise ConfigurationError("action_priors requires the built-in Thompson learner")
        if event_hook is not None and not callable(event_hook):
            raise ValidationError("event_hook must be callable or None")

        self.actions = normalized
        self.implicit_confidence_threshold = threshold
        self.passive_confidence_threshold = passive_threshold
        self.prompt_confidence_threshold = prompt_threshold
        self.implicit_learning_enabled = implicit_learning_enabled
        self.learning_mode = normalized_mode
        self.max_decision_age = max_decision_age
        self.evaluator = evaluator
        self.prompt_evaluator = prompt_evaluator
        self.event_hook = event_hook
        self.store = (
            learner.store
            if isinstance(learner, BaseLearner) and store is None
            else store or InMemoryStore()
        )
        self.learner = self._build_learner(
            learner, seed, alpha, beta, normalized_priors
        )

    def _build_learner(
        self,
        learner: str | BaseLearner,
        seed: int | None,
        prior_alpha: float,
        prior_beta: float,
        action_priors: Mapping[str, BetaPrior],
    ) -> BaseLearner:
        if isinstance(learner, BaseLearner):
            if learner.store is not self.store:
                raise ConfigurationError("custom learner and profile must use the same store")
            return learner
        if learner == "thompson":
            return ThompsonLearner(
                self.store,
                seed=seed,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                action_priors=action_priors,
            )
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

    @staticmethod
    def _idempotency_key(value: str) -> str:
        return _nonempty_string(value, "idempotency_key")

    def _emit(
        self,
        *,
        name: str,
        context: str,
        action: str | None,
        decision_id: str | None,
        status: str,
        version_before: int | None,
        version_after: int | None,
    ) -> None:
        if self.event_hook is None:
            return
        event = LifecycleEvent(
            name=name,
            user_id=self.user_id,
            context=context,
            action=action,
            decision_id=decision_id,
            status=status,
            policy_version_before=version_before,
            policy_version_after=version_after,
            created_at=datetime.now(timezone.utc),
        )
        try:
            self.event_hook(event)
        except Exception:
            return

    def _prompt_interaction(self, context: str, prompt: str) -> dict[str, object]:
        return {
            "context": context,
            "prompt": _string(prompt, "prompt"),
            "actions": self.actions,
        }

    def _decision(
        self,
        *,
        context: str,
        action: str,
        version: int,
        preference: PromptPreference | None,
        route_status: str | None,
    ) -> Decision:
        selection_source = (
            SelectionSource.PROMPT_OVERRIDE
            if preference is not None
            else SelectionSource.POLICY
        )
        decision = Decision(
            decision_id=str(uuid4()),
            user_id=self.user_id,
            context=context,
            action=action,
            policy_version=version,
            created_at=datetime.now(timezone.utc),
            selection_source=selection_source,
            selection_confidence=(
                None if preference is None else preference.confidence
            ),
        )
        self.store.create_decision(decision)
        self._emit(
            name="decision_created",
            context=context,
            action=action,
            decision_id=decision.decision_id,
            status="created" if route_status is None else route_status,
            version_before=version,
            version_after=version,
        )
        if route_status is not None:
            self._emit(
                name="prompt_preference_evaluated",
                context=context,
                action=action,
                decision_id=decision.decision_id,
                status=route_status,
                version_before=version,
                version_after=version,
            )
        return decision

    def _route_prompt(
        self, context: str, prompt: str | None
    ) -> tuple[PromptPreference | None, str | None]:
        if prompt is None:
            return None, None
        prompt = _string(prompt, "prompt")
        if not prompt.strip():
            return None, "prompt_no_clear_cue"
        if self.prompt_evaluator is None:
            raise ConfigurationError("choose(prompt=...) requires a prompt_evaluator")
        try:
            extracted = self.prompt_evaluator.extract(
                **self._prompt_interaction(context, prompt)
            )
            return self._validated_prompt_route(extracted)
        except ConfigurationError:
            raise
        except Exception as exc:
            return None, f"prompt_evaluator_error:{type(exc).__name__}"

    def _validated_prompt_route(
        self, value: object
    ) -> tuple[PromptPreference | None, str]:
        if not isinstance(value, PromptPreference):
            raise ValidationError("prompt extractor must return a PromptPreference")
        if value.action is None:
            return None, "prompt_no_clear_cue"
        self._action(value.action)
        if value.confidence < self.prompt_confidence_threshold:
            return None, "prompt_below_threshold"
        return value, "prompt_override"

    def _complete_choice(
        self,
        context: str,
        preference: PromptPreference | None,
        route_status: str | None,
    ) -> Decision:
        if preference is None:
            action, version = self.learner.choose(self.user_id, context, self.actions)
        else:
            assert preference.action is not None
            action = preference.action
            version = self.learner.snapshot(
                self.user_id, context, self.actions
            ).version
        return self._decision(
            context=context,
            action=action,
            version=version,
            preference=preference,
            route_status=route_status,
        )

    def choose(self, context: str, *, prompt: str | None = None) -> Decision:
        context = self._context(context)
        preference, route_status = self._route_prompt(context, prompt)
        return self._complete_choice(context, preference, route_status)

    async def achoose(self, context: str, *, prompt: str | None = None) -> Decision:
        context = self._context(context)
        if prompt is None:
            return self.choose(context)
        prompt = _string(prompt, "prompt")
        if not prompt.strip():
            return self._complete_choice(context, None, "prompt_no_clear_cue")
        if self.prompt_evaluator is None:
            raise ConfigurationError("achoose(prompt=...) requires a prompt_evaluator")
        try:
            extracted = await self.prompt_evaluator.aextract(
                **self._prompt_interaction(context, prompt)
            )
            preference, route_status = self._validated_prompt_route(extracted)
        except ConfigurationError:
            raise
        except Exception as exc:
            preference = None
            route_status = f"prompt_evaluator_error:{type(exc).__name__}"
        return self._complete_choice(context, preference, route_status)

    def _validate_decision(self, decision: Decision) -> Decision:
        if not isinstance(decision, Decision):
            raise ValidationError("decision must be a Decision")
        if decision.user_id != self.user_id:
            raise DecisionMismatchError("decision belongs to another user")
        self.learner.snapshot(self.user_id, decision.context, self.actions)
        self._action(decision.action)
        stored = self.store.get_decision(decision.decision_id)
        if stored is None:
            raise DecisionNotFoundError(f"unknown decision: {decision.decision_id}")
        if stored != decision:
            raise DecisionMismatchError("decision does not match its persisted record")
        return decision

    def _validate_decision_age(self, decision: Decision) -> None:
        if self.max_decision_age is not None:
            age = datetime.now(timezone.utc) - decision.created_at
            if age > self.max_decision_age:
                raise ValidationError("decision is older than max_decision_age")

    @staticmethod
    def _duplicate_result(observation: StoredObservation) -> ObservationResult:
        return ObservationResult(
            status=ObservationStatus.DUPLICATE,
            updated=False,
            recorded=True,
            duplicate=True,
            event=observation.event,
            observation_id=observation.observation_id,
            policy_version_before=observation.policy_version_before,
            policy_version_after=observation.policy_version_after,
        )

    def _existing_result(
        self,
        decision: Decision,
        idempotency_key: str,
        *,
        event: PreferenceEvent | None = None,
        learning_mode: LearningMode | None = None,
        status: ObservationStatus | None = None,
    ) -> ObservationResult | None:
        existing = self.store.find_observation(decision.decision_id, idempotency_key)
        if existing is None:
            return None
        if event is not None:
            assert learning_mode is not None and status is not None
            if not same_observation_request(existing, event, learning_mode, status):
                raise IdempotencyConflictError(
                    "idempotency key already identifies a different feedback operation"
                )
        result = self._duplicate_result(existing)
        self._emit(
            name="observation_deduplicated",
            context=decision.context,
            action=decision.action,
            decision_id=decision.decision_id,
            status=result.status.value,
            version_before=result.policy_version_before,
            version_after=result.policy_version_after,
        )
        return result

    def _effective_mode(self, apply: bool | None) -> LearningMode:
        if apply is not None and not isinstance(apply, bool):
            raise ValidationError("apply must be a boolean or None")
        if apply is True:
            return LearningMode.ACTIVE
        if apply is False:
            return LearningMode.SHADOW
        return self.learning_mode

    def _event_plan(
        self, event: PreferenceEvent, apply: bool | None
    ) -> tuple[LearningMode, ObservationStatus, StateUpdater | None]:
        mode = self._effective_mode(apply)
        updater = None
        if not event.has_preference_signal:
            status = ObservationStatus.NO_PREFERENCE_SIGNAL
        elif event.source == "implicit" and not self.implicit_learning_enabled:
            status = ObservationStatus.LEARNER_IGNORED
        elif (
            event.source in {"implicit", "passive"}
            and event.confidence
            < (
                self.passive_confidence_threshold
                if event.source == "passive"
                else self.implicit_confidence_threshold
            )
        ):
            status = ObservationStatus.BELOW_THRESHOLD
        elif mode is LearningMode.SHADOW:
            status = ObservationStatus.SHADOW
        else:
            assert event.reward is not None
            updater = self.learner.updater(event.reward)
            status = (
                ObservationStatus.UPDATED
                if updater is not None
                else ObservationStatus.LEARNER_IGNORED
            )
        return mode, status, updater

    def _record_event(
        self,
        *,
        decision: Decision,
        idempotency_key: str,
        event: PreferenceEvent,
        apply: bool | None,
    ) -> ObservationResult:
        mode, status, updater = self._event_plan(event, apply)

        stored = self.store.record_observation(
            decision=decision,
            observation_id=str(uuid4()),
            idempotency_key=idempotency_key,
            event=event,
            learning_mode=mode,
            status=status,
            initial_state=self.learner.initial_state(decision.action),
            updater=updater,
        )
        if stored.duplicate:
            return self._duplicate_result(stored)
        result = ObservationResult(
            status=status,
            updated=stored.applied,
            recorded=True,
            event=event,
            observation_id=stored.observation_id,
            policy_version_before=stored.policy_version_before,
            policy_version_after=stored.policy_version_after,
        )
        self._emit(
            name="observation_recorded",
            context=decision.context,
            action=decision.action,
            decision_id=decision.decision_id,
            status=status.value,
            version_before=stored.policy_version_before,
            version_after=stored.policy_version_after,
        )
        return result

    def _record_direct_event(
        self,
        *,
        decision: Decision,
        idempotency_key: str,
        event: PreferenceEvent,
        apply: bool | None,
    ) -> ObservationResult:
        mode, status, _ = self._event_plan(event, apply)
        existing = self._existing_result(
            decision,
            idempotency_key,
            event=event,
            learning_mode=mode,
            status=status,
        )
        if existing is not None:
            return existing
        self._validate_decision_age(decision)
        return self._record_event(
            decision=decision,
            idempotency_key=idempotency_key,
            event=event,
            apply=apply,
        )

    def _interaction(
        self,
        decision: Decision,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
    ) -> dict[str, object]:
        return {
            "context": decision.context,
            "actions": self.actions,
            "action": decision.action,
            "previous_prompt": _string(previous_prompt, "previous_prompt"),
            "previous_response": _string(previous_response, "previous_response"),
            "user_message": _string(user_message, "user_message"),
        }

    @staticmethod
    def _validated_event(value: object) -> PreferenceEvent:
        if not isinstance(value, PreferenceEvent):
            raise ValidationError("extractor must return a PreferenceEvent")
        return value

    def observe(
        self,
        *,
        decision: Decision,
        idempotency_key: str,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
        apply: bool | None = None,
    ) -> ObservationResult:
        if self.evaluator is None:
            raise ConfigurationError("observe requires an evaluator")
        decision = self._validate_decision(decision)
        key = self._idempotency_key(idempotency_key)
        existing = self._existing_result(decision, key)
        if existing is not None:
            return existing
        self._validate_decision_age(decision)
        interaction = self._interaction(
            decision, previous_prompt, previous_response, user_message
        )
        try:
            event = self._validated_event(self.evaluator.extract(**interaction))
        except ConfigurationError:
            raise
        except Exception as exc:
            self._emit(
                name="evaluator_failed",
                context=decision.context,
                action=decision.action,
                decision_id=decision.decision_id,
                status=type(exc).__name__,
                version_before=decision.policy_version,
                version_after=decision.policy_version,
            )
            return ObservationResult(
                ObservationStatus.EVALUATOR_ERROR,
                False,
                error=type(exc).__name__,
            )
        return self._record_event(
            decision=decision,
            idempotency_key=key,
            event=event,
            apply=apply,
        )

    async def aobserve(
        self,
        *,
        decision: Decision,
        idempotency_key: str,
        previous_prompt: str,
        previous_response: str,
        user_message: str,
        apply: bool | None = None,
    ) -> ObservationResult:
        if self.evaluator is None:
            raise ConfigurationError("aobserve requires an evaluator")
        decision = self._validate_decision(decision)
        key = self._idempotency_key(idempotency_key)
        existing = self._existing_result(decision, key)
        if existing is not None:
            return existing
        self._validate_decision_age(decision)
        interaction = self._interaction(
            decision, previous_prompt, previous_response, user_message
        )
        try:
            event = self._validated_event(await self.evaluator.aextract(**interaction))
        except ConfigurationError:
            raise
        except Exception as exc:
            self._emit(
                name="evaluator_failed",
                context=decision.context,
                action=decision.action,
                decision_id=decision.decision_id,
                status=type(exc).__name__,
                version_before=decision.policy_version,
                version_after=decision.policy_version,
            )
            return ObservationResult(
                ObservationStatus.EVALUATOR_ERROR,
                False,
                error=type(exc).__name__,
            )
        return self._record_event(
            decision=decision,
            idempotency_key=key,
            event=event,
            apply=apply,
        )

    def like(
        self,
        decision: Decision,
        *,
        idempotency_key: str,
        apply: bool | None = None,
    ) -> ObservationResult:
        decision = self._validate_decision(decision)
        key = self._idempotency_key(idempotency_key)
        return self._record_direct_event(
            decision=decision,
            idempotency_key=key,
            event=PreferenceEvent(
                FeedbackTarget.BEHAVIOR,
                FeedbackSentiment.POSITIVE,
                source="explicit",
            ),
            apply=apply,
        )

    def dislike(
        self,
        decision: Decision,
        *,
        idempotency_key: str,
        apply: bool | None = None,
    ) -> ObservationResult:
        decision = self._validate_decision(decision)
        key = self._idempotency_key(idempotency_key)
        return self._record_direct_event(
            decision=decision,
            idempotency_key=key,
            event=PreferenceEvent(
                FeedbackTarget.BEHAVIOR,
                FeedbackSentiment.NEGATIVE,
                source="explicit",
            ),
            apply=apply,
        )

    def signal(
        self,
        decision: Decision,
        *,
        signal_type: str,
        sentiment: str | FeedbackSentiment,
        confidence: float = 1.0,
        metadata: Mapping[str, object] | None = None,
        idempotency_key: str,
        apply: bool | None = None,
    ) -> ObservationResult:
        decision = self._validate_decision(decision)
        key = self._idempotency_key(idempotency_key)
        try:
            normalized_sentiment = FeedbackSentiment(sentiment)
        except (TypeError, ValueError) as exc:
            raise ValidationError("sentiment must be positive or negative") from exc
        if normalized_sentiment is FeedbackSentiment.NONE:
            raise ValidationError("sentiment must be positive or negative")
        event = PreferenceEvent(
            target=FeedbackTarget.BEHAVIOR,
            sentiment=normalized_sentiment,
            confidence=confidence,
            source="passive",
            signal_type=signal_type,
            metadata=dict(metadata or {}),
        )
        return self._record_direct_event(
            decision=decision,
            idempotency_key=key,
            event=event,
            apply=apply,
        )

    def prefer(
        self,
        context: str,
        *,
        preferred: str,
        rejected: str,
        idempotency_key: str,
        apply: bool | None = None,
    ) -> PolicyUpdateResult:
        context = self._context(context)
        preferred = self._action(preferred)
        rejected = self._action(rejected)
        key = self._idempotency_key(idempotency_key)
        if preferred == rejected:
            raise ValidationError("preferred and rejected actions must differ")
        mode = self._effective_mode(apply)
        updaters = {}
        if mode is LearningMode.ACTIVE:
            positive = self.learner.updater(1)
            negative = self.learner.updater(0)
            if positive is not None:
                updaters[preferred] = positive
            if negative is not None:
                updaters[rejected] = negative
        result = self.store.atomic_policy_update(
            user_id=self.user_id,
            context=context,
            idempotency_key=key,
            operation_identity={
                "operation": "prefer",
                "preferred": preferred,
                "rejected": rejected,
                "learning_mode": mode.value,
            },
            initial_states=self.learner.initial_states(self.actions),
            updaters=updaters,
        )
        self._emit(
            name="pairwise_preference",
            context=context,
            action=None,
            decision_id=None,
            status=(
                "duplicate"
                if result.duplicate
                else "shadow"
                if mode is LearningMode.SHADOW
                else "updated"
                if result.updated
                else "learner_ignored"
            ),
            version_before=result.policy_version_before,
            version_after=result.policy_version_after,
        )
        return result

    def set_learning_mode(self, mode: str | LearningMode) -> None:
        try:
            self.learning_mode = LearningMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValidationError("learning mode must be active or shadow") from exc

    def policy(self, context: str) -> dict[str, object]:
        context = self._context(context)
        snapshot = self.store.policy_snapshot(
            self.user_id,
            context,
            self.actions,
            self.learner.initial_states(self.actions),
        )
        actions: dict[str, dict[str, float]] = {}
        for action, state in snapshot.states.items():
            alpha = state["alpha"]
            beta = state["beta"]
            actions[action] = {
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": alpha / (alpha + beta),
            }
        return {"version": snapshot.version, "actions": actions}

    def state(self) -> dict[str, dict[str, dict[str, float]]]:
        return self.store.snapshot(self.user_id)

    def export_user(self) -> dict[str, Any]:
        return self.store.export_user(self.user_id)

    def delete_user(self) -> None:
        self.store.delete_user(self.user_id)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from adaptkit.events import Decision, LearningMode, ObservationStatus, PreferenceEvent
from adaptkit.exceptions import DecisionMismatchError, DecisionNotFoundError

from .base import (
    PolicySnapshot,
    PolicyUpdateResult,
    StateStore,
    StateUpdater,
    StoredObservation,
)


def _persisted_event(event: PreferenceEvent) -> PreferenceEvent:
    return PreferenceEvent(
        target=event.target,
        sentiment=event.sentiment,
        confidence=event.confidence,
        source=event.source,
    )


class InMemoryStore(StateStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str, str], dict[str, float]] = {}
        self._versions: dict[tuple[str, str], int] = {}
        self._decisions: dict[str, Decision] = {}
        self._observations: dict[tuple[str, str], StoredObservation] = {}
        self._operations: dict[tuple[str, str, str], PolicyUpdateResult] = {}
        self._lock = RLock()

    def policy_snapshot(
        self,
        user_id: str,
        context: str,
        actions: Sequence[str],
        initial_state: Mapping[str, float],
    ) -> PolicySnapshot:
        with self._lock:
            states = {
                action: dict(self._states.get((user_id, context, action), initial_state))
                for action in actions
            }
            return PolicySnapshot(self._versions.get((user_id, context), 0), states)

    def create_decision(self, decision: Decision) -> None:
        with self._lock:
            existing = self._decisions.get(decision.decision_id)
            if existing is not None and existing != decision:
                raise DecisionMismatchError("decision_id already identifies another decision")
            self._decisions[decision.decision_id] = decision

    def get_decision(self, decision_id: str) -> Decision | None:
        with self._lock:
            return self._decisions.get(decision_id)

    def find_observation(
        self, decision_id: str, idempotency_key: str
    ) -> StoredObservation | None:
        with self._lock:
            observation = self._observations.get((decision_id, idempotency_key))
            return None if observation is None else replace(observation, duplicate=True)

    def _validated_decision(self, decision: Decision) -> None:
        stored = self._decisions.get(decision.decision_id)
        if stored is None:
            raise DecisionNotFoundError(f"unknown decision: {decision.decision_id}")
        if stored != decision:
            raise DecisionMismatchError("decision does not match its persisted record")

    def record_observation(
        self,
        *,
        decision: Decision,
        observation_id: str,
        idempotency_key: str,
        event: PreferenceEvent,
        learning_mode: LearningMode,
        status: ObservationStatus,
        initial_state: Mapping[str, float],
        updater: StateUpdater | None,
    ) -> StoredObservation:
        with self._lock:
            self._validated_decision(decision)
            key = (decision.decision_id, idempotency_key)
            existing = self._observations.get(key)
            if existing is not None:
                return replace(existing, duplicate=True)

            policy_key = (decision.user_id, decision.context)
            version_before = self._versions.get(policy_key, 0)
            version_after = version_before
            if updater is not None:
                state_key = (decision.user_id, decision.context, decision.action)
                state = dict(self._states.get(state_key, initial_state))
                updater(state)
                self._states[state_key] = state
                version_after += 1
                self._versions[policy_key] = version_after

            observation = StoredObservation(
                observation_id=observation_id,
                decision_id=decision.decision_id,
                idempotency_key=idempotency_key,
                event=_persisted_event(event),
                learning_mode=learning_mode,
                status=status,
                applied=updater is not None,
                policy_version_before=version_before,
                policy_version_after=version_after,
                created_at=datetime.now(timezone.utc),
            )
            self._observations[key] = observation
            return observation

    def atomic_policy_update(
        self,
        *,
        user_id: str,
        context: str,
        idempotency_key: str,
        initial_state: Mapping[str, float],
        updaters: Mapping[str, StateUpdater],
    ) -> PolicyUpdateResult:
        with self._lock:
            operation_key = (user_id, context, idempotency_key)
            version_before = self._versions.get((user_id, context), 0)
            existing = self._operations.get(operation_key)
            if existing is not None:
                return replace(existing, updated=False, duplicate=True)

            for action, updater in updaters.items():
                state_key = (user_id, context, action)
                state = dict(self._states.get(state_key, initial_state))
                updater(state)
                self._states[state_key] = state
            version_after = version_before + bool(updaters)
            if updaters:
                self._versions[(user_id, context)] = version_after
            result = PolicyUpdateResult(bool(updaters), False, version_before, version_after)
            self._operations[operation_key] = result
            return result

    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        with self._lock:
            for (stored_user, context, action), state in self._states.items():
                if stored_user == user_id:
                    result.setdefault(context, {})[action] = dict(state)
        return result

    def export_user(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            policies = [
                {"context": context, "version": version}
                for (stored_user, context), version in sorted(self._versions.items())
                if stored_user == user_id
            ]
            decisions = [
                {
                    "decision_id": decision.decision_id,
                    "context": decision.context,
                    "action": decision.action,
                    "policy_version": decision.policy_version,
                    "created_at": decision.created_at.isoformat(),
                }
                for decision in sorted(self._decisions.values(), key=lambda item: item.decision_id)
                if decision.user_id == user_id
            ]
            decision_ids = {item["decision_id"] for item in decisions}
            observations = [
                {
                    "observation_id": observation.observation_id,
                    "decision_id": observation.decision_id,
                    "idempotency_key": observation.idempotency_key,
                    "target": observation.event.target.value,
                    "sentiment": observation.event.sentiment.value,
                    "confidence": observation.event.confidence,
                    "source": observation.event.source,
                    "learning_mode": observation.learning_mode.value,
                    "status": observation.status.value,
                    "applied": observation.applied,
                    "policy_version_before": observation.policy_version_before,
                    "policy_version_after": observation.policy_version_after,
                    "created_at": observation.created_at.isoformat(),
                }
                for observation in sorted(
                    self._observations.values(), key=lambda item: item.observation_id
                )
                if observation.decision_id in decision_ids
            ]
            return {
                "schema_version": self.schema_version,
                "user_id": user_id,
                "policies": policies,
                "policy_state": self.snapshot(user_id),
                "decisions": decisions,
                "observations": observations,
            }

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            decision_ids = {
                decision_id
                for decision_id, decision in self._decisions.items()
                if decision.user_id == user_id
            }
            self._states = {
                key: value for key, value in self._states.items() if key[0] != user_id
            }
            self._versions = {
                key: value for key, value in self._versions.items() if key[0] != user_id
            }
            self._decisions = {
                key: value for key, value in self._decisions.items() if key not in decision_ids
            }
            self._observations = {
                key: value
                for key, value in self._observations.items()
                if key[0] not in decision_ids
            }
            self._operations = {
                key: value for key, value in self._operations.items() if key[0] != user_id
            }

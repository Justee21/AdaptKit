from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from .base import StateStore


class InMemoryStore(StateStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str, str], dict[str, float]] = {}
        self._lock = RLock()

    def get(self, user_id: str, context: str, action: str) -> dict[str, float] | None:
        with self._lock:
            state = self._states.get((user_id, context, action))
            return None if state is None else dict(state)

    def set(
        self,
        user_id: str,
        context: str,
        action: str,
        state: Mapping[str, float],
    ) -> None:
        with self._lock:
            self._states[(user_id, context, action)] = dict(state)

    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        with self._lock:
            for (stored_user, context, action), state in self._states.items():
                if stored_user == user_id:
                    result.setdefault(context, {})[action] = dict(state)
        return result

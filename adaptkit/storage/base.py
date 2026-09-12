from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class StateStore(ABC):
    @abstractmethod
    def get(self, user_id: str, context: str, action: str) -> dict[str, float] | None:
        """Return a copy of the learner state, or None when it does not exist."""

    @abstractmethod
    def set(
        self,
        user_id: str,
        context: str,
        action: str,
        state: Mapping[str, float],
    ) -> None:
        """Store learner state for one user/context/action tuple."""

    @abstractmethod
    def snapshot(self, user_id: str) -> dict[str, dict[str, dict[str, float]]]:
        """Return a detached snapshot of all learner state for a user."""

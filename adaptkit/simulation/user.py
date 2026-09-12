from __future__ import annotations

import math
import random
from collections.abc import Mapping

from adaptkit.exceptions import ValidationError


class SimulatedUser:
    """Bernoulli user whose hidden success probabilities vary by context and action."""

    def __init__(
        self,
        preferences: Mapping[str, Mapping[str, float]],
        *,
        seed: int | None = None,
    ) -> None:
        if not preferences:
            raise ValidationError("preferences must not be empty")
        self._preferences: dict[str, dict[str, float]] = {}
        for context, action_rewards in preferences.items():
            if not isinstance(context, str) or not context.strip() or not action_rewards:
                raise ValidationError("each context must be non-empty and define actions")
            normalized: dict[str, float] = {}
            for action, probability in action_rewards.items():
                if not isinstance(action, str) or not action.strip():
                    raise ValidationError("action names must be non-empty strings")
                if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                    raise ValidationError("success probabilities must be numbers")
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValidationError("success probabilities must be within [0, 1]")
                normalized[action] = float(probability)
            self._preferences[context] = normalized
        self._rng = random.Random(seed)

    def expected_reward(self, context: str, action: str) -> float:
        try:
            return self._preferences[context][action]
        except KeyError as exc:
            raise ValidationError(f"unknown simulated context/action: {context!r}/{action!r}") from exc

    def best_action(self, context: str) -> str:
        try:
            rewards = self._preferences[context]
        except KeyError as exc:
            raise ValidationError(f"unknown simulated context: {context!r}") from exc
        return max(rewards, key=rewards.__getitem__)

    def react(self, context: str, action: str) -> int:
        """Return 1 for success and 0 for failure."""
        return int(self._rng.random() < self.expected_reward(context, action))

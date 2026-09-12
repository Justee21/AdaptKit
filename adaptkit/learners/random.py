from __future__ import annotations

import random
from collections.abc import Sequence

from .base import BaseLearner


class RandomLearner(BaseLearner):
    def __init__(self, store, *, seed: int | None = None) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)

    def choose(self, user_id: str, context: str, actions: Sequence[str]) -> str:
        return self._rng.choice(actions)

    def update(self, user_id: str, context: str, action: str, reward: int) -> None:
        return None

from __future__ import annotations

import random
from collections.abc import Sequence

from .base import BaseLearner


class ThompsonLearner(BaseLearner):
    def __init__(self, store, *, seed: int | None = None) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)

    def _state(self, user_id: str, context: str, action: str) -> dict[str, float]:
        state = self.store.get(user_id, context, action)
        return {"alpha": 1.0, "beta": 1.0} if state is None else state

    def choose(self, user_id: str, context: str, actions: Sequence[str]) -> str:
        samples = {}
        for action in actions:
            state = self._state(user_id, context, action)
            samples[action] = self._rng.betavariate(state["alpha"], state["beta"])
        return max(actions, key=samples.__getitem__)

    def update(self, user_id: str, context: str, action: str, reward: int) -> None:
        if reward not in (-1, 1):
            raise ValueError("reward must be 1 or -1")
        state = self._state(user_id, context, action)
        if reward == 1:
            state["alpha"] += 1
        else:
            state["beta"] += 1
        self.store.set(user_id, context, action, state)

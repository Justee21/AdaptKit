from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from threading import Lock

from adaptkit.storage.base import StateUpdater
from adaptkit.storage import StateStore
from adaptkit.priors import BetaPrior

from .base import BaseLearner


class ThompsonLearner(BaseLearner):
    def __init__(
        self,
        store: StateStore,
        *,
        seed: int | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        action_priors: Mapping[str, BetaPrior] | None = None,
    ) -> None:
        super().__init__(store)
        self._rng = random.Random(seed)
        self._rng_lock = Lock()
        self._default_prior = BetaPrior(prior_alpha, prior_beta)
        self._action_priors = dict(action_priors or {})

    def initial_state(self, action: str) -> Mapping[str, float]:
        return self._action_priors.get(action, self._default_prior).state()

    def select(
        self, states: Mapping[str, Mapping[str, float]], actions: Sequence[str]
    ) -> str:
        with self._rng_lock:
            samples = {
                action: self._rng.betavariate(
                    states[action]["alpha"], states[action]["beta"]
                )
                for action in actions
            }
        return max(actions, key=samples.__getitem__)

    def updater(self, reward: int) -> StateUpdater:
        if isinstance(reward, bool) or reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")

        def apply(state: dict[str, float]) -> None:
            state["alpha"] += reward
            state["beta"] += 1 - reward

        return apply

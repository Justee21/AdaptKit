from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationStep:
    reward: int
    expected_reward: float
    optimal_expected_reward: float
    chose_optimal: bool

    @property
    def regret(self) -> float:
        return self.optimal_expected_reward - self.expected_reward


def cumulative(values: list[float | int]) -> list[float]:
    total = 0.0
    result: list[float] = []
    for value in values:
        total += value
        result.append(total)
    return result


def summarize(steps: list[SimulationStep]) -> dict[str, float]:
    if not steps:
        raise ValueError("steps must not be empty")
    total_reward = sum(step.reward for step in steps)
    return {
        "average_reward": total_reward / len(steps),
        "cumulative_reward": float(total_reward),
        "cumulative_regret": sum(step.regret for step in steps),
        "optimal_action_rate": sum(step.chose_optimal for step in steps) / len(steps),
    }

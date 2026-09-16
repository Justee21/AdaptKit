from __future__ import annotations

import math
from typing import Any


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total == 0:
        return {"successes": successes, "total": total, "lower": None, "upper": None}
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def feedback_confidence_intervals(
    expected: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    learning_threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    counts = {
        "feedback_detection_accuracy": [0, len(expected)],
        "behavior_precision": [0, 0],
        "behavior_recall": [0, 0],
        "applied_behavior_precision": [0, 0],
        "applied_behavior_recall": [0, 0],
        "direction_accuracy": [0, 0],
        "false_positive_rate": [0, 0],
        "false_positive_learning_rate": [0, 0],
        "false_direction_learning_rate": [0, 0],
    }
    for row, prediction in zip(expected, predictions):
        expected_feedback = bool(row["expected_has_feedback"])
        if prediction.get("invalid"):
            if expected_feedback:
                counts["behavior_recall"][1] += 1
                counts["applied_behavior_recall"][1] += 1
                counts["direction_accuracy"][1] += 1
            else:
                counts["false_positive_rate"][1] += 1
                counts["false_positive_learning_rate"][1] += 1
            continue
        predicted_feedback = prediction["target"] == "behavior"
        applied = predicted_feedback and prediction.get("confidence", 1.0) >= learning_threshold
        counts["feedback_detection_accuracy"][0] += predicted_feedback == expected_feedback
        if predicted_feedback:
            counts["behavior_precision"][1] += 1
            counts["behavior_precision"][0] += expected_feedback
        if applied:
            counts["applied_behavior_precision"][1] += 1
            counts["applied_behavior_precision"][0] += expected_feedback
        if expected_feedback:
            counts["behavior_recall"][1] += 1
            counts["behavior_recall"][0] += predicted_feedback
            counts["applied_behavior_recall"][1] += 1
            counts["applied_behavior_recall"][0] += applied
            counts["direction_accuracy"][1] += 1
            counts["direction_accuracy"][0] += (
                predicted_feedback
                and prediction["sentiment"] == row["expected_direction"]
            )
            if applied:
                counts["false_direction_learning_rate"][1] += 1
                counts["false_direction_learning_rate"][0] += (
                    prediction["sentiment"] != row["expected_direction"]
                )
        else:
            counts["false_positive_rate"][1] += 1
            counts["false_positive_rate"][0] += predicted_feedback
            counts["false_positive_learning_rate"][1] += 1
            counts["false_positive_learning_rate"][0] += applied
    return {
        name: wilson_interval(int(values[0]), int(values[1]))
        for name, values in counts.items()
    }

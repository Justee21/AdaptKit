from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from adaptkit import LLMFeedbackExtractor


def conservative_rules(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Offline baseline for exercising the evaluator pipeline; it is not an LLM."""
    payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
    message = payload["latest_user_message"].lower()
    positive = ("perfect", "exactly", "very helpful", "great", "love that")
    negative = (
        "too much",
        "too terse",
        "please don't",
        "stop making",
        "just show",
        "i prefer",
        "from now on",
        "act immediately",
    )
    if any(phrase in message for phrase in positive):
        return {"target": "behavior", "sentiment": "positive", "confidence": 0.9}
    if any(phrase in message for phrase in negative):
        return {"target": "behavior", "sentiment": "negative", "confidence": 0.9}
    if any(
        phrase in message
        for phrase in ("weather tomorrow", "now help me", "let's work on", "deployment checklist")
    ):
        return {"target": "unrelated", "sentiment": "none", "confidence": 0.9}
    if any(
        phrase in message
        for phrase in ("actually, that feature", "incident was", "it uses put", "total should")
    ):
        return {"target": "answer_content", "sentiment": "none", "confidence": 0.9}
    if message.endswith("?"):
        return {"target": "task_continuation", "sentiment": "none", "confidence": 0.9}
    return {"target": "unrelated", "sentiment": "none", "confidence": 0.9}


def load_judge(spec: str | None) -> tuple[Callable, str]:
    if spec is None:
        return conservative_rules, "offline_conservative_rules"
    if ":" not in spec:
        raise ValueError("--judge must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function, spec


def score(
    expected: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    learning_threshold: float = 0.70,
) -> dict[str, float | int | None]:
    if len(expected) != len(predictions) or not expected:
        raise ValueError("expected and predictions must have the same nonzero length")
    detection_correct = 0
    target_correct = 0
    feedback_total = 0
    direction_correct = 0
    non_feedback_total = 0
    false_positives = 0
    false_positive_learning_updates = 0
    true_behavior_predictions = 0
    behavior_predictions = 0
    false_direction_updates = 0
    applied_behavior_predictions = 0
    true_applied_behavior_predictions = 0
    false_direction_learning_updates = 0
    invalid = 0
    for row, prediction in zip(expected, predictions):
        expected_feedback = row["expected_has_feedback"]
        if prediction.get("invalid"):
            invalid += 1
            if expected_feedback:
                feedback_total += 1
            else:
                non_feedback_total += 1
            continue
        target = prediction["target"]
        predicted_feedback = target == "behavior"
        predicted_direction = prediction["sentiment"]
        applied_behavior = (
            predicted_feedback
            and prediction.get("confidence", 1.0) >= learning_threshold
        )
        behavior_predictions += predicted_feedback
        applied_behavior_predictions += applied_behavior
        detection_correct += predicted_feedback == expected_feedback
        target_correct += target == row["expected_target"]
        if expected_feedback:
            feedback_total += 1
            true_behavior_predictions += predicted_feedback
            true_applied_behavior_predictions += applied_behavior
            direction_correct += predicted_feedback and predicted_direction == row["expected_direction"]
            false_direction_updates += (
                predicted_feedback and predicted_direction != row["expected_direction"]
            )
            false_direction_learning_updates += (
                applied_behavior and predicted_direction != row["expected_direction"]
            )
        else:
            non_feedback_total += 1
            false_positives += predicted_feedback
            false_positive_learning_updates += (
                predicted_feedback
                and prediction.get("confidence", 1.0) >= learning_threshold
            )
    return {
        "examples": len(expected),
        "learning_threshold": learning_threshold,
        "feedback_detection_accuracy": detection_correct / len(expected),
        "behavior_precision": (
            true_behavior_predictions / behavior_predictions
            if behavior_predictions
            else None
        ),
        "behavior_recall": (
            true_behavior_predictions / feedback_total if feedback_total else None
        ),
        "applied_behavior_precision": (
            true_applied_behavior_predictions / applied_behavior_predictions
            if applied_behavior_predictions
            else None
        ),
        "applied_behavior_recall": (
            true_applied_behavior_predictions / feedback_total
            if feedback_total
            else None
        ),
        "target_accuracy": target_correct / len(expected),
        "direction_accuracy": (
            direction_correct / feedback_total if feedback_total else None
        ),
        "false_direction_update_rate": (
            false_direction_updates / behavior_predictions
            if behavior_predictions
            else None
        ),
        "false_positive_rate": (
            false_positives / non_feedback_total if non_feedback_total else None
        ),
        "false_positive_learning_rate": (
            false_positive_learning_updates / non_feedback_total
            if non_feedback_total
            else None
        ),
        "false_direction_learning_rate": (
            false_direction_learning_updates / applied_behavior_predictions
            if applied_behavior_predictions
            else None
        ),
        "harmful_learning_updates": (
            false_positive_learning_updates + false_direction_learning_updates
        ),
        "invalid_outputs": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a judge against labeled feedback examples.")
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/data/evaluator_examples.json"))
    parser.add_argument("--judge", help="Optional synchronous judge as module:function")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluator_summary.json"))
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    judge, benchmark_type = load_judge(args.judge)
    extractor = LLMFeedbackExtractor(judge=judge)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        try:
            event = extractor.extract(
                context="benchmark",
                actions=[row["action"], "alternative"],
                previous_prompt=row["previous_prompt"],
                previous_response=row["previous_response"],
                action=row["action"],
                user_message=row["next_message"],
            )
            predictions.append(
                {
                    "target": event.target.value,
                    "sentiment": event.sentiment.value,
                    "confidence": event.confidence,
                }
            )
        except Exception as exc:
            predictions.append({"invalid": True, "error": f"{type(exc).__name__}: {exc}"})
    payload = {"benchmark_type": benchmark_type, "metrics": score(rows, predictions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

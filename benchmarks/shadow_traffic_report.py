from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.evaluator_benchmark import score
from benchmarks.statistics import feedback_confidence_intervals

ALLOWED_FIELDS = {
    "event_id",
    "expected_has_feedback",
    "expected_target",
    "expected_direction",
    "predicted_target",
    "predicted_direction",
    "confidence",
    "invalid",
}
EXPECTED_TARGETS = {
    "behavior",
    "answer_content",
    "task_continuation",
    "quoted_or_meta",
    "unrelated",
}
DIRECTIONS = {"positive", "negative", "none"}
EXPECTED_FIELDS = {
    "event_id",
    "expected_has_feedback",
    "expected_target",
    "expected_direction",
}
PREDICTION_FIELDS = {"predicted_target", "predicted_direction", "confidence"}


def _error(line_number: int, message: str) -> ValueError:
    return ValueError(f"line {line_number}: {message}")


def _validate_row(row: Any, line_number: int, seen_ids: set[str]) -> None:
    if not isinstance(row, dict):
        raise _error(line_number, "each JSONL record must be an object")
    unknown = set(row) - ALLOWED_FIELDS
    if unknown:
        raise _error(line_number, f"contains forbidden fields: {sorted(unknown)}")
    missing = EXPECTED_FIELDS - set(row)
    if missing:
        raise _error(line_number, f"missing required fields: {sorted(missing)}")
    invalid = row.get("invalid", False)
    if not isinstance(invalid, bool):
        raise _error(line_number, "invalid must be a boolean")
    if not invalid:
        missing_prediction = PREDICTION_FIELDS - set(row)
        if missing_prediction:
            raise _error(
                line_number,
                f"missing required prediction fields: {sorted(missing_prediction)}",
            )
    event_id = row["event_id"]
    if not isinstance(event_id, str) or not event_id.strip():
        raise _error(line_number, "event_id must be a nonempty string")
    if event_id in seen_ids:
        raise _error(line_number, "event_id must be unique")
    seen_ids.add(event_id)
    has_feedback = row["expected_has_feedback"]
    if not isinstance(has_feedback, bool):
        raise _error(line_number, "expected_has_feedback must be a boolean")
    expected_target = row["expected_target"]
    expected_direction = row["expected_direction"]
    if expected_target not in EXPECTED_TARGETS:
        raise _error(line_number, "expected_target is not allowed")
    if expected_direction not in DIRECTIONS:
        raise _error(line_number, "expected_direction is not allowed")
    if has_feedback != (expected_target == "behavior"):
        raise _error(line_number, "expected feedback and target are inconsistent")
    if (expected_target == "behavior") != (expected_direction in {"positive", "negative"}):
        raise _error(line_number, "expected target and direction are inconsistent")
    if invalid:
        if "confidence" in row:
            _validate_confidence(row["confidence"], line_number)
        return
    predicted_target = row["predicted_target"]
    predicted_direction = row["predicted_direction"]
    if predicted_target not in EXPECTED_TARGETS:
        raise _error(line_number, "predicted_target is not allowed")
    if predicted_direction not in DIRECTIONS:
        raise _error(line_number, "predicted_direction is not allowed")
    if (predicted_target == "behavior") != (
        predicted_direction in {"positive", "negative"}
    ):
        raise _error(line_number, "predicted target and direction are inconsistent")
    _validate_confidence(row["confidence"], line_number)


def _validate_confidence(value: Any, line_number: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(line_number, "confidence must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise _error(line_number, "confidence must be finite and within [0, 1]")


def load_structured_labels(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = []
    predictions = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(line_number, "invalid JSON") from exc
        _validate_row(row, line_number, seen_ids)
        expected.append(
            {
                "expected_has_feedback": row["expected_has_feedback"],
                "expected_target": row["expected_target"],
                "expected_direction": row["expected_direction"],
            }
        )
        if row.get("invalid"):
            predictions.append({"invalid": True})
        else:
            predictions.append(
                {
                    "target": row["predicted_target"],
                    "sentiment": row["predicted_direction"],
                    "confidence": row["confidence"],
                }
            )
    if not expected:
        raise ValueError("input must contain at least one structured labeled event")
    return expected, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score privacy-safe structured labels from a shadow deployment."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        raise SystemExit("--threshold must be within (0, 1]")
    expected, predictions = load_structured_labels(args.input)
    summary = {
        "benchmark_type": "shadow_traffic",
        "metrics": score(
            expected, predictions, learning_threshold=args.threshold
        ),
        "confidence_intervals_95": feedback_confidence_intervals(
            expected,
            predictions,
            learning_threshold=args.threshold,
        ),
        "contains_request_content": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

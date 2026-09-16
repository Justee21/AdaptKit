import json
import math
from pathlib import Path

import pytest

from benchmarks.evaluator_benchmark import score
from benchmarks.shadow_traffic_report import load_structured_labels


def _row(**overrides):
    row = {
        "event_id": "event-1",
        "expected_has_feedback": True,
        "expected_target": "behavior",
        "expected_direction": "positive",
        "predicted_target": "behavior",
        "predicted_direction": "positive",
        "confidence": 0.96,
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_shadow_report_accepts_only_structured_labels(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "expected_has_feedback": True,
                "expected_target": "behavior",
                "expected_direction": "positive",
                "predicted_target": "behavior",
                "predicted_direction": "positive",
                "confidence": 0.96,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected, predictions = load_structured_labels(path)
    assert expected[0]["expected_target"] == "behavior"
    assert predictions[0]["confidence"] == 0.96


def test_shadow_report_rejects_request_content(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "prompt": "sensitive text",
                "expected_has_feedback": False,
                "expected_target": "unrelated",
                "expected_direction": "none",
                "predicted_target": "unrelated",
                "predicted_direction": "none",
                "confidence": 0.9,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        load_structured_labels(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"event_id": ""}, "event_id must be a nonempty string"),
        ({"expected_has_feedback": 1}, "expected_has_feedback must be a boolean"),
        ({"expected_target": "unknown"}, "expected_target is not allowed"),
        ({"expected_direction": "up"}, "expected_direction is not allowed"),
        ({"predicted_target": "unknown"}, "predicted_target is not allowed"),
        ({"predicted_direction": "up"}, "predicted_direction is not allowed"),
        ({"confidence": True}, "confidence must be a number"),
        ({"confidence": math.nan}, "confidence must be finite"),
        ({"confidence": math.inf}, "confidence must be finite"),
        ({"confidence": -0.1}, "confidence must be finite"),
        ({"confidence": 1.1}, "confidence must be finite"),
        (
            {"expected_has_feedback": False},
            "expected feedback and target are inconsistent",
        ),
        (
            {"expected_direction": "none"},
            "expected target and direction are inconsistent",
        ),
        (
            {"predicted_direction": "none"},
            "predicted target and direction are inconsistent",
        ),
        ({"invalid": "yes"}, "invalid must be a boolean"),
    ],
)
def test_shadow_report_rejects_malformed_records(tmp_path, overrides, message) -> None:
    path = _write(tmp_path, [_row(**overrides)])
    with pytest.raises(ValueError, match=message):
        load_structured_labels(path)


def test_shadow_report_rejects_missing_fields(tmp_path) -> None:
    row = _row()
    del row["event_id"]
    with pytest.raises(ValueError, match="missing required fields"):
        load_structured_labels(_write(tmp_path, [row]))


def test_shadow_report_rejects_missing_prediction_fields(tmp_path) -> None:
    row = _row()
    del row["confidence"]
    with pytest.raises(ValueError, match="missing required prediction fields"):
        load_structured_labels(_write(tmp_path, [row]))


def test_shadow_report_rejects_duplicate_event_ids(tmp_path) -> None:
    with pytest.raises(ValueError, match="event_id must be unique"):
        load_structured_labels(_write(tmp_path, [_row(), _row()]))


def test_shadow_report_rejects_invalid_json_without_echoing_content(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"private": "DO-NOT-ECHO"', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: invalid JSON") as error:
        load_structured_labels(path)
    assert "DO-NOT-ECHO" not in str(error.value)


def test_invalid_record_can_omit_prediction_fields(tmp_path) -> None:
    row = _row(invalid=True)
    for field in ("predicted_target", "predicted_direction", "confidence"):
        del row[field]
    _, predictions = load_structured_labels(_write(tmp_path, [row]))
    assert predictions == [{"invalid": True}]


def test_score_handles_all_feedback_without_fake_false_positive_metrics() -> None:
    expected = [
        {
            "expected_has_feedback": True,
            "expected_target": "behavior",
            "expected_direction": "positive",
        }
    ]
    metrics = score(
        expected,
        [{"target": "behavior", "sentiment": "positive", "confidence": 0.99}],
    )
    assert metrics["false_positive_rate"] is None
    assert metrics["false_positive_learning_rate"] is None
    assert metrics["behavior_precision"] == 1.0


def test_score_handles_all_nonfeedback_without_fake_recall_metrics() -> None:
    expected = [
        {
            "expected_has_feedback": False,
            "expected_target": "unrelated",
            "expected_direction": "none",
        }
    ]
    metrics = score(
        expected,
        [{"target": "unrelated", "sentiment": "none", "confidence": 0.99}],
    )
    assert metrics["behavior_precision"] is None
    assert metrics["behavior_recall"] is None
    assert metrics["direction_accuracy"] is None
    assert metrics["false_positive_rate"] == 0.0


def test_shadow_report_scores_mixed_input(tmp_path) -> None:
    rows = [
        _row(),
        _row(
            event_id="event-2",
            expected_has_feedback=False,
            expected_target="unrelated",
            expected_direction="none",
            predicted_target="unrelated",
            predicted_direction="none",
        ),
    ]
    expected, predictions = load_structured_labels(_write(tmp_path, rows))
    metrics = score(expected, predictions, learning_threshold=0.95)
    assert metrics["feedback_detection_accuracy"] == 1.0
    assert metrics["behavior_precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0

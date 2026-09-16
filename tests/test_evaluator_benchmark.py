import json
import unittest
from difflib import SequenceMatcher
from pathlib import Path

from benchmarks.evaluator_benchmark import score
from benchmarks.openai_evaluator_benchmark import (
    PredictionBatch,
    _add_usage,
    aggregate_metrics,
    aligned_predictions,
    benchmark_input,
    load_examples,
    release_gate,
)
from benchmarks.openai_prompt_routing_benchmark import (
    release_gate as prompt_release_gate,
    score_prompt_routing,
)
from benchmarks.prompt_routing_cases import prompt_routing_examples
from benchmarks.statistics import wilson_interval


class EvaluatorBenchmarkTests(unittest.TestCase):
    def test_dataset_has_required_coverage(self):
        path = Path(__file__).parents[1] / "benchmarks/data/evaluator_examples.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 20)
        self.assertLessEqual(len(rows), 30)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "clear_positive_feedback",
                "clear_negative_feedback",
                "preference_correction",
                "ordinary_continuation",
                "topic_change",
                "unrelated_factual_correction",
                "ambiguous_message",
            },
        )
        self.assertTrue(all("expected_target" in row for row in rows))

    def test_adversarial_dataset_has_required_coverage(self):
        path = Path(__file__).parents[1] / "benchmarks/data/evaluator_adversarial_examples.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "paraphrase",
                "indirect_feedback",
                "quoted_feedback",
                "ambiguous_continuation",
                "prompt_injection",
                "polite_preference_correction",
                "praise_then_correction",
                "historical_reference",
            },
        )
        self.assertTrue(all("expected_target" in row for row in rows))

    def test_score_calculation(self):
        expected = [
            {"expected_has_feedback": True, "expected_target": "behavior", "expected_direction": "positive"},
            {"expected_has_feedback": True, "expected_target": "behavior", "expected_direction": "negative"},
            {"expected_has_feedback": False, "expected_target": "task_continuation", "expected_direction": "none"},
            {"expected_has_feedback": False, "expected_target": "answer_content", "expected_direction": "none"},
        ]
        predictions = [
            {"target": "behavior", "sentiment": "positive"},
            {"target": "behavior", "sentiment": "positive"},
            {"target": "behavior", "sentiment": "negative"},
            {"target": "answer_content", "sentiment": "none"},
        ]
        result = score(expected, predictions)
        self.assertEqual(result["feedback_detection_accuracy"], 0.75)
        self.assertAlmostEqual(result["behavior_precision"], 2 / 3)
        self.assertEqual(result["behavior_recall"], 1.0)
        self.assertEqual(result["direction_accuracy"], 0.5)
        self.assertAlmostEqual(result["false_direction_update_rate"], 1 / 3)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["false_positive_learning_rate"], 0.5)
        self.assertEqual(result["target_accuracy"], 0.75)

    def test_multi_run_metrics_report_mean_range_and_invalid_total(self):
        first = {
            "examples": 46,
            "feedback_detection_accuracy": 0.8,
            "invalid_outputs": 1,
        }
        second = {
            "examples": 46,
            "feedback_detection_accuracy": 1.0,
            "invalid_outputs": 2,
        }
        aggregate, ranges = aggregate_metrics([first, second])
        self.assertEqual(aggregate["examples"], 46)
        self.assertEqual(aggregate["feedback_detection_accuracy"], 0.9)
        self.assertEqual(aggregate["invalid_outputs"], 3)
        self.assertEqual(
            ranges["feedback_detection_accuracy"], {"min": 0.8, "max": 1.0}
        )

    def test_real_benchmark_alignment_rejects_inconsistent_target_and_sentiment(self):
        inconsistent = {
            "target": "task_continuation",
            "sentiment": "negative",
            "confidence": 0.8,
        }
        batch = PredictionBatch.model_validate({"predictions": [inconsistent] * 46})
        aligned, unexpected = aligned_predictions([{}] * 46, batch)
        self.assertTrue(all(row["invalid"] for row in aligned))
        self.assertEqual(unexpected, 0)

    def test_real_benchmark_accumulates_available_usage(self):
        class Usage:
            input_tokens = 10
            output_tokens = 5
            total_tokens = 15

        total: dict[str, int | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        _add_usage(total, Usage())
        _add_usage(total, Usage())
        self.assertEqual(
            total, {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        )

    def test_real_benchmark_uses_production_payload_names(self):
        row = {
            "previous_prompt": "prompt",
            "previous_response": "response",
            "action": "patch_first",
            "next_message": "message",
        }
        payload = json.loads(benchmark_input([row]).split("\n", 1)[1])
        self.assertEqual(payload["selected_action"], "patch_first")
        self.assertEqual(payload["selected_action_description"], "patch first")
        self.assertEqual(payload["action_descriptions"]["patch_first"], "patch first")
        self.assertEqual(payload["latest_user_message"], "message")
        self.assertNotIn("action", payload)
        self.assertNotIn("next_message", payload)

    def test_expanded_feedback_dataset_has_fixed_calibration_split(self):
        rows = load_examples()
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row["split"] == "calibration" for row in rows), 40)
        self.assertEqual(sum(row["split"] == "holdout" for row in rows), 80)
        self.assertTrue(all("example_id" in row for row in rows))

    def test_prompt_routing_dataset_and_metrics(self):
        rows = prompt_routing_examples()
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(row["split"] == "calibration" for row in rows), 20)
        predictions = [
            {"action": row["expected_action"], "confidence": 0.99} for row in rows
        ]
        metrics = score_prompt_routing(rows, predictions, threshold=0.90)
        self.assertEqual(metrics["prompt_cue_routing_accuracy"], 1.0)
        self.assertEqual(metrics["no_cue_false_override_rate"], 0.0)
        self.assertEqual(metrics["wrong_action_overrides"], 0)

    def test_prompt_metrics_are_undefined_for_absent_classes(self):
        explicit_only = [
            {"expected_action": "a"},
        ]
        no_cue_only = [
            {"expected_action": None},
        ]
        explicit_metrics = score_prompt_routing(
            explicit_only, [{"action": "a", "confidence": 1.0}], threshold=0.90
        )
        no_cue_metrics = score_prompt_routing(
            no_cue_only, [{"action": None, "confidence": 1.0}], threshold=0.90
        )
        self.assertIsNone(explicit_metrics["no_cue_false_override_rate"])
        self.assertIsNone(no_cue_metrics["prompt_cue_routing_accuracy"])
        self.assertIsNone(no_cue_metrics["routed_action_precision"])
        self.assertFalse(prompt_release_gate(explicit_metrics)["passed"])
        self.assertFalse(prompt_release_gate(no_cue_metrics)["passed"])

    def test_calibration_and_holdout_have_no_duplicate_or_near_duplicate_signals(self):
        datasets = (
            (load_examples(), lambda row: row["next_message"]),
            (
                load_examples(),
                lambda row: " ".join(
                    (
                        row["previous_prompt"],
                        row["previous_response"],
                        row["next_message"],
                    )
                ),
            ),
            (prompt_routing_examples(), lambda row: row["prompt"]),
        )
        for rows, text in datasets:
            calibration = [row for row in rows if row["split"] == "calibration"]
            holdout = [row for row in rows if row["split"] == "holdout"]
            self.assertFalse(
                {row["example_id"] for row in calibration}
                & {row["example_id"] for row in holdout}
            )
            for left in calibration:
                for right in holdout:
                    left_text = " ".join(text(left).casefold().split())
                    right_text = " ".join(text(right).casefold().split())
                    self.assertNotEqual(left_text, right_text)
                    self.assertLess(
                        SequenceMatcher(None, left_text, right_text).ratio(), 0.90
                    )

    def test_release_gate_applies_to_one_holdout_run(self):
        metrics = {
            "examples": 80,
            "applied_behavior_precision": 0.96,
            "applied_behavior_recall": 0.86,
            "false_direction_learning_rate": 0.0,
            "false_positive_learning_rate": 0.02,
            "invalid_outputs": 0,
        }
        self.assertTrue(release_gate(metrics)["passed"])
        metrics["false_direction_learning_rate"] = 0.01
        self.assertFalse(release_gate(metrics)["passed"])

    def test_release_gate_fails_when_a_required_metric_is_undefined(self):
        metrics = {
            "examples": 80,
            "applied_behavior_precision": None,
            "applied_behavior_recall": 0.90,
            "false_direction_learning_rate": None,
            "false_positive_learning_rate": 0.0,
            "invalid_outputs": 0,
        }
        self.assertFalse(release_gate(metrics)["passed"])

    def test_aggregate_metrics_preserves_fully_undefined_metrics(self):
        first = {"examples": 5, "behavior_precision": None, "invalid_outputs": 0}
        second = {"examples": 5, "behavior_precision": None, "invalid_outputs": 0}
        aggregate, ranges = aggregate_metrics([first, second])
        self.assertIsNone(aggregate["behavior_precision"])
        self.assertEqual(
            ranges["behavior_precision"], {"min": None, "max": None}
        )

    def test_wilson_interval_reports_counts_and_bounds(self):
        interval = wilson_interval(9, 10)
        self.assertEqual(interval["successes"], 9)
        self.assertEqual(interval["total"], 10)
        self.assertLess(interval["lower"], 0.9)
        self.assertGreater(interval["upper"], 0.9)


if __name__ == "__main__":
    unittest.main()

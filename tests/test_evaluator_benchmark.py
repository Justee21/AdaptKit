import json
import unittest
from pathlib import Path

from benchmarks.evaluator_benchmark import score
from benchmarks.openai_evaluator_benchmark import (
    PredictionBatch,
    _add_usage,
    aggregate_metrics,
    aligned_predictions,
    benchmark_input,
)


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


if __name__ == "__main__":
    unittest.main()

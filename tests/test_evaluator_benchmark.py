import json
import unittest
from pathlib import Path

from benchmarks.evaluator_benchmark import score


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
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "paraphrase",
                "indirect_feedback",
                "quoted_feedback",
                "ambiguous_continuation",
                "prompt_injection",
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
        self.assertEqual(result["direction_accuracy"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["target_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()

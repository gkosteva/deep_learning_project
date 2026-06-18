import unittest

from src.fake_news.training.metrics import (accuracy_score, confusion_counts, confusion_matrix,
                                            evaluate_classification, f1_score, precision_score,
                                            recall_score)


class TestConfusionCounts(unittest.TestCase):

    def test_when_lengths_mismatch_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            confusion_counts([0], [0, 1])

    def test_when_empty_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            confusion_counts([], [])

    def test_when_called_then_returns_tp_fp_tn_fn(self):
        references = [0, 0, 1, 1]
        predictions = [0, 1, 1, 0]
        self.assertEqual(confusion_counts(references, predictions, 0), (1, 1, 1, 1))


class TestAccuracyScore(unittest.TestCase):

    def test_when_all_correct_then_returns_one(self):
        self.assertEqual(accuracy_score([0, 1], [0, 1]), 1.0)

    def test_when_half_correct_then_returns_half(self):
        self.assertEqual(accuracy_score([0, 1], [0, 0]), 0.5)


class TestPrecisionScore(unittest.TestCase):

    def test_when_no_positive_predictions_then_returns_zero(self):
        self.assertEqual(precision_score([0, 0], [1, 1], positive_label=0), 0.0)

    def test_when_computed_then_matches_definition(self):
        self.assertEqual(precision_score([0, 1], [0, 0], positive_label=0), 0.5)


class TestRecallScore(unittest.TestCase):

    def test_when_no_actual_positives_then_returns_zero(self):
        self.assertEqual(recall_score([1, 1], [1, 1], positive_label=0), 0.0)

    def test_when_computed_then_matches_definition(self):
        self.assertEqual(recall_score([0, 0], [0, 1], positive_label=0), 0.5)


class TestF1Score(unittest.TestCase):

    def test_when_precision_and_recall_zero_then_returns_zero(self):
        self.assertEqual(f1_score([0, 0], [1, 1], positive_label=0), 0.0)

    def test_when_perfect_then_returns_one(self):
        self.assertEqual(f1_score([0, 1], [0, 1], positive_label=0), 1.0)


class TestConfusionMatrix(unittest.TestCase):

    def test_when_lengths_mismatch_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0], [0, 1])

    def test_when_called_then_counts_each_cell(self):
        matrix = confusion_matrix([0, 1, 1], [0, 1, 0])
        self.assertEqual(matrix, [[1, 0], [1, 1]])


class TestEvaluateClassification(unittest.TestCase):

    def test_when_called_then_returns_three_metrics(self):
        result = evaluate_classification([0, 1], [0, 1])
        self.assertEqual(set(result.keys()), {'accuracy', 'f1', 'recall'})


if __name__ == '__main__':
    unittest.main()

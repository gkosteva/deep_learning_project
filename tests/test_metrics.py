import unittest

from src.fake_news.training.metrics import accuracy_score, confusion_matrix, \
    evaluate_classification, macro_f1_score, macro_precision_score, macro_recall_score, \
    per_class_scores


class TestAccuracyScore(unittest.TestCase):

    def test_when_lengths_mismatch_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            accuracy_score([0], [0, 1])

    def test_when_empty_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            accuracy_score([], [])

    def test_when_all_correct_then_returns_one(self):
        self.assertEqual(accuracy_score([0, 1, 2], [0, 1, 2]), 1.0)

    def test_when_half_correct_then_returns_half(self):
        self.assertEqual(accuracy_score([0, 1], [0, 0]), 0.5)


class TestConfusionMatrix(unittest.TestCase):

    def test_when_num_classes_too_small_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0, 1], num_classes=1)

    def test_when_called_then_counts_each_cell(self):
        matrix = confusion_matrix([0, 1, 1], [0, 1, 0], num_classes=2)
        self.assertEqual(matrix, [[1, 0], [1, 1]])

    def test_when_multiclass_then_matrix_is_square(self):
        matrix = confusion_matrix([0, 1, 2], [0, 2, 2], num_classes=3)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix[0]), 3)


class TestPerClassScores(unittest.TestCase):

    def test_when_perfect_then_all_f1_are_one(self):
        matrix = confusion_matrix([0, 1, 2], [0, 1, 2], num_classes=3)
        scores = per_class_scores(matrix)
        self.assertTrue(all(abs(f1 - 1.0) < 1e-9 for _, _, f1 in scores))

    def test_when_class_absent_then_scores_are_zero(self):
        matrix = confusion_matrix([0, 0], [0, 0], num_classes=2)
        scores = per_class_scores(matrix)
        self.assertEqual(scores[1], (0.0, 0.0, 0.0))


class TestMacroScores(unittest.TestCase):

    def test_when_perfect_then_macro_metrics_are_one(self):
        refs, preds = [0, 1, 2], [0, 1, 2]
        self.assertEqual(macro_precision_score(refs, preds, 3), 1.0)
        self.assertEqual(macro_recall_score(refs, preds, 3), 1.0)
        self.assertEqual(macro_f1_score(refs, preds, 3), 1.0)

    def test_when_all_wrong_then_macro_f1_is_zero(self):
        self.assertEqual(macro_f1_score([0, 0], [1, 1], 2), 0.0)


class TestEvaluateClassification(unittest.TestCase):

    def test_when_called_then_returns_expected_metrics(self):
        result = evaluate_classification([0, 1, 2], [0, 1, 2], num_classes=3)
        self.assertEqual(set(result.keys()),
                         {'accuracy', 'macro_f1', 'macro_precision', 'macro_recall'})


if __name__ == '__main__':
    unittest.main()

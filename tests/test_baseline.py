import unittest

from src.fake_news.models.baseline import MajorityClassClassifier


class TestMajorityClassClassifierFit(unittest.TestCase):

    def test_when_fitted_then_learns_most_frequent_class(self):
        model = MajorityClassClassifier().fit([0, 0, 0, 1])
        self.assertEqual(model.majority_class, 0)

    def test_when_fitted_on_empty_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            MajorityClassClassifier().fit([])

    def test_when_tie_then_smallest_label_is_chosen(self):
        model = MajorityClassClassifier().fit([1, 0])
        self.assertEqual(model.majority_class, 0)


class TestMajorityClassClassifierMajorityClass(unittest.TestCase):

    def test_when_not_fitted_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            _ = MajorityClassClassifier().majority_class


class TestMajorityClassClassifierPredict(unittest.TestCase):

    def test_when_predicting_then_returns_majority_for_every_sample(self):
        model = MajorityClassClassifier().fit([1, 1, 0])
        self.assertEqual(model.predict(['a', 'b', 'c']), [1, 1, 1])


if __name__ == '__main__':
    unittest.main()

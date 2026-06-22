import unittest

from src.fake_news.models.tfidf_classifier import TfidfLogisticClassifier

_TEXTS = [
    'taxes rose sharply this year',
    'the budget deficit grew again',
    'jobs were created across the state',
    'unemployment fell to a record low',
    'taxes and the deficit keep rising',
    'new jobs and lower unemployment reported',
]
_LABELS = [0, 0, 1, 1, 0, 1]


class TestTfidfLogisticClassifier(unittest.TestCase):

    def setUp(self):
        self.model = TfidfLogisticClassifier(max_features=50).fit(_TEXTS, _LABELS)

    def test_when_predict_then_returns_one_label_per_text(self):
        predictions = self.model.predict(['taxes rose', 'new jobs'])
        self.assertEqual(len(predictions), 2)

    def test_when_predict_proba_then_rows_sum_to_one(self):
        probabilities = self.model.predict_proba(['taxes rose'])
        self.assertAlmostEqual(float(probabilities[0].sum()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()

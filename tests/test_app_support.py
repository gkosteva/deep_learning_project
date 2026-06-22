import unittest

from src.fake_news.app_support import TfidfService, describe, format_probabilities, top_prediction
from src.fake_news.models.tfidf_classifier import TfidfLogisticClassifier

_RANKED = [('true', 0.6), ('false', 0.3), ('half-true', 0.1)]


class TestDescribe(unittest.TestCase):

    def test_when_known_label_then_returns_description(self):
        self.assertIn('accurate', describe('true'))

    def test_when_unknown_label_then_returns_label(self):
        self.assertEqual(describe('mystery'), 'mystery')


class TestTopPrediction(unittest.TestCase):

    def test_when_ranked_then_returns_first(self):
        self.assertEqual(top_prediction(_RANKED), ('true', 0.6))

    def test_when_empty_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            top_prediction([])


class TestFormatProbabilities(unittest.TestCase):

    def test_when_called_then_rows_have_percent(self):
        rows = format_probabilities(_RANKED)
        self.assertEqual(rows[0]['label'], 'true')
        self.assertEqual(rows[0]['percent'], 60.0)


class TestTfidfService(unittest.TestCase):

    def setUp(self):
        texts = ['taxes rose', 'jobs created', 'taxes again', 'jobs grew']
        labels = [0, 1, 0, 1]
        model = TfidfLogisticClassifier(max_features=20).fit(texts, labels)
        self.service = TfidfService(model, ['fake', 'real'])

    def test_when_predict_proba_then_returns_ranked_labels(self):
        ranked = self.service.predict_proba('taxes rose')
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])

    def test_when_predict_then_returns_a_known_label(self):
        self.assertIn(self.service.predict('jobs created'), ['fake', 'real'])


if __name__ == '__main__':
    unittest.main()

import unittest

from src.fake_news.config import (ExperimentConfig, LSTMConfig, TransformerConfig)


class TestLSTMConfig(unittest.TestCase):

    def test_when_as_report_columns_then_includes_core_hyperparameters(self):
        columns = LSTMConfig().as_report_columns()
        self.assertIn('hidden_size', columns)
        self.assertIn('bidirectional', columns)

    def test_when_as_report_columns_then_includes_weight_decay(self):
        self.assertIn('weight_decay', LSTMConfig().as_report_columns())


class TestTransformerConfig(unittest.TestCase):

    def test_when_as_report_columns_then_includes_model_name(self):
        self.assertIn('model_name', TransformerConfig().as_report_columns())


class TestExperimentConfig(unittest.TestCase):

    def test_when_to_dict_then_returns_nested_mapping(self):
        result = ExperimentConfig().to_dict()
        self.assertIn('data', result)
        self.assertIn('lstm', result)


if __name__ == '__main__':
    unittest.main()

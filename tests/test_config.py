import unittest

from src.fake_news.config import DataConfig, ExperimentConfig, RNNConfig, TransformerConfig, \
    class_names, num_classes


class TestClassHelpers(unittest.TestCase):

    def test_when_task_six_then_six_class_names(self):
        self.assertEqual(num_classes('six'), 6)
        self.assertIn('pants-fire', class_names('six'))

    def test_when_task_binary_then_two_class_names(self):
        self.assertEqual(num_classes('binary'), 2)
        self.assertEqual(class_names('binary'), ['fake', 'real'])

    def test_when_task_invalid_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            class_names('nonsense')


class TestDataConfig(unittest.TestCase):

    def test_when_created_then_task_defaults_to_six(self):
        self.assertEqual(DataConfig().task, 'six')


class TestRNNConfig(unittest.TestCase):

    def test_when_as_report_columns_then_includes_rnn_type_and_hidden_size(self):
        columns = RNNConfig().as_report_columns()
        self.assertIn('rnn_type', columns)
        self.assertIn('hidden_size', columns)
        self.assertIn('weight_decay', columns)


class TestTransformerConfig(unittest.TestCase):

    def test_when_as_report_columns_then_includes_model_name(self):
        self.assertIn('model_name', TransformerConfig().as_report_columns())


class TestExperimentConfig(unittest.TestCase):

    def test_when_to_dict_then_returns_nested_mapping(self):
        result = ExperimentConfig().to_dict()
        self.assertIn('data', result)
        self.assertIn('rnn', result)

    def test_when_num_classes_property_then_matches_task(self):
        config = ExperimentConfig(data=DataConfig(task='binary'))
        self.assertEqual(config.num_classes, 2)
        self.assertEqual(config.class_names, ['fake', 'real'])


if __name__ == '__main__':
    unittest.main()

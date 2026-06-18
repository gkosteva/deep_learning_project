import os
import tempfile
import unittest

import pandas as pd

from src.fake_news.config import DataConfig, ExperimentConfig, LSTMConfig
from src.fake_news.main import _collect_examples, load_dataframe, run


def _tiny_config(directory: str) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(fake_path=os.path.join(directory, 'missing_fake.csv'),
                        true_path=os.path.join(directory, 'missing_true.csv'),
                        min_token_frequency=1,
                        max_sequence_length=20),
        lstm=LSTMConfig(embedding_dim=8, hidden_size=8, epochs=1, batch_size=32),
        report_path=os.path.join(directory, 'report.xlsx'),
        figures_dir=os.path.join(directory, 'figures'),
    )


class TestLoadDataframe(unittest.TestCase):

    def test_when_isot_missing_then_falls_back_to_synthetic(self):
        config = _tiny_config(tempfile.mkdtemp())
        frame, source = load_dataframe(config)
        self.assertEqual(source, 'synthetic')
        self.assertGreater(len(frame), 0)

    def test_when_isot_present_then_loads_isot(self):
        directory = tempfile.mkdtemp()
        fake_path = os.path.join(directory, 'Fake.csv')
        true_path = os.path.join(directory, 'True.csv')
        pd.DataFrame({'title': ['a'], 'text': ['x']}).to_csv(fake_path, index=False)
        pd.DataFrame({'title': ['b'], 'text': ['y']}).to_csv(true_path, index=False)
        config = ExperimentConfig(data=DataConfig(fake_path=fake_path, true_path=true_path))
        _, source = load_dataframe(config)
        self.assertEqual(source, 'ISOT')


class TestCollectExamples(unittest.TestCase):

    def test_when_called_then_mixes_correct_and_incorrect(self):
        texts = ['a', 'b', 'c', 'd']
        references = [0, 1, 0, 1]
        predictions = [0, 1, 1, 0]
        examples = _collect_examples(texts, references, predictions, limit=4)
        self.assertEqual(len(examples), 4)


class TestRun(unittest.TestCase):

    def test_when_run_then_report_and_figures_are_produced(self):
        directory = tempfile.mkdtemp()
        config = _tiny_config(directory)
        exit_code = run(config)
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(config.report_path))
        self.assertTrue(os.path.exists(os.path.join(config.figures_dir, 'class_distribution.png')))


if __name__ == '__main__':
    unittest.main()

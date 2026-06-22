import os
import tempfile
import unittest

from src.fake_news.config import DataConfig, ExperimentConfig
from src.fake_news.data.preprocessing import Vocabulary
from src.fake_news.inference import PredictionService, artifact_path, label_names, \
    save_rnn_artifact
from src.fake_news.models.rnn_classifier import RNNClassifier


def _setup(task='six'):
    vocabulary = Vocabulary.build(['taxes jobs budget deficit economy'], min_frequency=1)
    config = ExperimentConfig(data=DataConfig(task=task, embedding_dim=8, max_sequence_length=10))
    model = RNNClassifier(vocab_size=len(vocabulary),
                          embedding_dim=8,
                          hidden_size=8,
                          num_classes=config.num_classes,
                          pad_id=vocabulary.pad_id)
    return model, vocabulary, config


class TestSaveAndLoad(unittest.TestCase):

    def test_when_saved_then_service_loads_and_predicts(self):
        directory = tempfile.mkdtemp()
        model, vocabulary, config = _setup('six')
        save_rnn_artifact(model, vocabulary, config, 'LSTM (test)', artifact_dir=directory)
        service = PredictionService.load(artifact_path('six', directory))
        ranked = service.predict_proba('taxes and jobs')
        self.assertEqual(len(ranked), 6)
        self.assertAlmostEqual(sum(probability for _, probability in ranked), 1.0, places=4)
        self.assertIn(service.predict('taxes and jobs'), label_names('six'))

    def test_when_artifact_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            PredictionService.load('/tmp/no-such-artifact.pt')


class TestLabelNames(unittest.TestCase):

    def test_when_binary_then_returns_fake_and_real(self):
        self.assertEqual(label_names('binary'), ['fake', 'real'])


if __name__ == '__main__':
    unittest.main()

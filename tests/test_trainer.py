import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fake_news.models.rnn_classifier import RNNClassifier
from src.fake_news.training.trainer import Trainer, TrainingHistory, select_device


def _make_loader(num_samples: int = 16,
                 seq_len: int = 5,
                 vocab_size: int = 20,
                 num_classes: int = 3):
    features = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, num_classes, (num_samples, ))
    return DataLoader(TensorDataset(features, labels), batch_size=8)


class TestSelectDevice(unittest.TestCase):

    def test_when_called_then_returns_a_torch_device(self):
        self.assertIsInstance(select_device(), torch.device)

    def test_when_mps_available_then_returns_mps(self):
        with mock.patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(select_device().type, 'mps')

    def test_when_only_cuda_available_then_returns_cuda(self):
        with mock.patch('torch.backends.mps.is_available', return_value=False), \
                mock.patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(select_device().type, 'cuda')

    def test_when_no_accelerator_then_returns_cpu(self):
        with mock.patch('torch.backends.mps.is_available', return_value=False), \
                mock.patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(select_device().type, 'cpu')


class TestTrainerFit(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = RNNClassifier(vocab_size=20, embedding_dim=8, hidden_size=8, num_classes=3)
        self.trainer = Trainer(self.model,
                               learning_rate=1e-2,
                               device=torch.device('cpu'),
                               num_classes=3)

    def test_when_fit_then_history_has_one_entry_per_epoch(self):
        history = self.trainer.fit(_make_loader(), _make_loader(), epochs=2, patience=5)
        self.assertIsInstance(history, TrainingHistory)
        self.assertEqual(len(history.train_loss), 2)
        self.assertEqual(len(history.val_f1), 2)

    def test_when_validation_does_not_improve_then_training_stops_early(self):
        history = self.trainer.fit(_make_loader(), _make_loader(), epochs=10, patience=1)
        self.assertLessEqual(len(history.train_loss), 10)

    def test_when_weight_decay_set_then_training_still_runs(self):
        torch.manual_seed(0)
        model = RNNClassifier(vocab_size=20, embedding_dim=8, hidden_size=8, num_classes=3)
        trainer = Trainer(model,
                          learning_rate=1e-2,
                          device=torch.device('cpu'),
                          num_classes=3,
                          weight_decay=0.01)
        history = trainer.fit(_make_loader(), _make_loader(), epochs=1, patience=1)
        self.assertEqual(len(history.train_loss), 1)


class TestTrainerPredict(unittest.TestCase):

    def test_when_predict_then_returns_aligned_references_and_predictions(self):
        torch.manual_seed(0)
        model = RNNClassifier(vocab_size=20, embedding_dim=8, hidden_size=8, num_classes=3)
        trainer = Trainer(model, device=torch.device('cpu'), num_classes=3)
        loader = _make_loader(num_samples=8)
        references, predictions = trainer.predict(loader)
        self.assertEqual(len(references), len(predictions))
        self.assertEqual(len(references), 8)


if __name__ == '__main__':
    unittest.main()

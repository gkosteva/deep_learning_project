import unittest

import torch

from src.fake_news.models.lstm_classifier import LSTMClassifier


class TestLSTMClassifierInit(unittest.TestCase):

    def test_when_vocab_size_not_positive_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            LSTMClassifier(vocab_size=0)

    def test_when_num_classes_too_small_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            LSTMClassifier(vocab_size=10, num_classes=1)


class TestLSTMClassifierForward(unittest.TestCase):

    def test_when_given_batch_then_returns_logits_of_expected_shape(self):
        model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=8)
        token_ids = torch.randint(0, 20, (4, 6))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_when_bidirectional_then_forward_still_returns_logits(self):
        model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=8, bidirectional=True)
        token_ids = torch.randint(0, 20, (3, 5))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (3, 2))


class TestLSTMClassifierPretrainedEmbeddings(unittest.TestCase):

    def test_when_shape_mismatch_then_value_error_is_raised(self):
        weights = torch.randn(5, 8)
        with self.assertRaises(ValueError):
            LSTMClassifier(vocab_size=20, embedding_dim=8, pretrained_embeddings=weights)

    def test_when_given_then_embedding_rows_are_copied(self):
        weights = torch.randn(20, 8)
        model = LSTMClassifier(vocab_size=20,
                               embedding_dim=8,
                               hidden_size=8,
                               pad_id=0,
                               pretrained_embeddings=weights)
        torch.testing.assert_close(model.embedding.weight.data[3], weights[3])
        self.assertTrue(torch.all(model.embedding.weight.data[0] == 0))

    def test_when_frozen_then_embedding_is_not_trainable(self):
        weights = torch.randn(20, 8)
        model = LSTMClassifier(vocab_size=20,
                               embedding_dim=8,
                               hidden_size=8,
                               pretrained_embeddings=weights,
                               freeze_embeddings=True)
        self.assertFalse(model.embedding.weight.requires_grad)


class TestLSTMClassifierCountParameters(unittest.TestCase):

    def test_when_called_then_returns_positive_count(self):
        model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=8)
        self.assertGreater(model.count_parameters(), 0)


if __name__ == '__main__':
    unittest.main()

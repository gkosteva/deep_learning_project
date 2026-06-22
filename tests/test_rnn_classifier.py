import unittest

import torch

from src.fake_news.models.rnn_classifier import RNNClassifier


class TestRNNClassifierInit(unittest.TestCase):

    def test_when_vocab_size_not_positive_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            RNNClassifier(vocab_size=0)

    def test_when_num_classes_too_small_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            RNNClassifier(vocab_size=10, num_classes=1)

    def test_when_rnn_type_invalid_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            RNNClassifier(vocab_size=10, rnn_type='transformer')


class TestRNNClassifierForward(unittest.TestCase):

    def test_when_lstm_then_returns_logits_of_expected_shape(self):
        model = RNNClassifier(vocab_size=20, embedding_dim=8, hidden_size=8, num_classes=6)
        token_ids = torch.randint(0, 20, (4, 6))
        self.assertEqual(tuple(model(token_ids).shape), (4, 6))

    def test_when_gru_bidirectional_then_returns_logits(self):
        model = RNNClassifier(vocab_size=20,
                              embedding_dim=8,
                              hidden_size=8,
                              rnn_type='gru',
                              bidirectional=True,
                              num_classes=3)
        token_ids = torch.randint(0, 20, (3, 5))
        self.assertEqual(tuple(model(token_ids).shape), (3, 3))


class TestRNNClassifierPretrainedEmbeddings(unittest.TestCase):

    def test_when_shape_mismatch_then_value_error_is_raised(self):
        weights = torch.randn(5, 8)
        with self.assertRaises(ValueError):
            RNNClassifier(vocab_size=20, embedding_dim=8, pretrained_embeddings=weights)

    def test_when_given_then_embedding_rows_are_copied(self):
        weights = torch.randn(20, 8)
        model = RNNClassifier(vocab_size=20,
                              embedding_dim=8,
                              hidden_size=8,
                              pad_id=0,
                              pretrained_embeddings=weights)
        torch.testing.assert_close(model.embedding.weight.data[3], weights[3])
        self.assertTrue(torch.all(model.embedding.weight.data[0] == 0))

    def test_when_frozen_then_embedding_is_not_trainable(self):
        weights = torch.randn(20, 8)
        model = RNNClassifier(vocab_size=20,
                              embedding_dim=8,
                              hidden_size=8,
                              pretrained_embeddings=weights,
                              freeze_embeddings=True)
        self.assertFalse(model.embedding.weight.requires_grad)


class TestRNNClassifierCountParameters(unittest.TestCase):

    def test_when_called_then_returns_positive_count(self):
        model = RNNClassifier(vocab_size=20, embedding_dim=8, hidden_size=8)
        self.assertGreater(model.count_parameters(), 0)


if __name__ == '__main__':
    unittest.main()

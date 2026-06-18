import os
import tempfile
import unittest

import pandas as pd

from src.fake_news.data.dataset import (FakeNewsDataset, generate_synthetic_dataset, load_isot,
                                        stratified_split)
from src.fake_news.data.preprocessing import Vocabulary


class TestLoadIsot(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.fake_path = os.path.join(self.directory, 'Fake.csv')
        self.true_path = os.path.join(self.directory, 'True.csv')
        pd.DataFrame({'title': ['a', 'b'], 'text': ['x', 'y']}).to_csv(self.fake_path, index=False)
        pd.DataFrame({'title': ['c'], 'text': ['z']}).to_csv(self.true_path, index=False)

    def test_when_files_exist_then_labels_are_assigned(self):
        frame = load_isot(self.fake_path, self.true_path)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame['label'].tolist(), [0, 0, 1])

    def test_when_fake_file_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_isot(os.path.join(self.directory, 'nope.csv'), self.true_path)

    def test_when_true_file_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_isot(self.fake_path, os.path.join(self.directory, 'nope.csv'))


class TestGenerateSyntheticDataset(unittest.TestCase):

    def test_when_called_then_classes_are_balanced(self):
        frame = generate_synthetic_dataset(n_per_class=10, seed=1)
        self.assertEqual(len(frame), 20)
        self.assertEqual((frame['label'] == 0).sum(), 10)
        self.assertEqual((frame['label'] == 1).sum(), 10)

    def test_when_noise_is_zero_then_labels_stay_balanced(self):
        frame = generate_synthetic_dataset(n_per_class=8, seed=1, noise=0.0)
        self.assertEqual((frame['label'] == 0).sum(), 8)
        self.assertEqual((frame['label'] == 1).sum(), 8)

    def test_when_noise_is_one_then_labels_stay_balanced(self):
        frame = generate_synthetic_dataset(n_per_class=8, seed=1, noise=1.0)
        self.assertEqual((frame['label'] == 0).sum(), 8)
        self.assertEqual((frame['label'] == 1).sum(), 8)

    def test_when_called_then_tokens_are_alphabetic(self):
        frame = generate_synthetic_dataset(n_per_class=5, seed=1)
        tokens = ' '.join(frame['text'].tolist()).split()
        self.assertTrue(all(token.isalpha() for token in tokens))


class TestStratifiedSplit(unittest.TestCase):

    def setUp(self):
        self.frame = generate_synthetic_dataset(n_per_class=50, seed=2)

    def test_when_called_then_splits_cover_all_samples(self):
        train, val, test = stratified_split(self.frame, 0.2, 0.2, seed=0)
        self.assertEqual(len(train) + len(val) + len(test), len(self.frame))

    def test_when_test_size_invalid_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            stratified_split(self.frame, 0.2, 1.5, seed=0)

    def test_when_val_size_invalid_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            stratified_split(self.frame, 0.0, 0.2, seed=0)

    def test_when_sizes_sum_too_large_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            stratified_split(self.frame, 0.6, 0.6, seed=0)


class TestFakeNewsDataset(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary.build(['cat dog bird'], min_frequency=1)

    def test_when_lengths_mismatch_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            FakeNewsDataset(['a', 'b'], [0], self.vocabulary, 5)

    def test_when_indexed_then_returns_feature_and_label_tensors(self):
        dataset = FakeNewsDataset(['cat dog'], [1], self.vocabulary, 5)
        features, label = dataset[0]
        self.assertEqual(tuple(features.shape), (5, ))
        self.assertEqual(label.item(), 1)

    def test_when_len_called_then_returns_number_of_samples(self):
        dataset = FakeNewsDataset(['cat', 'dog'], [0, 1], self.vocabulary, 3)
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    unittest.main()

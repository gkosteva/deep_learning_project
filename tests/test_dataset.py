import os
import tempfile
import unittest

from src.fake_news.data.dataset import LiarDataset, generate_synthetic_liar, label_column, \
    load_liar, load_liar_split, select_text
from src.fake_news.data.preprocessing import Vocabulary

_ROW = '\t'.join([
    '1.json', 'false', 'A false claim about taxes.', 'taxes', 'jane-doe', 'Senator', 'Texas',
    'republican', '1', '2', '3', '4', '5', 'a speech'
])
_ROW_TRUE = '\t'.join([
    '2.json', 'true', 'A true claim about jobs.', 'jobs', 'john-roe', 'Governor', 'Ohio',
    'democrat', '0', '0', '1', '1', '0', 'an interview'
])


def _write_split(directory, name, rows):
    path = os.path.join(directory, name)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(rows) + '\n')
    return path


class TestLoadLiarSplit(unittest.TestCase):

    def test_when_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_liar_split('/tmp/nope-liar.tsv')

    def test_when_loaded_then_labels_are_mapped(self):
        directory = tempfile.mkdtemp()
        path = _write_split(directory, 'train.tsv', [_ROW, _ROW_TRUE])
        frame = load_liar_split(path)
        self.assertEqual(frame['label_six'].tolist(), [1, 5])
        self.assertEqual(frame['label_binary'].tolist(), [0, 1])

    def test_when_loaded_then_meta_text_combines_fields(self):
        directory = tempfile.mkdtemp()
        path = _write_split(directory, 'train.tsv', [_ROW])
        frame = load_liar_split(path)
        self.assertIn('jane-doe', frame['meta_text'][0])
        self.assertIn('republican', frame['meta_text'][0])


class TestLoadLiar(unittest.TestCase):

    def test_when_directory_has_all_splits_then_three_frames_returned(self):
        directory = tempfile.mkdtemp()
        _write_split(directory, 'train.tsv', [_ROW, _ROW_TRUE])
        _write_split(directory, 'valid.tsv', [_ROW])
        _write_split(directory, 'test.tsv', [_ROW_TRUE])
        train, val, test = load_liar(directory)
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 1))


class TestGenerateSyntheticLiar(unittest.TestCase):

    def test_when_called_then_every_label_is_present(self):
        frame = generate_synthetic_liar(n_per_class=5, seed=1)
        self.assertEqual(len(frame), 30)
        self.assertEqual(frame['label_six'].nunique(), 6)

    def test_when_called_then_binary_labels_only_zero_and_one(self):
        frame = generate_synthetic_liar(n_per_class=4, seed=1)
        self.assertEqual(set(frame['label_binary'].unique()), {0, 1})


class TestLabelColumn(unittest.TestCase):

    def test_when_task_six_then_returns_six_column(self):
        self.assertEqual(label_column('six'), 'label_six')

    def test_when_task_binary_then_returns_binary_column(self):
        self.assertEqual(label_column('binary'), 'label_binary')

    def test_when_task_invalid_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            label_column('nonsense')


class TestSelectText(unittest.TestCase):

    def setUp(self):
        self.frame = generate_synthetic_liar(n_per_class=2, seed=1)

    def test_when_metadata_false_then_returns_statement(self):
        texts = select_text(self.frame, use_metadata=False)
        self.assertEqual(texts[0], self.frame['statement'][0])

    def test_when_metadata_true_then_text_is_longer(self):
        plain = select_text(self.frame, use_metadata=False)[0]
        with_meta = select_text(self.frame, use_metadata=True)[0]
        self.assertGreaterEqual(len(with_meta), len(plain))


class TestLiarDataset(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary.build(['cat dog bird'], min_frequency=1)

    def test_when_lengths_mismatch_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            LiarDataset(['a', 'b'], [0], self.vocabulary, 5)

    def test_when_indexed_then_returns_feature_and_label_tensors(self):
        dataset = LiarDataset(['cat dog'], [3], self.vocabulary, 5)
        features, label = dataset[0]
        self.assertEqual(tuple(features.shape), (5, ))
        self.assertEqual(label.item(), 3)

    def test_when_len_called_then_returns_number_of_samples(self):
        dataset = LiarDataset(['cat', 'dog'], [0, 1], self.vocabulary, 3)
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    unittest.main()

import os
import tempfile
import unittest

from src.fake_news.config import LABELS_SIX, DataConfig, ExperimentConfig, RNNConfig
from src.fake_news.main import _collect_examples, load_data, make_splits, run

# A small word pool so tokens repeat across rows (TF-IDF needs df >= 2).
_WORDS = [
    'tax', 'budget', 'jobs', 'health', 'economy', 'plan', 'state', 'percent', 'reform', 'spending',
    'rate', 'people', 'bill', 'energy'
]


def _statement(seed: int) -> str:
    return ' '.join(_WORDS[(seed + offset) % len(_WORDS)] for offset in range(8))


def _write_liar(directory, rows_per_label: int = 12):
    lines = []
    counter = 0
    for _ in range(rows_per_label):
        for label in LABELS_SIX:
            counter += 1
            lines.append('\t'.join([
                f'{counter}.json', label,
                _statement(counter), 'economy', 'jane-doe', 'Senator', 'Texas', 'republican', '1',
                '2', '3', '4', '5', 'a speech'
            ]))
    for name in ('train.tsv', 'valid.tsv', 'test.tsv'):
        with open(os.path.join(directory, name), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    return directory


def _tiny_config(directory: str, liar_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(liar_dir=liar_dir,
                        task='six',
                        min_token_frequency=1,
                        max_sequence_length=16,
                        embedding_dim=16,
                        glove_path='/tmp/no-glove.txt',
                        word2vec_path='/tmp/no-word2vec.txt',
                        fasttext_path='/tmp/no-fasttext.vec'),
        rnn=RNNConfig(embedding_dim=16, hidden_size=8, epochs=1, patience=1, batch_size=32),
        report_path=os.path.join(directory, 'report.xlsx'),
        figures_dir=os.path.join(directory, 'figures'),
        artifact_dir=os.path.join(directory, 'artifacts'),
    )


class TestLoadData(unittest.TestCase):

    def test_when_liar_missing_then_file_not_found_is_raised(self):
        config = ExperimentConfig(data=DataConfig(liar_dir='/tmp/no-liar-here'))
        with self.assertRaises(FileNotFoundError):
            load_data(config)

    def test_when_liar_present_then_loads_liar(self):
        directory = _write_liar(tempfile.mkdtemp())
        config = ExperimentConfig(data=DataConfig(liar_dir=directory))
        _, _, _, source = load_data(config)
        self.assertEqual(source, 'LIAR')


class TestMakeSplits(unittest.TestCase):

    def test_when_called_then_texts_and_labels_align(self):
        directory = _write_liar(tempfile.mkdtemp())
        config = ExperimentConfig(data=DataConfig(liar_dir=directory))
        frames = load_data(config)[:3]
        (train_texts, train_labels), _, _ = make_splits(frames, config)
        self.assertEqual(len(train_texts), len(train_labels))


class TestCollectExamples(unittest.TestCase):

    def test_when_called_then_mixes_correct_and_incorrect(self):
        texts = ['a', 'b', 'c', 'd']
        references = [0, 1, 0, 1]
        predictions = [0, 1, 1, 0]
        names = ['fake', 'real']
        examples = _collect_examples(texts, references, predictions, names, limit=4)
        self.assertEqual(len(examples), 4)
        self.assertIn(examples[0][1], names)


class TestRun(unittest.TestCase):

    def test_when_run_then_report_and_figures_are_produced(self):
        directory = tempfile.mkdtemp()
        liar_dir = _write_liar(tempfile.mkdtemp())
        config = _tiny_config(directory, liar_dir=liar_dir)
        exit_code = run(config)
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(os.path.join(directory, 'report_six.xlsx')))
        self.assertTrue(
            os.path.exists(os.path.join(directory, 'figures', 'six', 'confusion_matrix.png')))


if __name__ == '__main__':
    unittest.main()

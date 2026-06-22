import os
import tempfile
import unittest

from src.fake_news.config import DataConfig, ExperimentConfig, RNNConfig
from src.fake_news.data.dataset import generate_synthetic_liar
from src.fake_news.main import _collect_examples, load_data, make_splits, run


def _write_liar(directory):
    rows = []
    frame = generate_synthetic_liar(n_per_class=3, seed=1)
    for _, record in frame.iterrows():
        rows.append('\t'.join([
            record['id'], record['label'], record['statement'], record['subject'],
            record['speaker'], record['job'], record['state'], record['party'],
            str(record['barely_true_count']),
            str(record['false_count']),
            str(record['half_true_count']),
            str(record['mostly_true_count']),
            str(record['pants_on_fire_count']), record['context']
        ]))
    for name in ('train.tsv', 'valid.tsv', 'test.tsv'):
        with open(os.path.join(directory, name), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(rows) + '\n')


def _tiny_config(directory: str, liar_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(liar_dir=liar_dir,
                        task='six',
                        min_token_frequency=1,
                        max_sequence_length=16,
                        embedding_dim=16),
        rnn=RNNConfig(embedding_dim=16, hidden_size=8, epochs=1, patience=1, batch_size=32),
        report_path=os.path.join(directory, 'report.xlsx'),
        figures_dir=os.path.join(directory, 'figures'),
    )


class TestLoadData(unittest.TestCase):

    def test_when_liar_missing_then_falls_back_to_synthetic(self):
        config = ExperimentConfig(data=DataConfig(liar_dir='/tmp/no-liar-here'))
        train, val, test, source = load_data(config)
        self.assertEqual(source, 'synthetic')
        self.assertGreater(len(train), 0)

    def test_when_liar_present_then_loads_liar(self):
        directory = tempfile.mkdtemp()
        _write_liar(directory)
        config = ExperimentConfig(data=DataConfig(liar_dir=directory))
        _, _, _, source = load_data(config)
        self.assertEqual(source, 'LIAR')


class TestMakeSplits(unittest.TestCase):

    def test_when_called_then_texts_and_labels_align(self):
        config = ExperimentConfig(data=DataConfig(liar_dir='/tmp/no-liar-here'))
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
        config = _tiny_config(directory, liar_dir='/tmp/no-liar-here')
        exit_code = run(config)
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(os.path.join(directory, 'report_six.xlsx')))
        self.assertTrue(
            os.path.exists(os.path.join(directory, 'figures', 'six', 'confusion_matrix.png')))


if __name__ == '__main__':
    unittest.main()

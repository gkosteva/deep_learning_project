"""Experiment 08 - improvement 5: BiLSTM with GloVe-initialised embeddings.

Place ``glove.6B.100d.txt`` under ``data/raw/`` to use the real vectors; without
it the loader synthesizes deterministic vectors so the experiment still runs.
"""
import torch

from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.config import LSTMConfig
from src.fake_news.data.embeddings import (build_embedding_matrix, load_or_synthesize_glove)
from src.fake_news.main import run_lstm_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary = prepare()
    vectors, source = load_or_synthesize_glove(config.data.glove_path, vocabulary,
                                               config.data.glove_dim, config.data.seed)
    matrix = build_embedding_matrix(vocabulary, vectors, config.data.glove_dim, config.data.seed)
    record, _, _, _ = run_lstm_experiment(
        'BiLSTM + GloVe (fine-tuned)',
        LSTMConfig(embedding_dim=config.data.glove_dim,
                   hidden_size=128,
                   bidirectional=True,
                   dropout=0.3),
        splits,
        vocabulary,
        config,
        f'Improvement 5: embedding layer initialised from GloVe ({source}) vectors '
        'and fine-tuned.',
        pretrained_embeddings=torch.tensor(matrix),
    )
    record.hyperparameters['embeddings'] = f'glove-{source}'
    append_record(LEDGER_PATH, record)
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'BiLSTM + GloVe metrics: {record.metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

"""Experiment 03 - improvement 1: add dropout regularisation."""
from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.config import LSTMConfig
from src.fake_news.main import run_lstm_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary = prepare()
    record, _, _, _ = run_lstm_experiment(
        'LSTM + dropout',
        LSTMConfig(dropout=0.3),
        splits,
        vocabulary,
        config,
        'Improvement 1: dropout 0.3 regularises the network to fight overfitting.',
    )
    append_record(LEDGER_PATH, record)
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'LSTM + dropout metrics: {record.metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.config import LSTMConfig
from src.fake_news.main import run_lstm_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary = prepare()
    record, _, _, _ = run_lstm_experiment(
        'BiLSTM + dropout',
        LSTMConfig(dropout=0.3, bidirectional=True),
        splits,
        vocabulary,
        config,
        'Improvement 2: bidirectional LSTM reads context both ways for richer features.',
    )
    append_record(LEDGER_PATH, record)
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'BiLSTM + dropout metrics: {record.metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

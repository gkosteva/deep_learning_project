from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.config import LSTMConfig
from src.fake_news.main import run_lstm_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary = prepare()
    record, _, _, _ = run_lstm_experiment(
        'Wider & deeper BiLSTM',
        LSTMConfig(embedding_dim=128,
                   hidden_size=128,
                   num_layers=2,
                   bidirectional=True,
                   dropout=0.3),
        splits,
        vocabulary,
        config,
        'Improvement 3: more capacity - wider embeddings/hidden state and a second '
        'LSTM layer.',
    )
    append_record(LEDGER_PATH, record)
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'Wider & deeper BiLSTM metrics: {record.metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

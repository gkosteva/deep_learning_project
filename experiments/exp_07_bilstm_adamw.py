from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.config import LSTMConfig
from src.fake_news.main import run_lstm_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary = prepare()
    record, _, _, _ = run_lstm_experiment(
        'Wider BiLSTM + AdamW weight decay',
        LSTMConfig(embedding_dim=128,
                   hidden_size=128,
                   num_layers=2,
                   bidirectional=True,
                   dropout=0.3,
                   weight_decay=0.01),
        splits,
        vocabulary,
        config,
        'Improvement 4: AdamW weight decay 0.01 regularises the larger model.',
    )
    append_record(LEDGER_PATH, record)
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'Wider BiLSTM + AdamW metrics: {record.metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

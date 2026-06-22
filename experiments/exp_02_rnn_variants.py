"""Step 4 - GRU and LSTM architecture variants."""
from dataclasses import replace

from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.main import run_rnn_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary, _, _ = prepare()
    base = config.rnn
    specs = [
        ('LSTM (learned embedding)', replace(base, rnn_type='lstm'),
         'Main model: learned Embedding + single-layer LSTM with masked mean pooling.'),
        ('GRU (learned embedding)', replace(base, rnn_type='gru'),
         'GRU variant with the same setup as the LSTM.'),
        ('BiLSTM + dropout', replace(base, rnn_type='lstm', bidirectional=True,
                                     dropout=0.3), 'Bidirectional LSTM with dropout 0.3.'),
        ('BiGRU + dropout', replace(base, rnn_type='gru', bidirectional=True,
                                    dropout=0.3), 'Bidirectional GRU with dropout 0.3.'),
        ('Wider & deeper BiLSTM',
         replace(base,
                 rnn_type='lstm',
                 hidden_size=192,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.3,
                 weight_decay=0.01),
         'More capacity: wider, two layers, dropout and AdamW weight decay.'),
    ]
    for name, rnn_config, comment in specs:
        record, *_ = run_rnn_experiment(name, rnn_config, splits, vocabulary, config, comment)
        append_record(LEDGER_PATH, record)
        print(f'{name}: {record.metrics}')
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    return 0


if __name__ == '__main__':
    exit(main())

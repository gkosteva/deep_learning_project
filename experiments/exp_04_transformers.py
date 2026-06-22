"""Step 5 - GPT and BERT architecture variants (fine-tuning).

Downloads pretrained weights from the HuggingFace hub on first run and is the
slowest experiment; defaults use small subsets/epochs from TransformerConfig.
"""
from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.main import run_transformer_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report

MODELS = [
    ('BERT (fine-tuned)', 'bert-base-uncased', 'BERT architecture fine-tuned on LIAR.'),
    ('RoBERTa (fine-tuned)', 'roberta-base', 'RoBERTa architecture fine-tuned on LIAR.'),
    ('DistilBERT (fine-tuned)', 'distilbert-base-uncased', 'Distilled BERT fine-tuned on LIAR.'),
    ('GPT-2 (fine-tuned)', 'gpt2', 'GPT architecture fine-tuned on LIAR.'),
]


def main() -> int:
    config, splits, _, _, _ = prepare()
    for name, model_name, comment in MODELS:
        record = run_transformer_experiment(name, model_name, splits, config, comment)
        if record is not None:
            append_record(LEDGER_PATH, record)
            print(f'{name}: {record.metrics}')
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    return 0


if __name__ == '__main__':
    exit(main())

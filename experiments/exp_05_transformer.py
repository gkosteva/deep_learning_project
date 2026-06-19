from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.main import POSITIVE_LABEL
from src.fake_news.reporting.ledger import append_record, rebuild_report
from src.fake_news.reporting.report_card import ExperimentRecord
from src.fake_news.training.metrics import evaluate_classification


def main() -> int:
    try:
        from src.fake_news.models.transformer_classifier import \
            fine_tune_distilbert
    except ImportError:
        print('transformers is not installed; skipping the DistilBERT experiment.')
        return 0

    config, splits, _ = prepare()
    (train_texts, train_labels), _, (test_texts, test_labels) = splits
    transformer_config = config.transformer

    train_texts = train_texts[:transformer_config.train_subset]
    train_labels = train_labels[:transformer_config.train_subset]
    eval_texts = test_texts[:transformer_config.eval_subset]
    eval_labels = test_labels[:transformer_config.eval_subset]

    predictions, summary = fine_tune_distilbert(train_texts, train_labels, eval_texts, eval_labels,
                                                transformer_config)
    metrics = evaluate_classification(eval_labels, predictions, POSITIVE_LABEL)

    hyperparameters = transformer_config.as_report_columns()
    hyperparameters.update(summary)
    append_record(
        LEDGER_PATH,
        ExperimentRecord(
            name='DistilBERT (fine-tuned)',
            hyperparameters=hyperparameters,
            metrics=metrics,
            comment='Stretch model: pre-trained transformer fine-tuned on a subset.',
        ))
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'DistilBERT metrics: {metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

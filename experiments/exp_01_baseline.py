import os

from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.main import POSITIVE_LABEL
from src.fake_news.models.baseline import MajorityClassClassifier
from src.fake_news.reporting.ledger import append_record, rebuild_report
from src.fake_news.reporting.report_card import ExperimentRecord
from src.fake_news.training.metrics import evaluate_classification


def main() -> int:
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    _, splits, _ = prepare()
    (_, train_labels), _, (test_texts, test_labels) = splits

    model = MajorityClassClassifier().fit(train_labels)
    predictions = model.predict(test_texts)
    metrics = evaluate_classification(test_labels, predictions, POSITIVE_LABEL)

    append_record(
        LEDGER_PATH,
        ExperimentRecord(
            name='Baseline (majority class)',
            hyperparameters={'strategy': 'most_frequent'},
            metrics=metrics,
            comment='Greediest statistical model: always predicts the majority class.',
        ))
    rebuild_report(LEDGER_PATH, REPORT_PATH)
    print(f'Baseline metrics: {metrics}')
    return 0


if __name__ == '__main__':
    exit(main())

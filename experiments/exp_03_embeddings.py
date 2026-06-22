"""Step 3 - word-embedding techniques: TF-IDF, Word2Vec, GloVe, FastText."""
from _common import LEDGER_PATH, REPORT_PATH, prepare

from src.fake_news.main import _embedding_experiment, run_tfidf_experiment
from src.fake_news.reporting.ledger import append_record, rebuild_report


def main() -> int:
    config, splits, vocabulary, _, _ = prepare()

    tfidf_record, _, _ = run_tfidf_experiment(splits, config)
    append_record(LEDGER_PATH, tfidf_record)
    print(f'TF-IDF: {tfidf_record.metrics}')

    for kind in ('word2vec', 'glove', 'fasttext'):
        record, *_ = _embedding_experiment(kind, splits, vocabulary, config)
        append_record(LEDGER_PATH, record)
        print(f'{record.name}: {record.metrics}')

    rebuild_report(LEDGER_PATH, REPORT_PATH)
    return 0


if __name__ == '__main__':
    exit(main())

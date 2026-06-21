import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fake_news.config import ExperimentConfig
from src.fake_news.data.dataset import stratified_split
from src.fake_news.main import build_vocabulary, load_dataframe

LEDGER_PATH = 'reports/records.jsonl'
REPORT_PATH = 'reports/model_report.xlsx'


def prepare(config: ExperimentConfig = None):
    config = config or ExperimentConfig()
    frame, source = load_dataframe(config)
    print(f'Loaded {len(frame)} articles from the {source} source.')
    train_df, val_df, test_df = stratified_split(frame, config.data.val_size,
                                                 config.data.test_size, config.data.seed)
    splits = (
        (train_df['text'].tolist(), train_df['label'].tolist()),
        (val_df['text'].tolist(), val_df['label'].tolist()),
        (test_df['text'].tolist(), test_df['label'].tolist()),
    )
    vocabulary = build_vocabulary(splits[0][0], config)
    return config, splits, vocabulary

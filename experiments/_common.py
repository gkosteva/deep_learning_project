import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fake_news.config import ExperimentConfig
from src.fake_news.main import build_vocabulary, load_data, make_splits

LEDGER_PATH = 'reports/records.jsonl'
REPORT_PATH = 'reports/model_report.xlsx'


def prepare(config: ExperimentConfig = None):
    config = config or ExperimentConfig()
    train_df, val_df, test_df, source = load_data(config)
    frames = (train_df, val_df, test_df)
    print(f'Loaded LIAR-style data ({source}): {len(train_df)} train / '
          f'{len(val_df)} val / {len(test_df)} test statements.')
    splits = make_splits(frames, config)
    vocabulary = build_vocabulary(splits[0][0], config)
    return config, splits, vocabulary, frames, source

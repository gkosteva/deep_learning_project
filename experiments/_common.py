"""Shared setup for the standalone experiment scripts.

Each ``exp_*.py`` script trains one model and appends its row to the shared
ledger, then rebuilds the Excel report. Run them in order (01 first, it resets
the ledger so the baseline is always row one).

Importing this module first puts the project root on ``sys.path`` so the scripts
work whether invoked as ``python experiments/exp_01_baseline.py`` or as a module.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fake_news.config import ExperimentConfig  # noqa: E402
from src.fake_news.data.dataset import stratified_split  # noqa: E402
from src.fake_news.main import build_vocabulary, load_dataframe  # noqa: E402

LEDGER_PATH = 'reports/records.jsonl'
REPORT_PATH = 'reports/model_report.xlsx'


def prepare(config: ExperimentConfig = None):
    """Load data, split it and build the vocabulary used by every experiment."""
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

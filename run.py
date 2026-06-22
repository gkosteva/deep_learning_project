import argparse

from src.fake_news import main
from src.fake_news.config import DataConfig, ExperimentConfig


def build_config(task: str, use_metadata: bool) -> ExperimentConfig:
    """Build the experiment config for the chosen classification task."""
    return ExperimentConfig(data=DataConfig(task=task, use_metadata=use_metadata))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the LIAR fake-news experiments.')
    parser.add_argument(
        '--task',
        choices=['six', 'binary'],
        default='six',
        help="Classification target: 'six' (full truth scale) or 'binary' "
        "(fake/real collapse). Default: six.",
    )
    parser.add_argument('--metadata',
                        action='store_true',
                        help='Append speaker/party/subject metadata to the statement text.')
    parser.add_argument('--transformers',
                        action='store_true',
                        help='Also fine-tune BERT/RoBERTa/DistilBERT/GPT-2 (slow, downloads).')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    config = build_config(args.task, args.metadata)
    exit(main.run(config, include_transformers=args.transformers))

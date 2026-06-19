import argparse

from src.fake_news import main
from src.fake_news.config import DataConfig, ExperimentConfig


def build_config(data_source: str) -> ExperimentConfig:
    """Build the experiment config for the chosen data source."""
    return ExperimentConfig(data=DataConfig(data_source=data_source))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the fake-news experiments.')
    parser.add_argument(
        '--data',
        choices=['auto', 'real', 'synthetic'],
        default='auto',
        help=("Which corpus to use: 'real' (ISOT only), 'synthetic' (generated "
              "only) or 'auto' (ISOT if present, else synthetic). Default: auto."),
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    exit(main.run(build_config(args.data)))

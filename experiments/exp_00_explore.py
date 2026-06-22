"""Step 2 - data exploration and analysis of the LIAR dataset."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from _common import prepare  # noqa: E402

from src.fake_news.config import LABELS_SIX  # noqa: E402
from src.fake_news.data.dataset import CREDIT_COLUMNS  # noqa: E402
from src.fake_news.reporting.plots import plot_class_distribution, plot_text_length_histogram \
    # noqa: E402

FIG_DIR = 'reports/figures/eda'


def _bar(series: pd.Series, title: str, path: str, ylabel: str = 'count') -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    series.plot(kind='bar', ax=axis, color='#1F4E78')
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis='x', rotation=40)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    _, _, _, frames, source = prepare()
    train_df, val_df, test_df = frames
    full = pd.concat(frames, ignore_index=True)

    print(f'\n=== LIAR exploration ({source}) ===')
    print(f'Observations: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, '
          f'total={len(full)}')
    print(f'Features ({full.shape[1]} columns): {list(full.columns)}')
    print('\nColumn dtypes:')
    print(full.dtypes)

    print('\nSix-class label distribution:')
    print(full['label_six'].value_counts().sort_index().rename(index={
        i: name
        for i, name in enumerate(LABELS_SIX)
    }))
    print('\nBinary label distribution (0=fake, 1=real):')
    print(full['label_binary'].value_counts().sort_index())

    lengths = full['statement'].str.split().apply(len)
    print('\nStatement length (words): '
          f'mean={lengths.mean():.1f}, median={lengths.median():.0f}, '
          f'min={lengths.min()}, max={lengths.max()}')

    print('\nAnomalies / data-quality checks:')
    print(f'  empty statements: {(full["statement"].str.strip() == "").sum()}')
    print(f'  missing speaker:  {(full["speaker"].str.strip() == "").sum()}')
    print(f'  missing context:  {(full["context"].str.strip() == "").sum()}')
    print(f'  duplicate statements: {full["statement"].duplicated().sum()}')

    print('\nTop 10 subjects:')
    subjects = full['subject'].str.split(',').explode().str.strip()
    print(subjects.value_counts().head(10))
    print('\nParty affiliation distribution:')
    print(full['party'].replace('', 'none').value_counts().head(8))

    correlations = full[CREDIT_COLUMNS +
                        ['label_binary']].corr()['label_binary'].drop('label_binary')
    print('\nCredit-history vs binary truthfulness correlation:')
    print(correlations)

    plot_class_distribution(full['label_six'].tolist(),
                            os.path.join(FIG_DIR, 'label_distribution_six.png'), LABELS_SIX)
    plot_class_distribution(full['label_binary'].tolist(),
                            os.path.join(FIG_DIR, 'label_distribution_binary.png'),
                            ['fake', 'real'])
    plot_text_length_histogram(full['statement'].tolist(),
                               os.path.join(FIG_DIR, 'statement_lengths.png'))
    _bar(full['party'].replace('', 'none').value_counts().head(8), 'Top party affiliations',
         os.path.join(FIG_DIR, 'party_distribution.png'))
    _bar(subjects.value_counts().head(10), 'Top 10 subjects',
         os.path.join(FIG_DIR, 'top_subjects.png'))
    _bar(correlations,
         'Credit-history correlation with truthfulness',
         os.path.join(FIG_DIR, 'credit_correlation.png'),
         ylabel='correlation')

    print(f'\nFigures written to {FIG_DIR}/')
    return 0


if __name__ == '__main__':
    exit(main())

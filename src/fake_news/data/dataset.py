"""Loading, splitting and tensorising the ISOT fake-news corpus.

Labels follow the convention used throughout the project:

* ``0`` -> fake news
* ``1`` -> real news

A synthetic generator is provided so the full pipeline (and the test-suite) can
run end to end without downloading the ~44k-article dataset.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessing import Vocabulary

LABEL_FAKE = 0
LABEL_REAL = 1


def _combine_title_and_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get('title', pd.Series([''] * len(frame))).fillna('')
    body = frame.get('text', pd.Series([''] * len(frame))).fillna('')
    return (title + '. ' + body).str.strip()


def load_isot(fake_path: str, true_path: str) -> pd.DataFrame:
    """Load and merge the ISOT ``Fake.csv`` and ``True.csv`` files.

    Returns a frame with two columns: ``text`` and ``label``.
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f'fake news file not found: {fake_path}')
    if not os.path.exists(true_path):
        raise FileNotFoundError(f'true news file not found: {true_path}')

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake_text = _combine_title_and_text(fake)
    true_text = _combine_title_and_text(true)

    frame = pd.DataFrame({
        'text': pd.concat([fake_text, true_text], ignore_index=True),
        'label': [LABEL_FAKE] * len(fake_text) + [LABEL_REAL] * len(true_text),
    })
    return frame


# A neutral, letters-only vocabulary shared by both classes. The classes differ
# only in how *often* they use each word, never in which words are available -
# this is what makes the synthetic task realistic instead of trivially separable.
_SYNTHETIC_VOCABULARY = [
    'time',
    'people',
    'year',
    'way',
    'day',
    'thing',
    'world',
    'school',
    'state',
    'family',
    'student',
    'group',
    'country',
    'problem',
    'hand',
    'place',
    'case',
    'week',
    'company',
    'system',
    'program',
    'question',
    'work',
    'government',
    'number',
    'night',
    'point',
    'home',
    'water',
    'room',
    'area',
    'money',
    'story',
    'fact',
    'month',
    'right',
    'study',
    'book',
    'job',
    'word',
]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def generate_synthetic_dataset(
    n_per_class: int = 300,
    seed: int = 42,
    separation: float = 0.7,
    noise: float = 0.12,
) -> pd.DataFrame:
    """Create a realistic-but-learnable synthetic corpus for demos and tests.

    Both classes share one vocabulary; only their word-frequency distributions
    differ, so the classes overlap and are *not* perfectly separable. With
    probability ``noise`` a document is drawn from the other class's
    distribution (feature noise), which caps the achievable accuracy the way
    real, messy data does. ``separation`` controls how far apart the two
    distributions are (higher -> easier).
    """
    rng = np.random.default_rng(seed)
    vocabulary = np.array(_SYNTHETIC_VOCABULARY)
    base = rng.normal(size=len(vocabulary))
    shift = rng.normal(size=len(vocabulary))
    distributions = {
        LABEL_FAKE: _softmax(base + separation * shift),
        LABEL_REAL: _softmax(base - separation * shift),
    }

    def make(label: int) -> str:
        source = 1 - label if rng.random() < noise else label
        length = int(rng.integers(12, 30))
        chosen = rng.choice(vocabulary, size=length, replace=True, p=distributions[source])
        return ' '.join(chosen)

    rows = []
    for _ in range(n_per_class):
        rows.append({'text': make(LABEL_FAKE), 'label': LABEL_FAKE})
        rows.append({'text': make(LABEL_REAL), 'label': LABEL_REAL})
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into train/val/test preserving the class ratio."""
    if not 0 < test_size < 1:
        raise ValueError('test_size must be in (0, 1)')
    if not 0 < val_size < 1:
        raise ValueError('val_size must be in (0, 1)')
    if val_size + test_size >= 1:
        raise ValueError('val_size + test_size must be < 1')

    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame['label'],
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val['label'],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


class FakeNewsDataset(Dataset):
    """Encodes raw texts into fixed-length id tensors for the LSTM models."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocabulary: Vocabulary,
        max_length: int,
    ):
        super().__init__()
        if len(texts) != len(labels):
            raise ValueError('texts and labels must have the same length')
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocabulary.encode(self.texts[index], self.max_length)
        features = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label

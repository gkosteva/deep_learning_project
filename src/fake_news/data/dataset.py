import csv
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import LABELS_SIX, SIX_TO_BINARY
from .preprocessing import Vocabulary

LIAR_COLUMNS = [
    'id',
    'label',
    'statement',
    'subject',
    'speaker',
    'job',
    'state',
    'party',
    'barely_true_count',
    'false_count',
    'half_true_count',
    'mostly_true_count',
    'pants_on_fire_count',
    'context',
]

CREDIT_COLUMNS = [
    'barely_true_count',
    'false_count',
    'half_true_count',
    'mostly_true_count',
    'pants_on_fire_count',
]

METADATA_TEXT_COLUMNS = ['subject', 'speaker', 'job', 'state', 'party', 'context']

_LABEL_TO_SIX = {label: index for index, label in enumerate(LABELS_SIX)}


def _build_meta_text(frame: pd.DataFrame) -> pd.Series:
    parts = [frame[column].fillna('').astype(str) for column in METADATA_TEXT_COLUMNS]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + ' ' + part
    return combined.str.replace(',', ' ', regex=False).str.strip()


def _finalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame['label'].isin(_LABEL_TO_SIX)].copy()  # drop rows with invalid labels
    frame['statement'] = frame['statement'].fillna('').astype(str)
    frame['label_six'] = frame['label'].map(_LABEL_TO_SIX).astype(int)
    frame['label_binary'] = frame['label_six'].map(SIX_TO_BINARY).astype(int)  # real/fake collapse
    for column in CREDIT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').fillna(0).astype(int)
    frame['meta_text'] = _build_meta_text(frame)
    return frame.reset_index(drop=True)


def load_liar_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'LIAR split not found: {path}')
    frame = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=LIAR_COLUMNS,
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
    )
    return _finalise_frame(frame)


def load_liar(liar_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = load_liar_split(os.path.join(liar_dir, 'train.tsv'))
    val = load_liar_split(os.path.join(liar_dir, 'valid.tsv'))
    test = load_liar_split(os.path.join(liar_dir, 'test.tsv'))
    return train, val, test


def label_column(task: str) -> str:
    if task == 'six':
        return 'label_six'
    if task == 'binary':
        return 'label_binary'
    raise ValueError(f"task must be 'six' or 'binary', got {task!r}")


def select_text(frame: pd.DataFrame, use_metadata: bool) -> List[str]:
    if use_metadata:
        combined = frame['statement'].astype(str) + ' ' + frame['meta_text'].astype(str)
        return combined.str.strip().tolist()
    return frame['statement'].astype(str).tolist()


class LiarDataset(Dataset):

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

import os
from typing import Dict, Tuple

import numpy as np

from .preprocessing import PAD_TOKEN, UNK_TOKEN, Vocabulary


def load_glove_vectors(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'glove file not found: {path}')
    vectors: Dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip().split(' ')
            if len(parts) < 2:
                continue
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def synthesize_glove_vectors(vocabulary: Vocabulary,
                             dim: int,
                             seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors: Dict[str, np.ndarray] = {}
    for token in vocabulary.token_to_id:
        if token in (PAD_TOKEN, UNK_TOKEN):
            continue
        vectors[token] = rng.normal(scale=0.5, size=dim).astype(np.float32)
    return vectors


def load_or_synthesize_glove(path: str,
                             vocabulary: Vocabulary,
                             dim: int,
                             seed: int = 42) -> Tuple[Dict[str, np.ndarray], str]:
    if os.path.exists(path):
        return load_glove_vectors(path), 'file'
    return synthesize_glove_vectors(vocabulary, dim, seed), 'synthetic'


def build_embedding_matrix(vocabulary: Vocabulary,
                           vectors: Dict[str, np.ndarray],
                           dim: int,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocabulary), dim)).astype(np.float32)
    matrix[vocabulary.pad_id] = 0.0
    for token, index in vocabulary.token_to_id.items():
        vector = vectors.get(token)
        if vector is not None:
            if vector.shape[0] != dim:
                raise ValueError(f'vector for {token!r} has dim {vector.shape[0]}, expected {dim}')
            matrix[index] = vector
    return matrix

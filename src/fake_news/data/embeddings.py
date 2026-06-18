"""Pretrained GloVe embeddings: loading and vocabulary alignment.

GloVe ships as a text file where each line is ``word v1 v2 ... vd``. We parse it
into a lookup, then build an embedding matrix whose row ``i`` is the vector for
the token with id ``i`` in our :class:`Vocabulary`. Tokens missing from GloVe
(out-of-vocabulary) get a small random vector; the padding row stays zero.

To use the real vectors, download ``glove.6B.zip`` from
https://nlp.stanford.edu/data/glove.6B.zip and place e.g. ``glove.6B.100d.txt``
under ``data/raw/``. When the file is absent we synthesize deterministic vectors
so the experiment still produces a report row.
"""
import os
from typing import Dict, Tuple

import numpy as np

from .preprocessing import PAD_TOKEN, UNK_TOKEN, Vocabulary


def load_glove_vectors(path: str) -> Dict[str, np.ndarray]:
    """Parse a GloVe text file into a ``{word: vector}`` mapping."""
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
    """Deterministic stand-in vectors used when no GloVe file is available."""
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
    """Return ``(vectors, source)`` where source is ``'file'`` or ``'synthetic'``."""
    if os.path.exists(path):
        return load_glove_vectors(path), 'file'
    return synthesize_glove_vectors(vocabulary, dim, seed), 'synthetic'


def build_embedding_matrix(vocabulary: Vocabulary,
                           vectors: Dict[str, np.ndarray],
                           dim: int,
                           seed: int = 42) -> np.ndarray:
    """Build a ``(vocab_size, dim)`` matrix aligned to the vocabulary ids."""
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

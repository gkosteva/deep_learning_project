import os
from typing import Dict, List, Tuple

import numpy as np

from .preprocessing import PAD_TOKEN, UNK_TOKEN, Vocabulary, tokenize


def load_vectors_file(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'vector file not found: {path}')
    vectors: Dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as handle:
        first = handle.readline().rstrip()
        parts = first.split(' ')
        if len(parts) > 2:
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        for line in handle:
            parts = line.rstrip().split(' ')
            if len(parts) < 3:
                continue
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def synthesize_vectors(vocabulary: Vocabulary, dim: int, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors: Dict[str, np.ndarray] = {}
    for token in vocabulary.token_to_id:
        if token in (PAD_TOKEN, UNK_TOKEN):
            continue
        vectors[token] = rng.normal(scale=0.5, size=dim).astype(np.float32)
    return vectors


def train_word2vec(texts: List[str],
                   dim: int,
                   seed: int = 42,
                   min_count: int = 1,
                   window: int = 5,
                   epochs: int = 10) -> Dict[str, np.ndarray]:
    from gensim.models import Word2Vec

    sentences = [tokenize(text) for text in texts]
    model = Word2Vec(sentences=sentences,
                     vector_size=dim,
                     window=window,
                     min_count=min_count,
                     workers=1,
                     seed=seed,
                     epochs=epochs)
    return {word: model.wv[word].astype(np.float32) for word in model.wv.index_to_key}


def train_fasttext(texts: List[str],
                   dim: int,
                   seed: int = 42,
                   min_count: int = 1,
                   window: int = 5,
                   epochs: int = 10) -> Dict[str, np.ndarray]:
    from gensim.models import FastText

    sentences = [tokenize(text) for text in texts]
    model = FastText(sentences=sentences,
                     vector_size=dim,
                     window=window,
                     min_count=min_count,
                     workers=1,
                     seed=seed,
                     epochs=epochs)
    return {word: model.wv[word].astype(np.float32) for word in model.wv.index_to_key}


def resolve_embeddings(
    kind: str,
    vocabulary: Vocabulary,
    texts: List[str],
    dim: int,
    seed: int = 42,
    path: str = '',
) -> Tuple[Dict[str, np.ndarray], str]:
    """Return word vectors for the requested technique with graceful fallbacks.

    - ``word2vec`` / ``fasttext`` are trained on the corpus via gensim; if gensim
      is unavailable, deterministic stand-in vectors are used instead.
    - ``glove`` is loaded from ``path`` when present, else synthesised.
    """
    if kind in ('word2vec', 'fasttext'):
        trainer = train_word2vec if kind == 'word2vec' else train_fasttext
        try:
            return trainer(texts, dim, seed), f'{kind}-trained'
        except ImportError:  # pragma: no cover - exercised only without gensim installed
            return synthesize_vectors(vocabulary, dim, seed), f'{kind}-synthetic'
    if kind == 'glove':
        if path and os.path.exists(path):
            return load_vectors_file(path), 'glove-file'
        return synthesize_vectors(vocabulary, dim, seed), 'glove-synthetic'
    raise ValueError(f'unknown embedding kind: {kind!r}')


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

"""The greediest statistical model: always predict the most frequent class.

This is the baseline (first row) of the Model Report File. Every other model is
judged by the percentage change of its metrics relative to this one.
"""
from collections import Counter
from typing import List, Sequence


class MajorityClassClassifier:
    """Predicts the single most frequent class observed during ``fit``."""

    def __init__(self) -> None:
        self._majority_class: int | None = None

    @property
    def majority_class(self) -> int:
        if self._majority_class is None:
            raise ValueError('classifier must be fitted before access')
        return self._majority_class

    def fit(self, labels: Sequence[int]) -> 'MajorityClassClassifier':
        """Learn the majority class from training labels."""
        if len(labels) == 0:
            raise ValueError('cannot fit on an empty label sequence')
        counts = Counter(labels)
        # ``most_common`` breaks ties by insertion order; sort for determinism.
        best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        self._majority_class = best[0]
        return self

    def predict(self, samples: Sequence[object]) -> List[int]:
        """Return the majority class for every input sample."""
        return [self.majority_class] * len(samples)

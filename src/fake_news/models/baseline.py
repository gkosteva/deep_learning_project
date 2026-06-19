from collections import Counter
from typing import List, Sequence


class MajorityClassClassifier:

    def __init__(self) -> None:
        self._majority_class: int | None = None

    @property
    def majority_class(self) -> int:
        if self._majority_class is None:
            raise ValueError('classifier must be fitted before access')
        return self._majority_class

    def fit(self, labels: Sequence[int]) -> 'MajorityClassClassifier':
        if len(labels) == 0:
            raise ValueError('cannot fit on an empty label sequence')
        counts = Counter(labels)
        best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        self._majority_class = best[0]
        return self

    def predict(self, samples: Sequence[object]) -> List[int]:
        return [self.majority_class] * len(samples)

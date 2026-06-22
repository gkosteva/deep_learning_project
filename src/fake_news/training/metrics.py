from typing import Dict, List, Sequence, Tuple


def _validate(references: Sequence[int], predictions: Sequence[int]) -> None:
    if len(references) != len(predictions):
        raise ValueError('references and predictions must have the same length')
    if len(references) == 0:
        raise ValueError('cannot compute metrics on empty sequences')


def accuracy_score(references: Sequence[int], predictions: Sequence[int]) -> float:
    _validate(references, predictions)
    correct = sum(1 for r, p in zip(references, predictions) if r == p)
    return correct / len(references)


def confusion_matrix(
    references: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> List[List[int]]:
    _validate(references, predictions)
    if num_classes < 2:
        raise ValueError('num_classes must be at least 2')
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for reference, prediction in zip(references, predictions):
        matrix[reference][prediction] += 1
    return matrix


def per_class_scores(matrix: Sequence[Sequence[int]]) -> List[Tuple[float, float, float]]:
    """Precision, recall and F1 for every class given a confusion matrix."""
    size = len(matrix)
    column_sums = [sum(matrix[row][col] for row in range(size)) for col in range(size)]
    row_sums = [sum(row) for row in matrix]
    scores: List[Tuple[float, float, float]] = []
    for index in range(size):
        true_pos = matrix[index][index]
        precision = true_pos / column_sums[index] if column_sums[index] else 0.0
        recall = true_pos / row_sums[index] if row_sums[index] else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        scores.append((precision, recall, f1))
    return scores


def _macro(references: Sequence[int], predictions: Sequence[int],
           num_classes: int) -> Tuple[float, float, float]:
    matrix = confusion_matrix(references, predictions, num_classes)
    scores = per_class_scores(matrix)
    size = len(scores)
    macro_precision = sum(score[0] for score in scores) / size
    macro_recall = sum(score[1] for score in scores) / size
    macro_f1 = sum(score[2] for score in scores) / size
    return macro_precision, macro_recall, macro_f1


def macro_precision_score(references, predictions, num_classes: int) -> float:
    return _macro(references, predictions, num_classes)[0]


def macro_recall_score(references, predictions, num_classes: int) -> float:
    return _macro(references, predictions, num_classes)[1]


def macro_f1_score(references, predictions, num_classes: int) -> float:
    return _macro(references, predictions, num_classes)[2]


def evaluate_classification(
    references: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> Dict[str, float]:
    macro_precision, macro_recall, macro_f1 = _macro(references, predictions, num_classes)
    return {
        'accuracy': accuracy_score(references, predictions),
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
    }

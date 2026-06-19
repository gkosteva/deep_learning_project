from typing import Dict, List, Sequence, Tuple


def _validate(references: Sequence[int], predictions: Sequence[int]) -> None:
    if len(references) != len(predictions):
        raise ValueError('references and predictions must have the same length')
    if len(references) == 0:
        raise ValueError('cannot compute metrics on empty sequences')


def confusion_counts(
    references: Sequence[int],
    predictions: Sequence[int],
    positive_label: int = 0,
) -> Tuple[int, int, int, int]:
    _validate(references, predictions)
    true_pos = false_pos = true_neg = false_neg = 0
    for reference, prediction in zip(references, predictions):
        if prediction == positive_label:
            if reference == positive_label:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if reference == positive_label:
                false_neg += 1
            else:
                true_neg += 1
    return true_pos, false_pos, true_neg, false_neg


def accuracy_score(references: Sequence[int], predictions: Sequence[int]) -> float:
    _validate(references, predictions)
    correct = sum(1 for r, p in zip(references, predictions) if r == p)
    return correct / len(references)


def precision_score(
    references: Sequence[int],
    predictions: Sequence[int],
    positive_label: int = 0,
) -> float:
    true_pos, false_pos, _, _ = confusion_counts(references, predictions, positive_label)
    denominator = true_pos + false_pos
    return true_pos / denominator if denominator else 0.0


def recall_score(
    references: Sequence[int],
    predictions: Sequence[int],
    positive_label: int = 0,
) -> float:
    true_pos, _, _, false_neg = confusion_counts(references, predictions, positive_label)
    denominator = true_pos + false_neg
    return true_pos / denominator if denominator else 0.0


def f1_score(
    references: Sequence[int],
    predictions: Sequence[int],
    positive_label: int = 0,
) -> float:
    precision = precision_score(references, predictions, positive_label)
    recall = recall_score(references, predictions, positive_label)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def confusion_matrix(
    references: Sequence[int],
    predictions: Sequence[int],
    num_classes: int = 2,
) -> List[List[int]]:
    _validate(references, predictions)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for reference, prediction in zip(references, predictions):
        matrix[reference][prediction] += 1
    return matrix


def evaluate_classification(
    references: Sequence[int],
    predictions: Sequence[int],
    positive_label: int = 0,
) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(references, predictions),
        'f1': f1_score(references, predictions, positive_label),
        'recall': recall_score(references, predictions, positive_label),
    }

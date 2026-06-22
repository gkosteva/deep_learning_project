import os
from typing import List, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from ..training.trainer import TrainingHistory  # noqa: E402


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_class_distribution(labels: Sequence[int], path: str, class_names: Sequence[str]) -> str:
    _ensure_dir(path)
    counts = [sum(1 for label in labels if label == index) for index in range(len(class_names))]
    colors = plt.cm.viridis([i / max(1, len(class_names) - 1) for i in range(len(class_names))])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(class_names, counts, color=colors)
    axis.set_title('Label distribution')
    axis.set_ylabel('Number of statements')
    axis.tick_params(axis='x', rotation=30)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_text_length_histogram(texts: Sequence[str], path: str, bins: int = 30) -> str:
    _ensure_dir(path)
    lengths = [len(text.split()) for text in texts]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(lengths, bins=bins, color='#1F4E78')
    axis.set_title('Statement length distribution')
    axis.set_xlabel('Words per statement')
    axis.set_ylabel('Frequency')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_training_curves(history: TrainingHistory, metric_path: str, loss_path: str) -> List[str]:
    _ensure_dir(metric_path)
    _ensure_dir(loss_path)
    epochs = range(1, len(history.train_loss) + 1)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, history.train_f1, marker='o', label='train macro-F1')
    axis.plot(epochs, history.val_f1, marker='o', label='validation macro-F1')
    axis.set_title('Train vs validation macro-F1')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Macro-F1')
    axis.legend()
    figure.tight_layout()
    figure.savefig(metric_path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, history.train_loss, marker='o', label='train loss')
    axis.plot(epochs, history.val_loss, marker='o', label='validation loss')
    axis.set_title('Train vs validation loss')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_path)
    plt.close(figure)
    return [metric_path, loss_path]


def plot_confusion_matrix(matrix: Sequence[Sequence[int]], path: str,
                          class_names: Sequence[str]) -> str:
    _ensure_dir(path)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap='Blues')
    axis.set_xticks(range(len(class_names)), labels=class_names)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.tick_params(axis='x', rotation=45)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    axis.set_title('Confusion matrix (best model)')
    threshold = max((max(row) for row in matrix), default=0) / 2
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = 'white' if value > threshold else 'black'
            axis.text(j, i, str(value), ha='center', va='center', color=color)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import macro_f1_score


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        num_classes: int = 2,
        weight_decay: float = 0.0,
    ):
        self.device = device or select_device()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
    ) -> Tuple[float, List[int], List[int]]:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_samples = 0
        references: List[int] = []
        predictions: List[int] = []

        with torch.set_grad_enabled(train):
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                references.extend(labels.tolist())
                predictions.extend(torch.argmax(logits, dim=1).tolist())

        mean_loss = total_loss / total_samples if total_samples else 0.0
        return mean_loss, references, predictions

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 2,
    ) -> TrainingHistory:
        history = TrainingHistory()
        best_val_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0

        for _ in range(epochs):
            train_loss, train_refs, train_preds = self._run_epoch(train_loader, train=True)
            val_loss, val_refs, val_preds = self._run_epoch(val_loader, train=False)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_f1.append(macro_f1_score(train_refs, train_preds, self.num_classes))
            current_val_f1 = macro_f1_score(val_refs, val_preds, self.num_classes)
            history.val_f1.append(current_val_f1)

            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, loader: DataLoader) -> Tuple[List[int], List[int]]:
        _, references, predictions = self._run_epoch(loader, train=False)
        return references, predictions

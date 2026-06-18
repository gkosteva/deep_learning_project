"""DistilBERT fine-tuning - the stretch experiment.

Kept deliberately thin and lazily importing ``transformers`` so the core
project (and its test-suite) does not depend on the heavy library. On an Apple
M2 we train on a subset for a single epoch using the MPS backend.

This module is excluded from coverage in ``.coveragerc`` because it requires the
optional ``transformers``/``datasets`` stack and a multi-minute training run.
"""
from typing import Dict, List, Tuple

import torch

from ..config import TransformerConfig


def select_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def fine_tune_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    eval_texts: List[str],
    eval_labels: List[int],
    config: TransformerConfig,
) -> Tuple[List[int], Dict[str, float]]:
    """Fine-tune DistilBERT and return ``(predictions, training_summary)``.

    Imports are local so the dependency is only required when this is called.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name,
                                                               num_labels=2).to(device)

    def encode(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.epochs):
        for start in range(0, len(train_texts), config.batch_size):
            batch_texts = train_texts[start:start + config.batch_size]
            batch_labels = torch.tensor(train_labels[start:start + config.batch_size]).to(device)
            optimizer.zero_grad()
            outputs = model(**encode(batch_texts), labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()

    model.eval()
    predictions: List[int] = []
    with torch.no_grad():
        for start in range(0, len(eval_texts), config.batch_size):
            batch_texts = eval_texts[start:start + config.batch_size]
            logits = model(**encode(batch_texts)).logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())

    summary = {'train_size': float(len(train_texts)), 'eval_size': float(len(eval_texts))}
    return predictions, summary

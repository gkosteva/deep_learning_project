from typing import Dict, List, Tuple

import torch

from ..config import TransformerConfig


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def fine_tune_transformer(
    train_texts: List[str],
    train_labels: List[int],
    eval_texts: List[str],
    eval_labels: List[int],
    config: TransformerConfig,
    num_classes: int,
) -> Tuple[List[int], Dict[str, float]]:
    """Fine-tune any HuggingFace sequence-classification model (BERT, RoBERTa,
    DistilBERT, GPT-2, ...) for `num_classes` labels and return test predictions."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        # Decoder-only models such as GPT-2 ship without a padding token.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name,
                                                               num_labels=num_classes)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

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

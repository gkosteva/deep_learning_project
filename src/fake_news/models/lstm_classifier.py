"""LSTM-based text classifier - the project's main model.

The same class covers the two chosen improvements: ``dropout`` adds
regularisation and ``bidirectional=True`` turns it into a BiLSTM. This keeps the
modelling story in a single, well-tested component.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> dropout -> linear classifier head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        if num_classes < 2:
            raise ValueError('num_classes must be at least 2')

        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map a batch of ``(batch, seq_len)`` token ids to class logits.

        Articles are right-padded, so we mean-pool the LSTM outputs over the
        real (non-pad) positions instead of reading a single, likely-padded
        time step. This lets the model learn from short documents.
        """
        embedded = self.embedding(token_ids)
        outputs, _ = self.lstm(embedded)
        mask = (token_ids != self.pad_id).unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(summed / counts)
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        """Total number of trainable parameters (the ``numel`` trick)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

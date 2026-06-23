from typing import Optional

import torch
import torch.nn as nn

RNN_TYPES = ('lstm', 'gru')

# An embedding + (Bi)LSTM/(Bi)GRU classifier with masked mean pooling.
class RNNClassifier(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 6,
        rnn_type: str = 'lstm',
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_id: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        if num_classes < 2:
            raise ValueError('num_classes must be at least 2')
        if rnn_type not in RNN_TYPES:
            raise ValueError(f'rnn_type must be one of {RNN_TYPES}, got {rnn_type!r}')

        self.pad_id = pad_id
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        if pretrained_embeddings is not None:
            self._load_pretrained_embeddings(pretrained_embeddings, vocab_size, embedding_dim,
                                             freeze_embeddings)

        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
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

    def _load_pretrained_embeddings(self, weights: torch.Tensor, vocab_size: int,
                                    embedding_dim: int, freeze: bool) -> None:
        if tuple(weights.shape) != (vocab_size, embedding_dim):
            raise ValueError(f'pretrained embeddings have shape {tuple(weights.shape)}, '
                             f'expected {(vocab_size, embedding_dim)}')
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
            self.embedding.weight[self.pad_id].zero_()
        self.embedding.weight.requires_grad = not freeze

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        outputs, _ = self.rnn(embedded)
        mask = (token_ids != self.pad_id).unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(summed / counts)
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

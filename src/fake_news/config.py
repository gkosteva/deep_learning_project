from dataclasses import asdict, dataclass, field
from typing import Dict


@dataclass
class DataConfig:
    fake_path: str = 'data/raw/Fake.csv'
    true_path: str = 'data/raw/True.csv'
    data_source: str = 'auto'
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    strip_reuters_leakage: bool = True
    max_vocab_size: int = 20000
    min_token_frequency: int = 2
    max_sequence_length: int = 200
    glove_path: str = 'data/raw/glove.6B.100d.txt'
    glove_dim: int = 100


@dataclass
class LSTMConfig:
    embedding_dim: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    patience: int = 2

    def as_report_columns(self) -> Dict[str, object]:
        return {
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }


@dataclass
class TransformerConfig:
    model_name: str = 'distilbert-base-uncased'
    learning_rate: float = 5e-5
    batch_size: int = 16
    epochs: int = 1
    max_length: int = 128
    train_subset: int = 4000
    eval_subset: int = 1000

    def as_report_columns(self) -> Dict[str, object]:
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'max_length': self.max_length,
        }


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    report_path: str = 'reports/model_report.xlsx'
    figures_dir: str = 'reports/figures'

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

from dataclasses import asdict, dataclass, field
from typing import Dict, List

# The six LIAR truthfulness labels, ordered from most false to most true.
LABELS_SIX: List[str] = [
    'pants-fire',
    'false',
    'barely-true',
    'half-true',
    'mostly-true',
    'true',
]

# Binary collapse of the six-way scale: the three "false-ish" labels map to
# fake (0) and the three "true-ish" labels map to real (1).
SIX_TO_BINARY: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
LABELS_BINARY: List[str] = ['fake', 'real']

TASKS = ('six', 'binary')


def class_names(task: str) -> List[str]:
    if task == 'six':
        return list(LABELS_SIX)
    if task == 'binary':
        return list(LABELS_BINARY)
    raise ValueError(f"task must be one of {TASKS}, got {task!r}")


def num_classes(task: str) -> int:
    return len(class_names(task))


@dataclass
class DataConfig:
    liar_dir: str = 'data/raw/liar'
    task: str = 'six'
    use_metadata: bool = False
    seed: int = 42
    max_vocab_size: int = 20000
    min_token_frequency: int = 2
    max_sequence_length: int = 64
    embedding_dim: int = 100
    glove_path: str = 'data/raw/glove.6B.100d.txt'
    word2vec_path: str = 'data/raw/word2vec.txt'
    fasttext_path: str = 'data/raw/fasttext.vec'


@dataclass
class RNNConfig:
    rnn_type: str = 'lstm'
    embedding_dim: int = 100
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 6
    patience: int = 3

    def as_report_columns(self) -> Dict[str, object]:
        return {
            'rnn_type': self.rnn_type,
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
    max_length: int = 64
    train_subset: int = 2000
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
    rnn: RNNConfig = field(default_factory=RNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    report_path: str = 'reports/model_report.xlsx'
    figures_dir: str = 'reports/figures'

    @property
    def num_classes(self) -> int:
        return num_classes(self.data.task)

    @property
    def class_names(self) -> List[str]:
        return class_names(self.data.task)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

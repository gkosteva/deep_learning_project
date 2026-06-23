import os
from typing import Dict, List, Tuple

import torch

from .config import ExperimentConfig, class_names
from .data.preprocessing import Vocabulary
from .models.rnn_classifier import RNNClassifier

DEFAULT_ARTIFACT_DIR = 'reports/artifacts'


def artifact_path(task: str, artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> str:
    return os.path.join(artifact_dir, f'best_model_{task}.pt')


def save_rnn_artifact(model: RNNClassifier,
                      vocabulary: Vocabulary,
                      config: ExperimentConfig,
                      model_name: str,
                      artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    payload = {
        'model_name': model_name,
        'task': config.data.task,
        'class_names': config.class_names,
        'max_sequence_length': config.data.max_sequence_length,
        'token_to_id': vocabulary.token_to_id,
        'arch': {
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_size': model.rnn.hidden_size,
            'num_layers': model.rnn.num_layers,
            'rnn_type': model.rnn_type,
            'bidirectional': model.rnn.bidirectional,
            'num_classes': model.classifier.out_features,
            'pad_id': model.pad_id,
        },
        'state_dict': model.state_dict(),
    }
    path = artifact_path(config.data.task, artifact_dir)
    torch.save(payload, path)
    return path

# Loads a saved RNN artifact and classifies free-text statements.
class PredictionService:

    def __init__(self, model: RNNClassifier, vocabulary: Vocabulary, labels: List[str],
                 max_length: int, model_name: str):
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.labels = labels
        self.max_length = max_length
        self.model_name = model_name

    @classmethod
    def load(cls, path: str) -> 'PredictionService':
        if not os.path.exists(path):
            raise FileNotFoundError(f'model artifact not found: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        arch = payload['arch']
        model = RNNClassifier(
            vocab_size=len(payload['token_to_id']),
            embedding_dim=arch['embedding_dim'],
            hidden_size=arch['hidden_size'],
            num_layers=arch['num_layers'],
            num_classes=arch['num_classes'],
            rnn_type=arch['rnn_type'],
            bidirectional=arch['bidirectional'],
            pad_id=arch['pad_id'],
        )
        model.load_state_dict(payload['state_dict'])
        vocabulary = Vocabulary(payload['token_to_id'])
        return cls(model, vocabulary, payload['class_names'], payload['max_sequence_length'],
                   payload['model_name'])

    def predict_proba(self, statement: str) -> List[Tuple[str, float]]:
        ids = self.vocabulary.encode(statement, self.max_length)
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()
        ranked = sorted(zip(self.labels, probabilities), key=lambda item: item[1], reverse=True)
        return ranked

    def predict(self, statement: str) -> str:
        return self.predict_proba(statement)[0][0]


def label_names(task: str) -> List[str]:
    return class_names(task)

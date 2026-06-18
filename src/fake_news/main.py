"""End-to-end pipeline: data -> baseline -> LSTM -> improvements -> report.

Running ``python run.py`` executes every experiment and writes the Model Report
File plus all figures. If the ISOT CSVs are not present under ``data/raw`` the
pipeline transparently falls back to a small synthetic corpus so the project is
runnable out of the box.
"""
import os
from dataclasses import replace
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, LSTMConfig
from .data.dataset import (FakeNewsDataset, generate_synthetic_dataset, load_isot,
                           stratified_split)
from .data.embeddings import build_embedding_matrix, load_or_synthesize_glove
from .data.preprocessing import Vocabulary
from .models.baseline import MajorityClassClassifier
from .models.lstm_classifier import LSTMClassifier
from .reporting.plots import (plot_class_distribution, plot_confusion_matrix,
                              plot_text_length_histogram, plot_training_curves)
from .reporting.report_card import ExperimentRecord, ModelReport
from .training.metrics import confusion_matrix, evaluate_classification
from .training.trainer import Trainer, TrainingHistory

POSITIVE_LABEL = 0  # 'fake' is the class we care about catching


def load_dataframe(config: ExperimentConfig):
    """Load the ISOT corpus, or a synthetic stand-in when it is missing."""
    try:
        frame = load_isot(config.data.fake_path, config.data.true_path)
        return frame, 'ISOT'
    except FileNotFoundError:
        frame = generate_synthetic_dataset(seed=config.data.seed)
        return frame, 'synthetic'


def build_vocabulary(train_texts: List[str], config: ExperimentConfig) -> Vocabulary:
    return Vocabulary.build(
        train_texts,
        max_size=config.data.max_vocab_size,
        min_frequency=config.data.min_token_frequency,
    )


def make_dataloader(texts, labels, vocabulary, config: ExperimentConfig, batch_size: int,
                    shuffle: bool) -> DataLoader:
    dataset = FakeNewsDataset(texts, labels, vocabulary, config.data.max_sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_lstm_experiment(
    name: str,
    lstm_config: LSTMConfig,
    splits,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    comment: str,
    device: Optional[torch.device] = None,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    freeze_embeddings: bool = False,
) -> Tuple[ExperimentRecord, TrainingHistory, List[int], List[int]]:
    """Train one LSTM variant and return its report record plus diagnostics."""
    torch.manual_seed(config.data.seed)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = splits

    train_loader = make_dataloader(train_texts,
                                   train_labels,
                                   vocabulary,
                                   config,
                                   lstm_config.batch_size,
                                   shuffle=True)
    val_loader = make_dataloader(val_texts,
                                 val_labels,
                                 vocabulary,
                                 config,
                                 lstm_config.batch_size,
                                 shuffle=False)
    test_loader = make_dataloader(test_texts,
                                  test_labels,
                                  vocabulary,
                                  config,
                                  lstm_config.batch_size,
                                  shuffle=False)

    model = LSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=lstm_config.embedding_dim,
        hidden_size=lstm_config.hidden_size,
        num_layers=lstm_config.num_layers,
        bidirectional=lstm_config.bidirectional,
        dropout=lstm_config.dropout,
        pad_id=vocabulary.pad_id,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
    )
    trainer = Trainer(model,
                      lstm_config.learning_rate,
                      device=device,
                      positive_label=POSITIVE_LABEL,
                      weight_decay=lstm_config.weight_decay)
    history = trainer.fit(train_loader, val_loader, lstm_config.epochs, lstm_config.patience)
    references, predictions = trainer.predict(test_loader)
    metrics = evaluate_classification(references, predictions, POSITIVE_LABEL)

    hyperparameters = lstm_config.as_report_columns()
    hyperparameters['parameters'] = model.count_parameters()
    record = ExperimentRecord(name=name,
                              hyperparameters=hyperparameters,
                              metrics=metrics,
                              comment=comment)
    return record, history, references, predictions


def run(config: Optional[ExperimentConfig] = None) -> int:
    """Execute the full experiment suite and write the report. Returns 0."""
    config = config or ExperimentConfig()
    frame, source = load_dataframe(config)
    print(f'Loaded {len(frame)} articles from the {source} source.')

    train_df, val_df, test_df = stratified_split(frame, config.data.val_size,
                                                 config.data.test_size, config.data.seed)
    splits = (
        (train_df['text'].tolist(), train_df['label'].tolist()),
        (val_df['text'].tolist(), val_df['label'].tolist()),
        (test_df['text'].tolist(), test_df['label'].tolist()),
    )
    vocabulary = build_vocabulary(splits[0][0], config)

    report = ModelReport(main_metric='f1')

    baseline = MajorityClassClassifier().fit(splits[0][1])
    baseline_predictions = baseline.predict(splits[2][0])
    baseline_metrics = evaluate_classification(splits[2][1], baseline_predictions, POSITIVE_LABEL)
    report.add(
        ExperimentRecord(
            name='Baseline (majority class)',
            hyperparameters={'strategy': 'most_frequent'},
            metrics=baseline_metrics,
            comment=('Greediest statistical model: always predicts the majority class. '
                     'Reference point for every other model.'),
        ))

    base = config.lstm
    bigger = replace(base,
                     embedding_dim=128,
                     hidden_size=128,
                     num_layers=2,
                     bidirectional=True,
                     dropout=0.3)
    experiments = [
        ('LSTM (main model)', base,
         'Embedding + single-layer LSTM. First real model in the journey.'),
        ('LSTM + dropout', replace(base, dropout=0.3),
         'Improvement 1: dropout 0.3 regularises the network to fight overfitting.'),
        ('BiLSTM + dropout', replace(base, dropout=0.3, bidirectional=True),
         'Improvement 2: bidirectional LSTM reads context both ways for richer features.'),
        ('Wider & deeper BiLSTM', bigger,
         'Improvement 3: more capacity - wider embeddings/hidden state and a second '
         'LSTM layer.'),
        ('Wider BiLSTM + AdamW weight decay', replace(bigger, weight_decay=0.01),
         'Improvement 4: AdamW weight decay 0.01 regularises the larger model.'),
    ]

    results = [
        run_lstm_experiment(name, lstm_config, splits, vocabulary, config, comment)
        for name, lstm_config, comment in experiments
    ]
    results.append(_run_glove_experiment(config, splits, vocabulary))
    for record, _, _, _ in results:
        report.add(record)

    best_index = max(range(len(results)), key=lambda i: results[i][0].metrics['f1'])
    _, best_history, best_refs, best_preds = results[best_index]

    diagram_paths = _generate_figures(config, splits, best_history, best_refs, best_preds)
    examples = _collect_examples(splits[2][0], best_refs, best_preds)
    report_path = report.save(config.report_path,
                              diagram_paths=diagram_paths[-2:],
                              examples=examples)
    print(f'Report written to {report_path}.')
    return 0


def _run_glove_experiment(config: ExperimentConfig, splits, vocabulary: Vocabulary):
    """Build a GloVe-initialised BiLSTM experiment (improvement 5)."""
    vectors, source = load_or_synthesize_glove(config.data.glove_path, vocabulary,
                                               config.data.glove_dim, config.data.seed)
    matrix = build_embedding_matrix(vocabulary, vectors, config.data.glove_dim, config.data.seed)
    embeddings = torch.tensor(matrix)
    glove_config = replace(config.lstm,
                           embedding_dim=config.data.glove_dim,
                           hidden_size=128,
                           bidirectional=True,
                           dropout=0.3)
    record, history, references, predictions = run_lstm_experiment(
        'BiLSTM + GloVe (fine-tuned)',
        glove_config,
        splits,
        vocabulary,
        config,
        f'Improvement 5: embedding layer initialised from GloVe ({source}) vectors '
        'and fine-tuned.',
        pretrained_embeddings=embeddings,
    )
    record.hyperparameters['embeddings'] = f'glove-{source}'
    return record, history, references, predictions


def _generate_figures(config, splits, history, references, predictions) -> List[str]:
    figures_dir = config.figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    all_labels = splits[0][1] + splits[1][1] + splits[2][1]
    all_texts = splits[0][0] + splits[1][0] + splits[2][0]
    paths = [
        plot_class_distribution(all_labels, os.path.join(figures_dir, 'class_distribution.png')),
        plot_text_length_histogram(all_texts, os.path.join(figures_dir, 'text_lengths.png')),
        plot_confusion_matrix(confusion_matrix(references, predictions),
                              os.path.join(figures_dir, 'confusion_matrix.png')),
    ]
    if history is not None:
        paths.extend(
            plot_training_curves(history, os.path.join(figures_dir, 'train_val_f1.png'),
                                 os.path.join(figures_dir, 'train_val_loss.png')))
    return paths


def _collect_examples(texts, references, predictions, limit: int = 6):
    """Pick a few correct and incorrect predictions for the best-model sheet."""
    correct = [(texts[i], references[i], predictions[i])
               for i in range(len(references))
               if references[i] == predictions[i]]
    wrong = [(texts[i], references[i], predictions[i])
             for i in range(len(references))
             if references[i] != predictions[i]]
    half = limit // 2
    return correct[:half] + wrong[:limit - half]

import os
from dataclasses import replace
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, RNNConfig
from .data.dataset import LiarDataset, generate_synthetic_liar, label_column, load_liar, \
    select_text
from .data.embeddings import build_embedding_matrix, resolve_embeddings
from .data.preprocessing import Vocabulary
from .inference import save_rnn_artifact
from .models.baseline import MajorityClassClassifier
from .models.rnn_classifier import RNNClassifier
from .models.tfidf_classifier import TfidfLogisticClassifier
from .reporting.plots import plot_class_distribution, plot_confusion_matrix, \
    plot_text_length_histogram, plot_training_curves
from .reporting.report_card import ExperimentRecord, ModelReport
from .training.metrics import confusion_matrix, evaluate_classification
from .training.trainer import Trainer, TrainingHistory


def load_data(config: ExperimentConfig):
    try:
        train, val, test = load_liar(config.data.liar_dir)
        return train, val, test, 'LIAR'
    except FileNotFoundError:
        frame = generate_synthetic_liar(seed=config.data.seed)
        n = len(frame)
        train = frame.iloc[:int(0.7 * n)].reset_index(drop=True)
        val = frame.iloc[int(0.7 * n):int(0.85 * n)].reset_index(drop=True)
        test = frame.iloc[int(0.85 * n):].reset_index(drop=True)
        return train, val, test, 'synthetic'


def make_splits(frames, config: ExperimentConfig):
    train_df, val_df, test_df = frames
    column = label_column(config.data.task)
    use_meta = config.data.use_metadata
    return (
        (select_text(train_df, use_meta), train_df[column].tolist()),
        (select_text(val_df, use_meta), val_df[column].tolist()),
        (select_text(test_df, use_meta), test_df[column].tolist()),
    )


def build_vocabulary(train_texts: List[str], config: ExperimentConfig) -> Vocabulary:
    return Vocabulary.build(
        train_texts,
        max_size=config.data.max_vocab_size,
        min_frequency=config.data.min_token_frequency,
    )


def make_dataloader(texts, labels, vocabulary, config: ExperimentConfig, batch_size: int,
                    shuffle: bool) -> DataLoader:
    dataset = LiarDataset(texts, labels, vocabulary, config.data.max_sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_rnn_experiment(
    name: str,
    rnn_config: RNNConfig,
    splits,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    comment: str,
    device: Optional[torch.device] = None,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    freeze_embeddings: bool = False,
) -> Tuple[ExperimentRecord, TrainingHistory, List[int], List[int], RNNClassifier]:
    torch.manual_seed(config.data.seed)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = splits
    num_classes = config.num_classes

    train_loader = make_dataloader(train_texts,
                                   train_labels,
                                   vocabulary,
                                   config,
                                   rnn_config.batch_size,
                                   shuffle=True)
    val_loader = make_dataloader(val_texts,
                                 val_labels,
                                 vocabulary,
                                 config,
                                 rnn_config.batch_size,
                                 shuffle=False)
    test_loader = make_dataloader(test_texts,
                                  test_labels,
                                  vocabulary,
                                  config,
                                  rnn_config.batch_size,
                                  shuffle=False)

    model = RNNClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=rnn_config.embedding_dim,
        hidden_size=rnn_config.hidden_size,
        num_layers=rnn_config.num_layers,
        num_classes=num_classes,
        rnn_type=rnn_config.rnn_type,
        bidirectional=rnn_config.bidirectional,
        dropout=rnn_config.dropout,
        pad_id=vocabulary.pad_id,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
    )
    trainer = Trainer(model,
                      rnn_config.learning_rate,
                      device=device,
                      num_classes=num_classes,
                      weight_decay=rnn_config.weight_decay)
    history = trainer.fit(train_loader, val_loader, rnn_config.epochs, rnn_config.patience)
    references, predictions = trainer.predict(test_loader)
    metrics = evaluate_classification(references, predictions, num_classes)

    hyperparameters = rnn_config.as_report_columns()
    hyperparameters['parameters'] = model.count_parameters()
    record = ExperimentRecord(name=name,
                              hyperparameters=hyperparameters,
                              metrics=metrics,
                              comment=comment)
    return record, history, references, predictions, model


def run_tfidf_experiment(
        splits, config: ExperimentConfig) -> Tuple[ExperimentRecord, List[int], List[int]]:
    (train_texts, train_labels), _, (test_texts, test_labels) = splits
    model = TfidfLogisticClassifier(max_features=config.data.max_vocab_size, seed=config.data.seed)
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    metrics = evaluate_classification(test_labels, predictions, config.num_classes)
    record = ExperimentRecord(
        name='TF-IDF + Logistic Regression',
        hyperparameters={
            'features': 'tfidf 1-2gram',
            'classifier': 'logreg'
        },
        metrics=metrics,
        comment='Sparse bag-of-words baseline embedding; strong, cheap reference for the RNNs.',
    )
    return record, test_labels, predictions


def run_transformer_experiment(name: str, model_name: str, splits, config: ExperimentConfig,
                               comment: str) -> Optional[ExperimentRecord]:  # pragma: no cover
    try:
        from .models.transformer_classifier import fine_tune_transformer
    except ImportError:
        print(f'transformers not available; skipping {name}.')
        return None

    (train_texts, train_labels), _, (test_texts, test_labels) = splits
    transformer_config = replace(config.transformer, model_name=model_name)
    train_texts = train_texts[:transformer_config.train_subset]
    train_labels = train_labels[:transformer_config.train_subset]
    eval_texts = test_texts[:transformer_config.eval_subset]
    eval_labels = test_labels[:transformer_config.eval_subset]

    predictions, summary = fine_tune_transformer(train_texts, train_labels, eval_texts,
                                                 eval_labels, transformer_config,
                                                 config.num_classes)
    metrics = evaluate_classification(eval_labels, predictions, config.num_classes)
    hyperparameters = transformer_config.as_report_columns()
    hyperparameters.update(summary)
    return ExperimentRecord(name=name,
                            hyperparameters=hyperparameters,
                            metrics=metrics,
                            comment=comment)


def _embedding_experiment(kind: str, splits, vocabulary, config: ExperimentConfig):
    dim = config.data.embedding_dim
    path = {
        'glove': config.data.glove_path,
        'word2vec': config.data.word2vec_path,
        'fasttext': config.data.fasttext_path,
    }.get(kind, '')
    vectors, source = resolve_embeddings(kind, vocabulary, splits[0][0], dim, config.data.seed,
                                         path)
    matrix = torch.tensor(build_embedding_matrix(vocabulary, vectors, dim, config.data.seed))
    rnn_config = replace(config.rnn,
                         rnn_type='lstm',
                         embedding_dim=dim,
                         hidden_size=128,
                         bidirectional=True,
                         dropout=0.3)
    record, history, refs, preds, model = run_rnn_experiment(
        f'BiLSTM + {kind} ({source})',
        rnn_config,
        splits,
        vocabulary,
        config,
        f'Embedding technique experiment: {kind} vectors ({source}) feeding a BiLSTM.',
        pretrained_embeddings=matrix,
    )
    record.hyperparameters['embeddings'] = source
    return record, history, refs, preds, model


def run(config: Optional[ExperimentConfig] = None, include_transformers: bool = False) -> int:
    config = config or ExperimentConfig()
    train_df, val_df, test_df, source = load_data(config)
    frames = (train_df, val_df, test_df)
    print(f'Loaded LIAR-style data ({source}): '
          f'{len(train_df)} train / {len(val_df)} val / {len(test_df)} test statements.')

    splits = make_splits(frames, config)
    vocabulary = build_vocabulary(splits[0][0], config)
    num_classes = config.num_classes
    class_names = config.class_names

    report = ModelReport(
        title=f'Model Report - LIAR Fake News Detection ({config.data.task}-class, {source})',
        main_metric='macro_f1')

    baseline = MajorityClassClassifier().fit(splits[0][1])
    baseline_predictions = baseline.predict(splits[2][0])
    baseline_metrics = evaluate_classification(splits[2][1], baseline_predictions, num_classes)
    report.add(
        ExperimentRecord(
            name='Baseline (majority class)',
            hyperparameters={
                'task': config.data.task,
                'strategy': 'most_frequent'
            },
            metrics=baseline_metrics,
            comment='Greediest statistical model: always predicts the majority class.',
        ))

    base = config.rnn
    rnn_specs = [
        ('LSTM (learned embedding)', replace(base, rnn_type='lstm'),
         'Main model: learned Embedding layer + single-layer LSTM, masked mean pooling.'),
        ('GRU (learned embedding)', replace(base, rnn_type='gru'),
         'GRU variant: lighter gating than the LSTM on the same setup.'),
        ('BiLSTM + dropout', replace(base, rnn_type='lstm', bidirectional=True, dropout=0.3),
         'Bidirectional LSTM with dropout reads context both ways and regularises.'),
        ('BiGRU + dropout', replace(base, rnn_type='gru', bidirectional=True, dropout=0.3),
         'Bidirectional GRU with dropout: the GRU counterpart of the BiLSTM.'),
        ('Wider & deeper BiLSTM',
         replace(base,
                 rnn_type='lstm',
                 hidden_size=192,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.3,
                 weight_decay=0.01),
         'More capacity: wider hidden state, two layers, dropout and AdamW weight decay.'),
    ]

    rnn_results = [
        run_rnn_experiment(name, cfg, splits, vocabulary, config, comment)
        for name, cfg, comment in rnn_specs
    ]

    embedding_results = [
        _embedding_experiment(kind, splits, vocabulary, config)
        for kind in ('word2vec', 'glove', 'fasttext')
    ]

    tfidf_record, tfidf_refs, tfidf_preds = run_tfidf_experiment(splits, config)
    report.add(tfidf_record)

    neural_results = rnn_results + embedding_results
    for record, *_ in neural_results:
        report.add(record)

    if include_transformers:  # pragma: no cover
        for name, model_name, comment in (
            ('BERT (fine-tuned)', 'bert-base-uncased', 'BERT architecture fine-tuned on LIAR.'),
            ('RoBERTa (fine-tuned)', 'roberta-base', 'RoBERTa architecture fine-tuned on LIAR.'),
            ('DistilBERT (fine-tuned)', 'distilbert-base-uncased',
             'Distilled BERT fine-tuned on LIAR.'),
            ('GPT-2 (fine-tuned)', 'gpt2', 'GPT architecture fine-tuned on LIAR.'),
        ):
            record = run_transformer_experiment(name, model_name, splits, config, comment)
            if record is not None:
                report.add(record)

    best_index = max(range(len(neural_results)),
                     key=lambda i: neural_results[i][0].metrics['macro_f1'])
    best_record, best_history, best_refs, best_preds, best_model = neural_results[best_index]

    figures_dir = os.path.join(config.figures_dir, config.data.task)
    figure_config = replace(config, figures_dir=figures_dir)
    diagram_paths = _generate_figures(figure_config, splits, best_history, best_refs, best_preds,
                                      class_names)
    examples = _collect_examples(splits[2][0], best_refs, best_preds, class_names)

    base_report, ext = os.path.splitext(config.report_path)
    report_path = f'{base_report}_{config.data.task}{ext}'
    saved_path = report.save(report_path, diagram_paths=diagram_paths[-2:], examples=examples)
    print(f'Report written to {saved_path}.')

    save_rnn_artifact(best_model, vocabulary, config, best_record.name)
    print(f'Best model ({best_record.name}) saved for the Streamlit app.')
    return 0


def _generate_figures(config, splits, history, references, predictions, class_names) -> List[str]:
    figures_dir = config.figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    all_labels = splits[0][1] + splits[1][1] + splits[2][1]
    all_texts = splits[0][0] + splits[1][0] + splits[2][0]
    paths = [
        plot_class_distribution(all_labels, os.path.join(figures_dir, 'class_distribution.png'),
                                class_names),
        plot_text_length_histogram(all_texts, os.path.join(figures_dir, 'text_lengths.png')),
        plot_confusion_matrix(confusion_matrix(references, predictions, len(class_names)),
                              os.path.join(figures_dir, 'confusion_matrix.png'), class_names),
    ]
    if history is not None:
        paths.extend(
            plot_training_curves(history, os.path.join(figures_dir, 'train_val_f1.png'),
                                 os.path.join(figures_dir, 'train_val_loss.png')))
    return paths


def _collect_examples(texts, references, predictions, class_names, limit: int = 6):

    def row(index):
        return (texts[index], class_names[references[index]], class_names[predictions[index]])

    correct = [row(i) for i in range(len(references)) if references[i] == predictions[i]]
    wrong = [row(i) for i in range(len(references)) if references[i] != predictions[i]]
    half = limit // 2
    return correct[:half] + wrong[:limit - half]

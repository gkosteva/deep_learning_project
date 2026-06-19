import json
import os
from typing import List

from .report_card import ExperimentRecord, ModelReport


def append_record(jsonl_path: str, record: ExperimentRecord) -> None:
    directory = os.path.dirname(jsonl_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        'name': record.name,
        'hyperparameters': record.hyperparameters,
        'metrics': record.metrics,
        'comment': record.comment,
    }
    with open(jsonl_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload) + '\n')


def load_records(jsonl_path: str) -> List[ExperimentRecord]:
    if not os.path.exists(jsonl_path):
        return []
    records: List[ExperimentRecord] = []
    with open(jsonl_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                ExperimentRecord(
                    name=payload['name'],
                    hyperparameters=payload['hyperparameters'],
                    metrics=payload['metrics'],
                    comment=payload['comment'],
                ))
    return records


def rebuild_report(jsonl_path: str, report_path: str, main_metric: str = 'f1') -> str:
    report = ModelReport(main_metric=main_metric)
    for record in load_records(jsonl_path):
        report.add(record)
    return report.save(report_path)

"""Persist experiment records as JSON lines so scripts can build up the report.

Each experiment script appends one record; rebuilding the workbook from the
ledger keeps the rows in creation order, exactly as the report rules require.
"""
import json
import os
from typing import List

from .report_card import ExperimentRecord, ModelReport


def append_record(jsonl_path: str, record: ExperimentRecord) -> None:
    """Append a single record to the ledger as one JSON line."""
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
    """Read every record from the ledger, preserving order."""
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
    """Rebuild the Excel report from all records currently in the ledger."""
    report = ModelReport(main_metric=main_metric)
    for record in load_records(jsonl_path):
        report.add(record)
    return report.save(report_path)

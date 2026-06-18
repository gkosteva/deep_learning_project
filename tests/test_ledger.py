import os
import tempfile
import unittest

from src.fake_news.reporting.ledger import (append_record, load_records, rebuild_report)
from src.fake_news.reporting.report_card import ExperimentRecord


def _record(name='Baseline'):
    return ExperimentRecord(name, {'strategy': 'most_frequent'}, {
        'accuracy': 0.5,
        'f1': 0.4,
        'recall': 0.5
    }, 'note')


class TestLoadRecords(unittest.TestCase):

    def test_when_file_missing_then_returns_empty_list(self):
        self.assertEqual(load_records('/tmp/does-not-exist.jsonl'), [])


class TestAppendRecord(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, 'records.jsonl')

    def test_when_records_appended_then_they_load_in_order(self):
        append_record(self.path, _record('A'))
        append_record(self.path, _record('B'))
        records = load_records(self.path)
        self.assertEqual([record.name for record in records], ['A', 'B'])

    def test_when_ledger_has_blank_lines_then_they_are_ignored(self):
        append_record(self.path, _record('A'))
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write('\n   \n')
        self.assertEqual(len(load_records(self.path)), 1)


class TestRebuildReport(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.ledger = os.path.join(self.directory, 'records.jsonl')
        self.report = os.path.join(self.directory, 'report.xlsx')

    def test_when_ledger_has_records_then_report_file_is_created(self):
        append_record(self.ledger, _record('A'))
        path = rebuild_report(self.ledger, self.report)
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()

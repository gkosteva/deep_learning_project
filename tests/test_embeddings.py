import os
import tempfile
import unittest

import numpy as np

from src.fake_news.data.embeddings import (build_embedding_matrix, load_glove_vectors,
                                           load_or_synthesize_glove, synthesize_glove_vectors)
from src.fake_news.data.preprocessing import Vocabulary


def _vocabulary():
    return Vocabulary.build(['cat dog bird'], min_frequency=1)


def _write_glove(directory, lines):
    path = os.path.join(directory, 'glove.txt')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


class TestLoadGloveVectors(unittest.TestCase):

    def test_when_file_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_glove_vectors('/tmp/no-such-glove.txt')

    def test_when_file_valid_then_vectors_are_parsed(self):
        directory = tempfile.mkdtemp()
        path = _write_glove(directory, ['cat 0.1 0.2', 'dog 0.3 0.4'])
        vectors = load_glove_vectors(path)
        self.assertEqual(set(vectors.keys()), {'cat', 'dog'})
        self.assertEqual(vectors['cat'].shape, (2, ))

    def test_when_line_has_no_vector_then_it_is_skipped(self):
        directory = tempfile.mkdtemp()
        path = _write_glove(directory, ['cat 0.1 0.2', 'lonely'])
        vectors = load_glove_vectors(path)
        self.assertNotIn('lonely', vectors)


class TestSynthesizeGloveVectors(unittest.TestCase):

    def test_when_called_then_special_tokens_are_excluded(self):
        vectors = synthesize_glove_vectors(_vocabulary(), dim=5)
        self.assertNotIn('<pad>', vectors)
        self.assertNotIn('<unk>', vectors)
        self.assertEqual(vectors['cat'].shape, (5, ))


class TestLoadOrSynthesizeGlove(unittest.TestCase):

    def test_when_file_exists_then_source_is_file(self):
        directory = tempfile.mkdtemp()
        path = _write_glove(directory, ['cat 0.1 0.2'])
        _, source = load_or_synthesize_glove(path, _vocabulary(), dim=2)
        self.assertEqual(source, 'file')

    def test_when_file_missing_then_source_is_synthetic(self):
        _, source = load_or_synthesize_glove('/tmp/missing.txt', _vocabulary(), dim=4)
        self.assertEqual(source, 'synthetic')


class TestBuildEmbeddingMatrix(unittest.TestCase):

    def test_when_built_then_shape_matches_vocabulary_and_dim(self):
        vocabulary = _vocabulary()
        matrix = build_embedding_matrix(vocabulary, {}, dim=3)
        self.assertEqual(matrix.shape, (len(vocabulary), 3))

    def test_when_built_then_pad_row_is_zero(self):
        vocabulary = _vocabulary()
        matrix = build_embedding_matrix(vocabulary, {}, dim=3)
        self.assertTrue(np.all(matrix[vocabulary.pad_id] == 0.0))

    def test_when_token_in_vectors_then_its_row_is_used(self):
        vocabulary = _vocabulary()
        vector = np.ones(3, dtype=np.float32)
        matrix = build_embedding_matrix(vocabulary, {'cat': vector}, dim=3)
        np.testing.assert_array_equal(matrix[vocabulary.token_to_id['cat']], vector)

    def test_when_vector_dim_mismatch_then_value_error_is_raised(self):
        vocabulary = _vocabulary()
        with self.assertRaises(ValueError):
            build_embedding_matrix(vocabulary, {'cat': np.ones(2, dtype=np.float32)}, dim=3)


if __name__ == '__main__':
    unittest.main()

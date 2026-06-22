import os
import tempfile
import unittest

import numpy as np

from src.fake_news.data.embeddings import build_embedding_matrix, load_vectors_file, \
    resolve_embeddings, synthesize_vectors, train_word2vec
from src.fake_news.data.preprocessing import Vocabulary


def _vocabulary():
    return Vocabulary.build(['cat dog bird'], min_frequency=1)


def _write_vectors(directory, lines):
    path = os.path.join(directory, 'vectors.txt')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


class TestLoadVectorsFile(unittest.TestCase):

    def test_when_file_missing_then_file_not_found_is_raised(self):
        with self.assertRaises(FileNotFoundError):
            load_vectors_file('/tmp/no-such-vectors.txt')

    def test_when_glove_format_then_vectors_are_parsed(self):
        directory = tempfile.mkdtemp()
        path = _write_vectors(directory, ['cat 0.1 0.2', 'dog 0.3 0.4'])
        vectors = load_vectors_file(path)
        self.assertEqual(set(vectors.keys()), {'cat', 'dog'})
        self.assertEqual(vectors['cat'].shape, (2, ))

    def test_when_word2vec_header_then_it_is_skipped(self):
        directory = tempfile.mkdtemp()
        path = _write_vectors(directory, ['2 2', 'cat 0.1 0.2', 'dog 0.3 0.4'])
        vectors = load_vectors_file(path)
        self.assertEqual(set(vectors.keys()), {'cat', 'dog'})

    def test_when_line_too_short_then_it_is_skipped(self):
        directory = tempfile.mkdtemp()
        path = _write_vectors(directory, ['cat 0.1 0.2', 'dog 0.3 0.4', 'lonely'])
        vectors = load_vectors_file(path)
        self.assertNotIn('lonely', vectors)


class TestSynthesizeVectors(unittest.TestCase):

    def test_when_called_then_special_tokens_are_excluded(self):
        vectors = synthesize_vectors(_vocabulary(), dim=5)
        self.assertNotIn('<pad>', vectors)
        self.assertNotIn('<unk>', vectors)
        self.assertEqual(vectors['cat'].shape, (5, ))


class TestResolveEmbeddings(unittest.TestCase):

    def test_when_glove_file_present_then_source_is_file(self):
        directory = tempfile.mkdtemp()
        path = _write_vectors(directory, ['cat 0.1 0.2', 'dog 0.3 0.4'])
        _, source = resolve_embeddings('glove', _vocabulary(), ['cat dog'], dim=2, path=path)
        self.assertEqual(source, 'glove-file')

    def test_when_glove_file_missing_then_source_is_synthetic(self):
        _, source = resolve_embeddings('glove', _vocabulary(), ['cat dog'], dim=4, path='')
        self.assertEqual(source, 'glove-synthetic')

    def test_when_word2vec_then_vectors_are_trained(self):
        vectors, source = resolve_embeddings('word2vec',
                                             _vocabulary(), ['cat dog bird', 'cat dog'],
                                             dim=8)
        self.assertEqual(source, 'word2vec-trained')
        self.assertIn('cat', vectors)

    def test_when_fasttext_then_vectors_are_trained(self):
        vectors, source = resolve_embeddings('fasttext',
                                             _vocabulary(), ['cat dog bird', 'cat dog'],
                                             dim=8)
        self.assertEqual(source, 'fasttext-trained')
        self.assertIn('cat', vectors)

    def test_when_kind_unknown_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            resolve_embeddings('bogus', _vocabulary(), ['cat'], dim=2)


class TestTrainWord2Vec(unittest.TestCase):

    def test_when_trained_then_vectors_have_requested_dim(self):
        vectors = train_word2vec(['cat dog bird', 'cat dog'], dim=6, epochs=1)
        self.assertEqual(vectors['cat'].shape, (6, ))


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

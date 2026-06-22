import unittest

from src.fake_news.data.preprocessing import PAD_TOKEN, UNK_TOKEN, Vocabulary, clean_text, tokenize


class TestCleanText(unittest.TestCase):

    def test_when_text_has_urls_and_punctuation_then_they_are_removed(self):
        text = 'Visit https://example.com NOW!!! Amazing, deal.'
        cleaned = clean_text(text)
        self.assertNotIn('http', cleaned)
        self.assertNotIn('!', cleaned)
        self.assertEqual(cleaned, 'visit now amazing deal')

    def test_when_text_has_digits_then_they_are_kept(self):
        self.assertEqual(clean_text('Spending rose 25 percent'), 'spending rose 25 percent')

    def test_when_text_is_none_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            clean_text(None)


class TestTokenize(unittest.TestCase):

    def test_when_text_has_words_then_returns_token_list(self):
        self.assertEqual(tokenize('Hello, World!'), ['hello', 'world'])

    def test_when_text_is_empty_then_returns_empty_list(self):
        self.assertEqual(tokenize('!!!'), [])


class TestVocabularyInit(unittest.TestCase):

    def test_when_special_tokens_missing_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            Vocabulary({'hello': 0})

    def test_when_built_then_pad_and_unk_ids_are_zero_and_one(self):
        vocabulary = Vocabulary({PAD_TOKEN: 0, UNK_TOKEN: 1})
        self.assertEqual(vocabulary.pad_id, 0)
        self.assertEqual(vocabulary.unk_id, 1)
        self.assertEqual(len(vocabulary), 2)


class TestVocabularyBuild(unittest.TestCase):

    def test_when_min_frequency_set_then_rare_tokens_excluded(self):
        texts = ['cat cat dog', 'cat dog', 'rare']
        vocabulary = Vocabulary.build(texts, min_frequency=2)
        self.assertIn('cat', vocabulary.token_to_id)
        self.assertIn('dog', vocabulary.token_to_id)
        self.assertNotIn('rare', vocabulary.token_to_id)

    def test_when_max_size_reached_then_extra_tokens_dropped(self):
        texts = ['a b c d e f g']
        vocabulary = Vocabulary.build(texts, max_size=4, min_frequency=1)
        self.assertEqual(len(vocabulary), 4)


class TestVocabularyEncode(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary.build(['cat dog bird'], min_frequency=1)

    def test_when_sequence_shorter_than_max_then_it_is_padded(self):
        encoded = self.vocabulary.encode('cat', max_length=4)
        self.assertEqual(len(encoded), 4)
        self.assertEqual(encoded[-1], self.vocabulary.pad_id)

    def test_when_sequence_longer_than_max_then_it_is_truncated(self):
        encoded = self.vocabulary.encode('cat dog bird cat dog', max_length=2)
        self.assertEqual(len(encoded), 2)

    def test_when_token_unknown_then_unk_id_is_used(self):
        encoded = self.vocabulary.encode('elephant', max_length=1)
        self.assertEqual(encoded[0], self.vocabulary.unk_id)

    def test_when_max_length_not_positive_then_value_error_is_raised(self):
        with self.assertRaises(ValueError):
            self.vocabulary.encode('cat', max_length=0)


if __name__ == '__main__':
    unittest.main()

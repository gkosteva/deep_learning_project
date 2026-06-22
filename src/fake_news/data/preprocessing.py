import re
from collections import Counter
from typing import Dict, List

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

_URL = re.compile(r'https?://\S+|www\.\S+')
_NON_LETTER = re.compile(r"[^a-z0-9\s']")
_MULTISPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    if text is None:
        raise ValueError('text must not be None')
    text = text.lower()
    text = _URL.sub(' ', text)
    text = _NON_LETTER.sub(' ', text)
    text = _MULTISPACE.sub(' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split(' ')


class Vocabulary:

    def __init__(self, token_to_id: Dict[str, int]):
        if PAD_TOKEN not in token_to_id or UNK_TOKEN not in token_to_id:
            raise ValueError('vocabulary must contain pad and unk tokens')
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, max_length: int) -> List[int]:
        if max_length <= 0:
            raise ValueError('max_length must be positive')
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]
        ids = ids[:max_length]
        padding = [self.pad_id] * (max_length - len(ids))
        return ids + padding

    @classmethod
    def build(
        cls,
        texts: List[str],
        max_size: int = 20000,
        min_frequency: int = 1,
    ) -> 'Vocabulary':
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        eligible = [(token, count) for token, count in counter.items() if count >= min_frequency]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        for token, _ in eligible:
            if len(token_to_id) >= max_size:
                break
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

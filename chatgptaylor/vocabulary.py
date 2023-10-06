import funcy

from typing import Dict, List

UNKNOWN = "UNK"


class Vocabulary:
    def __init__(
        self,
        unknown: str = UNKNOWN,
        init_tokens: List[str] = None,
    ):
        self._vocabulary = {
            unknown: 0,
        }
        self.__vocabulary_transposed = self._transpose_vocabulary(self._vocabulary)

        init_tokens = init_tokens or []
        self.add_tokens(init_tokens)
        self.unknown = unknown

    @staticmethod
    def _transpose_vocabulary(vocabulary: Dict[str, int]):
        return {
            i: token for token, i in vocabulary.items()
        }

    @property
    def _vocabulary_transposed(self):
        if len(self._vocabulary) > len(self.__vocabulary_transposed):
            self.__vocabulary_transposed = self._transpose_vocabulary(self._vocabulary)

        return self.__vocabulary_transposed

    def __len__(self):
        return len(self._vocabulary)

    def add_token(self, token: str):
        if token in self._vocabulary:
            return

        max_index = max(self._vocabulary_transposed)
        self._vocabulary[token] = max_index + 1

    def add_tokens(
        self,
        tokens: List[str],
        enforce_sorted: bool = False,
    ):
        if enforce_sorted:
            tokens = sorted(tokens)

        funcy.lmap(self.add_token, tokens)

    def encode(
        self,
        sentence: List[str],
    ):
        unknown_index = self._vocabulary[self.unknown]
        return [self._vocabulary.get(token, unknown_index) for token in sentence]

    def encode_batch(
        self,
        batch: List[List[str]],
    ):
        return funcy.lmap(self.encode, batch)

    def decode(
        self,
        sentence: List[int],
    ):
        return [self._vocabulary_transposed[token] for token in sentence]

    def decode_batch(
        self,
        batch: List[List[int]],
    ):
        return funcy.lmap(self.decode, batch)

    def __getitem__(self, item):
        return self._vocabulary[item]

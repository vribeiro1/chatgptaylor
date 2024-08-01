import functools
import pickle
import string

from tqdm import tqdm
from typing import List, Tuple

import regex as re
import torch

from chatgptaylor import TOKENIZER_VOCABULARY_FILEPATH, TOKENIZER_MERGES_FILEPATH

BYTES_PUNCTUATION = {c.encode("utf-8") for c in string.punctuation}


class Tokenizer:
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    END_OF_DOCUMENT = "<|end_of_document|>"

    def __init__(self):
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def dump(self, vocabulary_filepath: str, merges_filepath: str):
        with open(vocabulary_filepath, "wb") as f:
            pickle.dump(self.vocabulary, f)

        with open(merges_filepath, "wb") as f:
            pickle.dump(self.merges, f)

    def __len__(self):
        return len(self.vocabulary)

    @classmethod
    def load(cls, vocabulary_filepath: str = None, merges_filepath: str = None):
        vocabulary_filepath = vocabulary_filepath or TOKENIZER_VOCABULARY_FILEPATH
        merges_filepath = merges_filepath or TOKENIZER_MERGES_FILEPATH

        tokenizer = cls()

        with open(vocabulary_filepath, "rb") as f:
            tokenizer.vocabulary = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            tokenizer.merges = pickle.load(f)

        return tokenizer

    @staticmethod
    def get_pairs(sequence, skip_rules: List = None):
        skip_rules = skip_rules or []

        pairs = []
        for token1, token2 in zip(sequence[:-1], sequence[1:]):
            if any(rule(token1, token2) for rule in skip_rules):
                continue

            pairs.append((token1, token2))
        return pairs

    @staticmethod
    def get_stats(pairs: List[Tuple[str, str]]):
        counts = {}
        for pair in pairs:
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @classmethod
    def merge(cls, bytes, pair, new_token):
        new_bytes = []
        skip = False
        for i in range(len(bytes)):
            if skip:
                skip = False
                continue

            token = bytes[i]
            if i == len(bytes) - 1:
                new_bytes.append(token)
                continue

            next_token = bytes[i + 1]
            if (token, next_token) == pair:
                new_bytes.append(new_token)
                skip = True
            else:
                new_bytes.append(token)

        return new_bytes

    @classmethod
    def _train(cls, dataset, vocabulary, merges, num_merges: int = 50000):
        progress_bar = tqdm(range(num_merges), total=num_merges, desc="Training tokenizer")
        dataset = list(dataset.encode("utf-8"))

        for _ in progress_bar:
            skip_rules = [
                lambda token1, token2 : vocabulary[token1] == b"\n" or vocabulary[token2] == b"\n",
                lambda token1, token2 : (
                    vocabulary[token1] in BYTES_PUNCTUATION or
                    vocabulary[token2] in BYTES_PUNCTUATION
                ),
                lambda token1, _ : (
                    len(vocabulary[token1]) > 1 and vocabulary[token1].endswith(b" ")
                ),
                lambda _, token2 : vocabulary[token2].startswith(b" "),
            ]

            pairs = cls.get_pairs(dataset, skip_rules)
            if len(pairs) == 0:  # Nothing else to merge
                break

            stats = cls.get_stats(pairs)
            (most_common_pair, _) = max(
                [(k, v) for k, v in stats.items()], key=lambda args: args[1]
            )

            new_token = max(vocabulary) + 1
            token1, token2 = most_common_pair

            merges[most_common_pair] = new_token
            vocabulary[new_token] = vocabulary[token1] + vocabulary[token2]
            dataset = cls.merge(dataset, most_common_pair, new_token)

        return vocabulary, merges

    def train(self, filepaths: List[str]):
        contents = []
        for filepath in filepaths:
            with open(filepath, "r") as f:
                contents.append(f.read().strip())

        dataset = "\n\n".join(contents)
        vocabulary, merges = self._train(
            dataset, self.vocabulary, self.merges, num_merges=1500
        )

        self.vocabulary = vocabulary
        self.merges = merges

    def encode(self, text):
        parts_of_text = re.findall(self.GPT4_SPLIT_PATTERN, text)

        parts_text_bytes = []
        for part in parts_of_text:
            part_bytes = list(part.encode("utf-8"))

            while len(part_bytes) > 1:
                pairs = Tokenizer.get_pairs(part_bytes)
                stats = self.get_stats(pairs)
                lowest_index_pair = min(
                    stats, key=lambda pair: self.merges.get(pair, float("inf"))
                )

                if lowest_index_pair not in self.merges:
                    break  # Nothing else can be merged

                merge_token = self.merges[lowest_index_pair]
                part_bytes = Tokenizer.merge(part_bytes, lowest_index_pair, merge_token)

            parts_text_bytes.append(part_bytes)

        text_bytes = functools.reduce(lambda l1, l2 : l1 + l2, parts_text_bytes, [])
        return torch.tensor(text_bytes, dtype=torch.int32)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = [id_.item() for id_ in ids]
        text_bytes = b"".join(self.vocabulary[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
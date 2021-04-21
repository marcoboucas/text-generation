"""Tokenizer for ngrams."""
from functools import reduce
from dataprocessing.text_operations import process_text, split_to_tokens
from typing import List, Sequence


class NgramTokenizer():
    """Ngram Tokenizer."""

    def __init__(self, nbr_grams: int) -> None:
        """Init."""
        self.n = nbr_grams
        self.special_tokens = {
            "start": "<START>",
            "end": "<END>",
            "unknown": "<UNK>"
        }

    def train(self, texts: List[str]) -> None:
        """Find the vocabulary."""
        tokens = list(map(lambda x: set(split_to_tokens(process_text(x))), texts))
        tokens = reduce()
        




    def encode(self, text: Sequence[str]) -> List[List[int]]:
        """Encode a list of strings."""

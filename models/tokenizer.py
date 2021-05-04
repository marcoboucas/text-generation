"""Tokenizer for ngrams."""
import re
from string import ascii_letters
from functools import reduce
from dataprocessing.text_operations import process_text, split_to_tokens
from typing import List, Sequence, Dict


class Tokenizer:
    """Ngram Tokenizer."""

    def __init__(self, nbr_grams: int) -> None:
        """Init."""
        self.n = nbr_grams
        self.special_tokens = {"start": "<START>", "end": "<END>", "unknown": "<UNK>"}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    def train(self, texts: List[str], remove_numbers: bool = True) -> None:
        """Find the vocabulary."""
        if remove_numbers:

            def is_a_token(token: str) -> bool:
                """Check if this is a token worth taking."""
                return not re.match(r"\b[\d-]+\b", token)

        else:
            is_a_token = lambda x: True
        tokens = list(
            map(
                lambda x: set(filter(is_a_token, split_to_tokens(process_text(x)))),
                texts,
            )
        )
        tokens = reduce(lambda x, y: x | y, tokens)
        self.id_to_token = list(self.special_tokens.values()) + list(tokens)
        self.token_to_id = {x: i for i, x in enumerate(self.id_to_token)}

    def encode(self, texts: Sequence[str]) -> List[List[int]]:
        """Encode a list of strings."""
        tokens = map(lambda x: split_to_tokens(process_text(x)), texts)

        tokens = map(
            lambda x: [self.special_tokens["start"]] * self.n
            + x
            + [self.special_tokens["end"]],
            tokens,
        )
        tokens = map(
            lambda x: [
                self.token_to_id.get(
                    xi, self.token_to_id[self.special_tokens["unknown"]]
                )
                for xi in x
            ],
            tokens,
        )
        return list(tokens)

    def decode(self, token_ids: Sequence[Sequence[int]]) -> List[str]:
        """Decode the texts."""
        return list(
            map(
                lambda token_id: " ".join([self.id_to_token[x] for x in token_id]),
                token_ids,
            )
        )

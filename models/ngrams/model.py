"""Ngram model."""

from typing import List, Dict, Tuple
from random import randint, random, choices


class NgramModel:
    """Text Generation model based on ngrams."""

    def __init__(self, nbr_grams: int):
        """Init."""
        self.n = nbr_grams
        self.probas: Dict[Tuple[int, ...], Dict[int, float]] = {}
        self.default_proba: Dict[Tuple[int, ...], float] = {}
        self.vocab_size: int = 0
        self.start_token: int = 0

    def train(
        self, tokenized_dataset: List[List[int]], vocab_size: int, start_token: int
    ) -> None:
        """Train the model."""
        learned_paths: Dict[Tuple[int, ...], Dict[str, int]] = {}
        for sentence in tokenized_dataset:
            for i in range(0, len(sentence) - self.n):
                key = tuple(sentence[i : i + self.n])
                value = sentence[i + self.n]
                if key not in learned_paths:
                    learned_paths[key] = {}
                try:
                    learned_paths[key][value] += 1
                except KeyError:
                    learned_paths[key][value] = 1

        self.start_token = start_token
        self.vocab_size = vocab_size
        self.probas = {}
        self.default_proba = {}
        for ngram, possibilities in learned_paths.items():
            nbr_posibilities = sum(possibilities.values()) + self.vocab_size
            print(nbr_posibilities)
            self.probas[ngram] = {
                token: count / nbr_posibilities
                # No +1 above, to simplify the prediction
                for token, count in possibilities.items()
            }
            self.default_proba[ngram] = nbr_posibilities

    def predict_one(self, beginning: List[int]) -> int:
        """Predict the next value."""
        beginning = [self.start_token] * (len(beginning) - self.n) + beginning
        key = tuple(beginning)
        if key in self.probas:
            p = random()
            if p < sum(self.probas[key].values()):
                choices(self.probas[key].keys(), self.probas[key].value())
            else:
                return randint(0, self.vocab_size)
        else:
            return randint(0, self.vocab_size)

    def predict(
        self, end_token: int, beginning: List[int], max_size: int = 40
    ) -> List[int]:
        """Predict the following tokens."""
        prev_token = end_token + 1
        sentence = beginning[:]
        while prev_token != end_token and len(sentence) < max_size:
            prev_token = self.predict_one(sentence)
            sentence.append(prev_token)
        return sentence

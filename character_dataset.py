"""Character dataset."""

import re
from typing import List, Dict, Sequence
from functools import reduce
from string import digits, ascii_lowercase


def load_dataset():
    """Load the dataset."""
    with open(
        "./dataset/Vingt_mille_lieues_sous_les_mers_Texte_entier.txt", "r"
    ) as file:
        text = file.read()
    text = re.sub(r"\[\d{1,3}\]", "", text)
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"[ïî]", "", text)
    text = re.sub(r"[èèéêéêëê]", "", text)
    text = re.sub(r"[àâà]", "a", text)
    text = re.sub(r"[ûùü]", "u", text)
    text = re.sub(r"[ô]", "o", text)
    return text


class Tokenizer:
    """Tokenizer."""

    def __init__(self) -> None:
        """Init."""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[int] = []

    def train(self, dataset: List[str]) -> None:
        """Train the tokenizer."""
        tokens = reduce(lambda x, y: x | y, map(lambda x: set(x.lower()), dataset))
        tokens = tokens | set(digits + ascii_lowercase)
        self.id_to_token = list(tokens)
        self.token_to_id = {k: i for i, k in enumerate(self.id_to_token)}

    def encode_one(self, input: str) -> List[int]:
        """Encode one."""
        return list(map(lambda x: self.token_to_id[x], input.lower()))

    def encode(self, inputs: Sequence[str]) -> List[List[int]]:
        """Encode a list of texts."""
        return list(map(self.encode_one, inputs))

    def decode_one(self, input_id: List[int]) -> str:
        """Decode one."""
        return "".join(map(lambda x: self.id_to_token[x], input_id))

    def decode(self, input_ids: List[List[int]]) -> List[str]:
        """Decode a list of input_ids."""
        return list(map(self.decode_one, input_ids))


if __name__ == "__main__":
    dataset = load_dataset()
    tokenizer = Tokenizer()
    tokenizer.train(dataset)
    print(tokenizer.decode_one(tokenizer.encode_one("I love bananas")))
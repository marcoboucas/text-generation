"""Test a CNN model."""

from typing import List
from pprint import pprint

from models.tokenizer import Tokenizer


def rule(x: List[int]) -> int:
    """Rule for the next values."""
    if len(x) == 0:
        return 1
    if x[-1] > 10:
        return sum(map(int, list(str(x[-1]))))
    return x[-1] + 2


dataset = []
for i in range(40):
    dataset_element = [i % 2]
    for j in range(20):
        dataset_element.append(rule(dataset_element))
    dataset.append(" ".join(map(str, dataset_element)))

pprint(dataset)
tokenizer = Tokenizer(1)
tokenizer.train(dataset, remove_numbers=False)
tokenized_dataset = tokenizer.encode(dataset)
del dataset
pprint(tokenized_dataset)
"""Dataset to test the models."""

from typing import List
from models.tokenizer import Tokenizer
import torch


def rule(x: List[int], max_value: int = 22) -> int:
    """Rule for the next values."""
    if len(x) == 0:
        return 1
    if x[-1] < max_value:
        return x[-1] + 1
    return 0


def generate_dataset(size: int = 20, length: int = 4, max_value: int = 9):
    """Generate the dataset."""
    orig_dataset = []
    for i in range(size):
        dataset_element = [i % max_value + 1]
        for j in range(length):
            dataset_element.append(rule(dataset_element, max_value))
        orig_dataset.append(" ".join(map(str, dataset_element)))
    return orig_dataset


def generate_tokenized_dataset(
    size: int = 20,
    length: int = 4,
    max_value: int = 9,
    tokenizer: Tokenizer = Tokenizer(1),
):
    """Generate the tokenized dataset."""
    orig_dataset = generate_dataset(size, length, max_value)
    tokenizer.train(orig_dataset, remove_numbers=False)
    tokenized_dataset = tokenizer.encode(orig_dataset, add_special_tokens=False)
    return tokenized_dataset


class MyDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self, content):
        super(MyDataset, self).__init__()
        self.content = content

    def _get(
        self,
        content_idx,
    ):
        return self.content[content_idx][:-1], 4  # self.content[content_idx][-1]

    def __getitem__(self, index):
        x, y = self._get(index)
        return (
            torch.tensor(x, dtype=torch.int64),
            torch.tensor(y, dtype=torch.int64),
        )

    def __len__(self):
        return len(self.content)


if __name__ == "__main__":
    print(generate_dataset(3, 10))
    print(generate_tokenized_dataset(3, 10, 9, Tokenizer(1)))

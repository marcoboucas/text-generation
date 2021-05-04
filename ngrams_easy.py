"""Run the ngrams model."""

from typing import List
import pandas as pd


from models.ngrams.model import NgramModel
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


# dataset = load_20_mille_lieues_sous_les_mers().split("\n")

n = 5
tokenizer = Tokenizer(n)
tokenizer.train(dataset)

tokenized_dataset = tokenizer.encode(dataset)

print("Tokenizer informations")
print(tokenizer.token_to_id, tokenizer.id_to_token)
print("\n")

print("Model Training")
model = NgramModel(nbr_grams=n)
model.train(
    tokenized_dataset=tokenized_dataset,
    vocab_size=len(tokenizer.token_to_id),
    start_token=tokenizer.token_to_id[tokenizer.special_tokens["start"]],
)
print("\n")

print("Text generation")
model_prediction = model.predict(
    end_token=tokenizer.token_to_id[tokenizer.special_tokens["end"]],
    beginning=[0],
    max_size=30,
)
print(model_prediction)
print(tokenizer.decode([model_prediction])[0])
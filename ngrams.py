"""Run the ngrams model."""

import pandas as pd


from models.ngrams.model import NgramModel
from dataprocessing.load_datasets import load_20_mille_lieues_sous_les_mers
from models.ngrams.tokenizer import NgramTokenizer

dataset = load_20_mille_lieues_sous_les_mers().split("\n")

n = 5
tokenizer = NgramTokenizer(n)
tokenizer.train(dataset)

tokenized_dataset = tokenizer.encode(dataset)


model = NgramModel(nbr_grams=n)
model.train(
    tokenized_dataset=tokenized_dataset,
    vocab_size=len(tokenizer.token_to_id),
    start_token=tokenizer.token_to_id[tokenizer.special_tokens["start"]],
)

stats = pd.Series(dict(zip(model.probas.keys(), map(sum, model.probas.values()))))
print(stats.describe())
print(
    " ".join(
        tokenizer.decode(
            [
                model.predict(
                    end_token=tokenizer.token_to_id[tokenizer.special_tokens["end"]],
                    beginning=[0],
                    max_size=30,
                )
            ]
        )[0]
    )
)

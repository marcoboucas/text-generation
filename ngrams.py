from pprint import pprint

import datasets
from models.ngrams.model import NgramModel
from models.tokenizer import Tokenizer

n = 5

raw_dataset = datasets.load_dataset(
    "bookcorpus", data_dir="./dataset/bookcorpus_dataset"
)
dataset = raw_dataset["train"][:50000]["text"]
del raw_dataset
print("Dataset loaded")
tokenizer = Tokenizer(n)
tokenizer.train(dataset)
print("Tokenizer trained")

tokenized_dataset = tokenizer.encode(dataset)
del dataset
print("Tokenizer informations")
print("\n")

print("Model Training")
model = NgramModel(nbr_grams=n)
model.train(
    tokenized_dataset=tokenized_dataset,
    vocab_size=len(tokenizer.token_to_id),
    start_token=tokenizer.token_to_id[tokenizer.special_tokens["start"]],
)
print("\n")
print(tokenizer.special_tokens, tokenizer.token_to_id[tokenizer.special_tokens["end"]])
print("Text generation")
model_prediction = model.predict(
    end_token=tokenizer.token_to_id[tokenizer.special_tokens["end"]],
    beginning=tokenizer.encode(["I love"])[0][:-1],
    max_size=30,
)
print(model_prediction)
print(tokenizer.decode([model_prediction])[0])
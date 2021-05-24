"""CNN with keras."""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from test_dataset import generate_tokenized_dataset
from models.tokenizer import Tokenizer
import datasets

window_size = 30

if False:
    tokenizer = Tokenizer(1)
    dataset = generate_tokenized_dataset(40, window_size, 9, tokenizer)
    tokenized_dataset = tokenizer.encode(dataset)

    x = np.array([_[:-1] for _ in tokenized_dataset])
    y = np.array([_[-1] for _ in tokenized_dataset])
elif False:
    tokenizer = Tokenizer(1)
    raw_dataset = datasets.load_dataset(
        "bookcorpus", data_dir="./dataset/bookcorpus_dataset"
    )
    dataset = raw_dataset["train"][:50000]["text"]
    del raw_dataset
    print("Dataset loaded")
    tokenizer = Tokenizer(1)
    tokenizer.train(dataset)
    print("Tokenizer trained")

    tokenized_dataset = tokenizer.encode(dataset)
    x, y = [], []
    for dataset_element in tokenized_dataset:
        for i in range(window_size, len(dataset_element) - window_size):
            x.append(dataset_element[i : window_size + i])
            y.append(dataset_element[window_size + i])

    x = np.array(x)
    y = np.array(y)
else:
    from character_dataset import load_dataset, Tokenizer

    tokenizer = Tokenizer()
    dataset = load_dataset().lower()
    tokenizer.train([dataset])
    dataset = tokenizer.encode_one(dataset)
    x = []
    y = []
    for i in range(window_size, len(dataset) - window_size - 1):
        x.append(dataset[i : i + window_size])
        y.append(dataset[i + window_size])
    x = np.array(x)
    y = np.array(y)

model = keras.Sequential(
    [
        keras.layers.Input((window_size,)),
        keras.layers.Embedding(input_dim=len(tokenizer.id_to_token), output_dim=10),
        keras.layers.Conv1D(filters=10, kernel_size=3),
        keras.layers.Flatten(),
        keras.layers.Dense(len(tokenizer.id_to_token)),
        keras.layers.Softmax(),
    ]
)
print(x.shape, y.shape)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
)
print(model.summary())

model.fit(x, y, batch_size=1024, epochs=5)

model.save("./model_saved/cnn_keras", overwrite=True)
import pickle

with open("./tokenizer_saved.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
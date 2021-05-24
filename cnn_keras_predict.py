"""Use the trained model to predict new sentences."""
import pickle
import datasets
import numpy as np
from tensorflow import keras
from models.tokenizer import Tokenizer

model = keras.models.load_model("./model_saved/cnn_keras")
print(model.summary())
window_size = 30

with open("./tokenizer_saved.pkl", "rb") as file:
    tokenizer: Tokenizer = pickle.load(file)

sentence = "The man took a shovel and started".lower()
token_sentence = tokenizer.encode_one(sentence)
print(token_sentence)
while len(token_sentence) < 100:
    window = np.array([token_sentence[-window_size:]])
    print(window.shape)
    output = model.predict(window).argmax()
    token_sentence.append(output)
    print(tokenizer.decode_one(token_sentence))
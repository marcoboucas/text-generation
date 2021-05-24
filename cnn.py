"""Test a CNN model."""

from test_dataset import MyDataset, generate_tokenized_dataset
from typing import List
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from models.tokenizer import Tokenizer
from models.cnn import CnnNetwork
from pytorch_model_summary import summary

window_size = 1


tokenizer = Tokenizer(1)
dataset = MyDataset(
    generate_tokenized_dataset(20, window_size, max_value=3, tokenizer=tokenizer)
)
print("Dataset examples")
for i in range(3):
    print(dataset[i])
print("Tokenizer -->", tokenizer.id_to_token)


print("Model")
model = CnnNetwork(
    vocab_size=len(tokenizer.token_to_id),
    embedding_dim=4,
    padding_idx=tokenizer.token_to_id[tokenizer.special_tokens["pad"]],
    window_size=window_size,
)
print(summary(model, torch.zeros((1, window_size), dtype=torch.int32)))


trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)


# loop over the dataset multiple times
for epoch in range(40):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Loss: {}".format(running_loss))

print("Finished Training")


with torch.no_grad():
    input = "0"
    print("original input", input)
    tokenized_input = tokenizer.encode([input])[0]
    print("tokenized input", tokenized_input)
    input = torch.tensor([tokenized_input[:window_size]])
    print("Model input", input)
    output = model(input)
    print("probas: ", output)
    print("argmax", output.argmax().item())
    print(tokenizer.decode([[output.argmax()]]))
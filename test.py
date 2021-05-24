import torch
from torch import nn


criterion = nn.CrossEntropyLoss()

x = 0.01
print(
    criterion(
        torch.tensor([[x, 1 - x]], dtype=torch.float64),
        torch.tensor([0], dtype=torch.int64),
    )
)

"""CNN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnNetwork(nn.Module):
    """CNN Neural network."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, padding_idx: int, window_size: int
    ):
        """Init."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            ),
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim * window_size, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=vocab_size),
            nn.Softmax(dim=1),
        )
        """
        nn.Conv1d(
            in_channels=window_size,
            out_channels=10,
            kernel_size=3,
        ),
        nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4),
        nn.MaxPool1d(kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=50, out_features=vocab_size),
        nn.Softmax(),
        """

    def forward(self, x):
        """Compute the model output."""
        x = self.model(x)
        return x


if __name__ == "__main__":
    from random import randint
    from pytorch_model_summary import summary

    window_size = 20
    vocab_size = 100
    model = CnnNetwork(
        vocab_size=vocab_size, embedding_dim=12, padding_idx=0, window_size=window_size
    )
    input = torch.tensor(
        [[randint(1, vocab_size - 1) for i in range(window_size)]], dtype=torch.int32
    )
    print(input.shape)
    print(summary(model, input))
    model(input)
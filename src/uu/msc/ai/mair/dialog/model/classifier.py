import torch
from torch import nn
from torch.nn import functional as F


class Conv1DClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, n_classes: int, max_tokens: int=50, embed_len: int=256):
        super().__init__()
        self.max_tokens = max_tokens
        self.embed_len = embed_len

        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embed_len)
        self.conv1 = nn.Conv1d(embed_len, 32, kernel_size=7, padding="same")
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding="same")
        self.pooling = nn.MaxPool1d(2)
        self.linear = nn.Linear(32, n_classes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(batch)
        x = x.reshape(len(x), self.embed_len, self.max_tokens)
        x = F.relu(self.conv1(x))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x, _ = x.max(dim=-1)
        x = self.linear(x)
        return x

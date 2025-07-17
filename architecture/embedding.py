# Module for time embedding; adapted from Diffusor

import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    A trainable sinusoidal embedding for encoding time/position into a embed_dim vector
    """

    def __init__(self, embed_dim: int, activation: nn.Module = nn.Mish()):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.activation = activation

    def compute_sinusoidal_emb(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compute_sinusoidal_emb(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MLPCondEmbedding(nn.Module):
    """
    A trainable embedding for encoding the condition into a embed_dim vector
    """

    def __init__(
        self, embed_dim: int, cond_dim: int, activation: nn.Module = nn.Mish()
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(cond_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

import torch.nn as nn

from .embedding import MLPCondEmbedding, SinusoidalTimeEmbedding


class MLP(nn.Module):
    """
    MLP that takes (t, xt, cond) as input and outputs the vector field.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        Args:
            input_dim: The dimension of the input.
            cond_dim: The dimension of the condition.
            hidden_dim: The dimension of the hidden layers.
            num_layers: The number of layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        hidden_dim = max(hidden_dim, input_dim * 2)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.cond_emb = MLPCondEmbedding(hidden_dim, cond_dim)
        self.traj_emb = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, xt, cond=None):
        xt_shape = xt.shape
        t_emb = self.time_emb(t)
        traj_emb = self.traj_emb(xt.flatten(1))
        cond_emb = self.cond_emb(cond) if cond is not None else 0
        return self.net(t_emb + traj_emb + cond_emb).reshape(xt_shape)

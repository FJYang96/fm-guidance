import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalMLP(nn.Module):
    """
    MLP that takes (t, xt, start, goal) as input and outputs the flow.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=4, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or (input_dim - 1)  # xt is 2D, t is 1D
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, xt, cond):
        # t: (B, 1), xt: (B, 2), cond: (B, 4)
        x = torch.cat([t, xt, cond], dim=-1)
        return self.net(x)


class UNet1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConditionalUNet1D(nn.Module):
    """
    Simple 1D CNN for trajectory data, conditioned on start and goal.
    Input: (B, C, T) where C=2 (x, y), T=num_steps
    Conditioning: start, goal (B, 2) each, t (B, 1)
    Output: (B, 2, T) flow field
    """

    def __init__(self, in_ch=2, base_ch=32, num_layers=3, cond_dim=5):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for i in range(num_layers):
            self.layers.append(UNet1DBlock(ch, base_ch * (2**i)))
            ch = base_ch * (2**i)
        # Final layer to output in_ch channels
        self.final = nn.Conv1d(ch, in_ch, 1)
        # Conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, base_ch), nn.ReLU(), nn.Linear(base_ch, base_ch)
        )

    def forward(self, t, xt, start, goal):
        # t: (B, 1), xt: (B, 2, T), start: (B, 2), goal: (B, 2)
        B, C, T = xt.shape
        cond = torch.cat([t, start, goal], dim=-1)  # (B, 5)
        cond_emb = self.cond_mlp(cond)  # (B, base_ch)

        # Add conditioning to input
        cond_expanded = cond_emb.unsqueeze(-1).expand(-1, -1, T)  # (B, base_ch, T)
        # Project conditioning to match input channels
        cond_proj = F.linear(
            cond_expanded.permute(0, 2, 1),
            torch.eye(C, cond_expanded.shape[1], device=xt.device),
        ).permute(0, 2, 1)
        x = xt + cond_proj

        # Simple feedforward through layers
        for layer in self.layers:
            x = layer(x)

        # Final layer ensures output has in_ch channels and same temporal dimension
        return self.final(x)

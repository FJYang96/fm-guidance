import torch
import torch.nn as nn

from .abstract_sampler import Sampler
from .cbf_util import CBFQP


class CBFGuidedSampler(Sampler):
    def __init__(
        self,
        model: nn.Module,
        cbf_qp: CBFQP,
        per_state_barrier: bool = False,
    ):
        self.model = model
        self.conditions = None
        self.cbf_qp = cbf_qp
        self.per_state_barrier = per_state_barrier

    def solve_cbf_qp(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_nom = self.model(t, x, self.conditions)
        if self.per_state_barrier:
            assert x.ndim == 3
            u = self.cbf_qp.solve(
                x.reshape(-1, x.shape[-1]),
                u_nom.reshape(-1, x.shape[-1]),
            ).reshape(x.shape)
        else:
            u = self.cbf_qp.solve(x, u_nom)
        return u

    def forward(self, t, x):
        return self.solve_cbf_qp(x, t)

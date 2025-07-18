from typing import Callable

import cvxpy as cp
import torch
import torch.nn as nn

from .abstract_sampler import Sampler


class CBFGuidedSampler(Sampler):
    def __init__(
        self,
        model: nn.Module,
        h: Callable,
        alpha: Callable,
        P: torch.Tensor | None = None,
        per_state_barrier: bool = False,
    ):
        """
        The CBF-QP is defined as:
            min_v   1/2 (v-u_t(x))^T P (v-u_t(x))
            s.t.    \\nabla_x h(x) v + alpha(h(x)) >= 0
        v is a vector of the same dimension as the state space.

        In many of our tasks, x is a trajectory. When per_state_barrier is True,
        the CBF-QP is solved for each state separately. When per_state_barrier is False,
        the CBF-QP is solved for the entire trajectory.

        :param: model: The model that defines the vector field.
        :param: h: The barrier function (state -> real) whose 0-superlevel set is the safe set.
        :param: alpha: The class-kappa function (real -> real) for the cbf.
        :param: P: A p.d. matrix that defines the norm of the CBF-QP. None means the 2-norm.
        :param: per_state_barrier: Whether to apply the barrier function to each state separately.
        """
        self.model = model
        self.conditions = None
        self.h = h
        self.alpha = alpha
        self.P = P
        self.per_state_barrier = per_state_barrier

    def _solve_cbf_qp_closed_form(
        self, x: torch.Tensor, u_nom: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve the CBF-QP for a single state.
        :param: x: The state (batch_size, dim_x)
        :param: u_nom: The nominal vector field from the flow model (batch_size, dim_x)
        :param: t: The time (batch_size,)
        """
        Pinv = (
            torch.linalg.pinv(self.P) if self.P is not None else torch.eye(x.shape[1])
        )
        x_ = x.clone().requires_grad_(True)
        h_x = self.h(x_)
        grad_h = torch.autograd.grad(h_x.sum(), x_)[0]
        constr = (grad_h * u_nom).sum(dim=1) + self.alpha(h_x)
        mult = torch.minimum(constr, torch.zeros_like(constr)) / torch.sum(
            grad_h * (grad_h @ Pinv), dim=1
        )
        if mult.sum() > 0:
            print(h_x[:5])
            print(mult[:5])
            print("-" * 100)
        return u_nom - mult.unsqueeze(1) * grad_h @ Pinv.T

    def _solve_cbf_qp_cvxpy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Solve the CBF-QP for a single state using cvxpy.
        """
        # TODO: make sure that this guy is implemented correctly.
        raise NotImplementedError("Untested")
        """
        ts = 1.0  # Scaling factor for the time variable.
        flow_grad = self.model.forward(t / ts, x) / ts
        x_cp = x.clone().flatten(1).requires_grad_(True)
        out = -self.conditions(x_cp)
        out.backward()
        clf_grad = x_cp.grad.cpu().numpy()
        cbf_v = []
        for i in range(x.size(0)):
            expert_v = flow_grad[i].flatten().detach().cpu().numpy()
            vv = cp.Variable(expert_v.shape)
            constr = [(clf_grad[i] @ vv) >= -0.1 * out[i].item()]
            prob = cp.Problem(cp.Minimize(cp.sum(cp.square(expert_v - vv))), constr)
            prob.solve()
            cbf_v.append(torch.tensor(vv.value).reshape(*flow_grad[i].shape).float())
        cbf_v = torch.stack(cbf_v).to(x.device)
        return cbf_v
        """

    def solve_cbf_qp(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Solve the CBF-QP for the given state.
        """
        ts = 1.0  # Scaling factor for the time variable.
        u_nom = self.model(t / ts, x, self.conditions) / ts
        if self.per_state_barrier:
            assert x.ndim == 3, (
                "x must be a trajectory of shape (batch_size, num_steps, dim_x)"
            )
            u = self._solve_cbf_qp_closed_form(
                x.reshape(-1, x.shape[-1]), u_nom.reshape(-1, x.shape[-1])
            ).reshape(x.shape)
        else:
            u = self._solve_cbf_qp_closed_form(x, u_nom)
        return u

    def forward(self, t, x, ts=1):
        return self.solve_cbf_qp(x, t)

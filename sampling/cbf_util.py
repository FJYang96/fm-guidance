from typing import Callable, List

import cvxpy as cp
import torch


class CBF:
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        h: List[Callable[[torch.Tensor], torch.Tensor]],
        alpha: List[Callable[[torch.Tensor], torch.Tensor]],
        jac_h: List[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        assert len(h) == len(alpha)
        if jac_h is not None:
            assert len(jac_h) == len(h)
        self.f = f
        self.g = g
        self.h = h
        self.jac_h = jac_h
        self.alpha = alpha

    def evaluate_cbf_condition(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h_x = torch.stack([h(x) for h in self.h], dim=1)
        if self.jac_h is None:
            jac_h = torch.stack([torch.autograd.jacobian(h, x) for h in self.h], dim=1)
        else:
            jac_h = torch.stack([jac_h(x) for jac_h in self.jac_h], dim=1)

        alpha_x = torch.stack([alpha(h_x) for alpha in self.alpha], dim=1)
        # TODO: make the shapes match in the return computation
        return jac_h @ (self.f(x) + self.g(x) @ u) + alpha_x

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the CBF decrease condition for a given state and control input.
        """
        return self.evaluate_cbf_condition(x, u)


class CBFQP:
    def __init__(
        self,
        cbf: CBF,
        P: torch.Tensor | None = None,
        method: str = "closed_form",
    ):
        self.cbf = cbf
        self.P = P
        self.method = method

    def solve(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        if self.method == "closed_form":
            return self._solve_closed_form(x, u_nom)
        elif self.method == "cvxpy":
            return self._solve_cvxpy(x, u_nom)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _solve_closed_form(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        # TODO: make this work for multiple CBFs
        Pinv = (
            torch.linalg.pinv(self.P)
            if self.P is not None
            else torch.eye(x.shape[1], device=x.device)
        )
        x_ = x.clone().requires_grad_(True)
        h_x = self.h(x_)
        grad_h = torch.autograd.grad(h_x.sum(), x_)[0]
        constr = (grad_h * u_nom).sum(dim=1) + self.alpha(h_x)
        mult = torch.minimum(constr, torch.zeros_like(constr)) / torch.sum(
            grad_h * (grad_h @ Pinv), dim=1
        )
        return u_nom - mult.unsqueeze(1) * grad_h @ Pinv.T

    def _solve_cvxpy(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        # TODO: make this work for multiple CBFs
        raise NotImplementedError("CVXPY-based solver not yet implemented.")

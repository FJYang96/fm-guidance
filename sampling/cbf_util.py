from typing import Callable, List

import cvxpy as cp
import torch


class CBF:
    def __init__(
        self,
        h: List[Callable[[torch.Tensor], torch.Tensor]],
        f: Callable[[torch.Tensor], torch.Tensor],
        g: Callable[[torch.Tensor], torch.Tensor],
        alpha: List[Callable[[torch.Tensor], torch.Tensor]],
        jac_h: List[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        assert len(h) == len(alpha)
        if jac_h is not None:
            assert len(jac_h) == len(h)
        self.f = f
        self.g = g
        self.h = h
        self.jac_h_funcs = jac_h
        self.alpha = alpha

    def evaluate_cbf_condition(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h_x = torch.stack([h(x) for h in self.h], dim=0)  # (n_cbfs, batch)
        jac_h = self.jac_h(x)  # (n_cbfs, batch, dim_x)
        alpha_x = torch.stack(
            [alpha(hxx) for alpha, hxx in zip(self.alpha, h_x)], dim=0
        )  # (n_cbfs, batch)
        x_dot = self.f(x) + (self.g(x) @ u.unsqueeze(-1)).squeeze(-1)  # (batch, dim_x)
        h_dot = torch.einsum("cbx,bx->cb", jac_h, x_dot)  # (n_cbfs, batch)
        return h_dot + alpha_x

    def jac_h(self, x: torch.Tensor) -> torch.Tensor:
        if self.jac_h_funcs is None:
            x_ = x.clone().requires_grad_(True)
            h_x = [h(x_) for h in self.h]
            grad_h = torch.stack(
                [torch.autograd.grad(h_x[i].sum(), x_)[0] for i in range(len(h_x))],
                dim=0,
            )
            return grad_h
        else:
            return torch.stack([jac_h(x) for jac_h in self.jac_h_funcs], dim=0)

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
    ):
        self.cbf = cbf
        self.P = P

    def solve(
        self, x: torch.Tensor, u_nom: torch.Tensor, method: str = "cvxpy"
    ) -> torch.Tensor:
        if len(self.cbf.h) == 1:
            return self._solve_closed_form(x, u_nom)
        else:
            return self._solve_cvxpy(x, u_nom)

    def _solve_closed_form(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        assert len(self.cbf.h) == 1, "Closed-form solver only supports single CBF"
        Pinv = (
            torch.linalg.pinv(self.P)
            if self.P is not None
            else torch.eye(x.shape[1], device=x.device)
        )
        h_x = self.cbf.h[0](x)  # (batch,)
        grad_h = self.cbf.jac_h(x)[0]  # (batch, dim_x)
        lie_fh = (grad_h * self.cbf.f(x)).sum(dim=1)  # (batch,)
        lie_gh = torch.einsum("bnm,bn->bm", self.cbf.g(x), grad_h)  # (batch, dim_u)
        x_dot = lie_fh + (lie_gh * u_nom).sum(dim=1)  # (batch,)
        constr = x_dot + self.cbf.alpha[0](h_x)
        mult = torch.minimum(constr, torch.zeros_like(constr)) / torch.sum(
            lie_gh * (lie_gh @ Pinv), dim=1
        )
        return u_nom - mult.unsqueeze(1) * lie_gh @ Pinv.T

    def _solve_cvxpy(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        h_x = torch.stack([h(x) for h in self.cbf.h], dim=0)  # (n_cbfs, batch)
        jac_h = self.cbf.jac_h(x)  # (n_cbfs, batch, dim_x)
        alpha_x = torch.stack(
            [alpha(hxx) for alpha, hxx in zip(self.cbf.alpha, h_x)], dim=0
        )  # (n_cbfs, batch)

        lie_fh = (jac_h * self.cbf.f(x)).sum(dim=2)  # (n_cbfs, batch)
        lie_gh = torch.einsum(
            "bnm,cbn->cbm", self.cbf.g(x), jac_h
        )  # (n_cbfs, batch, dim_u)

        # Loop over batch as cvxpy cannot be batched.
        u_cbf = []
        for i in range(x.shape[0]):
            alpha_h = alpha_x[:, i].numpy()
            uu = cp.Variable(u_nom[i].shape)
            constraints = [
                lie_fh[:, i] + (lie_gh[:, i] @ uu) + alpha_h >= 0,
            ]
            P = (
                self.P
                if self.P is not None
                else torch.eye(u_nom[i].shape[0], device=x.device)
            )
            prob = cp.Problem(cp.Minimize(cp.quad_form(uu - u_nom[i], P)), constraints)
            prob.solve()
            u_cbf.append(torch.tensor(uu.value).float())
        return torch.stack(u_cbf, dim=0)


def single_integrator_cbf(hs, dim_x, jac_hs=None, alphas=None):
    if alphas is None:
        alphas = [lambda h: h] * len(hs)
    return CBF(
        h=hs,
        f=lambda x: torch.zeros_like(x),
        g=lambda x: torch.eye(dim_x).unsqueeze(0).repeat(x.shape[0], 1, 1),
        jac_h=jac_hs,
        alpha=alphas,
    )


def circular_barrier(
    x0: torch.Tensor, r: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: torch.linalg.norm(x - x0, axis=-1) ** 2 - r**2


def circular_barrier_jac(
    x0: torch.Tensor, r: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: 2 * (x - x0)

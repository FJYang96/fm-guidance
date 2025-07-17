from abc import ABC, abstractmethod

import torch
import torchdiffeq


class Sampler(ABC):
    @abstractmethod
    def forward(self, t, x):
        pass

    def sample(self, ic, conditions, ts=10.0, num_steps=10):
        """
        Sample from a vector field using the ODE solver.
        Args:
            vector_field: The vector field to sample from.
            ic: The initial condition.
            conditions: The conditions to sample from.
            ts: The time step.
            num_steps: The number of steps to take.
        Returns:
            The sampled trajectory.
        """
        self.conditions = conditions
        return torchdiffeq.odeint(
            self.forward,
            ic,
            torch.linspace(0, ts, num_steps, device=ic.device),
            atol=1e-4,
            rtol=1e-4,
            method="rk4",
        )


class NominalSampler(Sampler):
    def __init__(self, model):
        self.model = model
        self.conditions = None

    def forward(self, t, x, ts=1):
        # TODO: I forgot why I added the ts here...
        return self.model(t / ts, x, self.conditions) / ts


"""
def forward_w_cbf(t, x, model=model, classifier=classfier):
    flow_grad = model.forward(t / ts, x) / ts
    x_cp = x.clone().flatten(1).requires_grad_(True)
    out = -classifier(x_cp)
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


ic = torch.randn(len(gcl), 1, 28, 28, device=device)

traj_wo = torchdiffeq.odeint(
    forward_wo_cbf,
    ic,
    torch.linspace(0, ts, num_steps, device=device),
    atol=1e-4,
    rtol=1e-4,
    method="rk4",
)

traj_w = torchdiffeq.odeint(
    forward_w_cbf,
    ic,
    torch.linspace(0, ts, num_steps, device=device),
    atol=1e-4,
    rtol=1e-4,
    method="rk4",
)
"""

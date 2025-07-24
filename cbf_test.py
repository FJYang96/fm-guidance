import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

from sampling.cbf_util import (
    CBFQP,
    circular_barrier,
    circular_barrier_jac,
    single_integrator_cbf,
)


def draw_circular_obstacle(x, y, r):
    theta = torch.linspace(0, 2 * np.pi, 100)
    x_circ = x + r * torch.cos(theta)
    y_circ = y + r * torch.sin(theta)
    plt.plot(x_circ, y_circ, color="black")


def simulate_trajectories(x0, t_span, vf):
    """
    Simulate batched 2D trajectories in the vector field dx/dt = [1, 0].
    x0: (batch, 2) initial positions
    t_span: (T,) time points
    Returns: (T, batch, 2) trajectory
    """
    # odeint expects (N, D) for x0, returns (T, N, D)
    traj = odeint(vf, x0, t_span, method="euler")
    return traj  # (T, batch, 2)


def visualize_trajectories_gif(trajectory, obstacles, filename="vectorfield.gif"):
    """
    trajectory: (T, batch, 2)
    """
    images = []
    T, batch, _ = trajectory.shape
    x_min, x_max = -1, 5
    y_min, y_max = -2, 2
    for t in range(T):
        plt.clf()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal")
        plt.grid(True, alpha=0.3)
        plt.scatter(trajectory[t, :, 0], trajectory[t, :, 1], color="C1")
        for obstacle in obstacles:
            draw_circular_obstacle(*obstacle)
        # plt.legend()
        plt.title(f"t = {t}")
        plt.savefig("vf_tmp.png")
        images.append(imageio.v2.imread("vf_tmp.png"))
    os.remove("vf_tmp.png")
    plt.close()
    imageio.mimsave(filename, images, fps=10)


def constant_vector_field(t, x):
    return torch.cat(
        [torch.ones_like(x[..., :1]), torch.zeros_like(x[..., 1:2])], dim=-1
    )


obstacle_x, obstacle_y, obstacle_r = 2, 0, 1
obstacles = [(0.7, 1, 0.5), (2, -1, 0.5)]

# Construct CBF and CBF-QP
hs = [circular_barrier(torch.tensor([x, y]), r) for x, y, r in obstacles]
jac_hs = [circular_barrier_jac(torch.tensor([x, y]), r) for x, y, r in obstacles]
alphas = [lambda h: h] * len(obstacles)
cbf = single_integrator_cbf(hs, 2, jac_hs, alphas)
cbf_qp = CBFQP(cbf)


def cbf_vector_field(t, x):
    u_cbf = cbf_qp.solve(x, constant_vector_field(0, x))
    return u_cbf


batch = 20
x0 = torch.zeros(batch, 2)
x0[:, 1] = torch.linspace(-1, 1, batch)
t_span = torch.linspace(0, 4, 30)
trajectory = simulate_trajectories(x0, t_span, cbf_vector_field)  # (T, batch, 2)
trajectory_np = trajectory.detach().cpu().numpy()
visualize_trajectories_gif(trajectory_np, obstacles, filename="results/vectorfield.gif")

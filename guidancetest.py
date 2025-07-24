import matplotlib.pyplot as plt
import numpy as np
import torch

import toyenv
from architecture.traj_unet import TemporalUnet
from sampling.cbf import CBFGuidedSampler
from sampling.cbf_util import (
    CBFQP,
    circular_barrier,
    circular_barrier_jac,
    single_integrator_cbf,
)
from sampling.nominal import NominalSampler

# Gather trajectories for training
env = toyenv.Environment2D(bounds=(0.0, 10.0, 0.0, 10.0))

num_steps = 32

model = TemporalUnet(
    horizon=num_steps,
    transition_dim=2,
    cond_dim=4,
    dim=32,
    dim_mults=(1, 2, 4),
    attention=False,
)

conditioned = True
model.load_state_dict(
    torch.load("models/model_499.pth", map_location=torch.device("cpu"))
)

test_start = torch.tensor([2.0, 2.0])
test_goal = torch.tensor([8.0, 8.0])
conditions = (
    (
        torch.from_numpy(np.concatenate([test_start, test_goal], axis=-1))
        .unsqueeze(0)
        .float()
    )
    if conditioned
    else None
)

num_samples = 10
obs_center = torch.tensor([5.0, 5.0])
obs_radius = 0.5
env.add_obstacle(obs_center, obs_radius)

hs = [circular_barrier(obs_center, obs_radius)]
jac_hs = [circular_barrier_jac(obs_center, obs_radius)]
alphas = [lambda h: 30 * (torch.exp(h) - 1)] * len(hs)
cbf = single_integrator_cbf(hs, 2, jac_hs, alphas)
cbf_qp = CBFQP(cbf)

sampler = CBFGuidedSampler(model, cbf_qp, per_state_barrier=True)
# Trajectory is of shape (num_diffusion_steps, num_samples, num_steps, dim_x)
trajectory = (
    sampler.sample(
        torch.randn(num_samples, num_steps, 2), conditions, ts=1, num_steps=num_steps
    )
    .detach()
    .cpu()
    .numpy()
)
final_trajectories = trajectory[-1]


def visualize_diffusion(trajectory, start, goal):
    """
    trajectory is of shape (num_diffusion_steps, num_samples, dim_x)
    """
    import os

    import imageio

    # Make a gif of the diffusion
    images = []
    for t in range(trajectory.shape[0]):
        plt.clf()
        plt.xlim(env.x_min, env.x_max)
        plt.ylim(env.y_min, env.y_max)
        plt.gca().set_aspect("equal")
        plt.grid(True, alpha=0.3)
        plt.scatter(
            trajectory[t, :, 0],
            trajectory[t, :, 1],
            color="C1",
            label=f"t={t}",
            alpha=np.linspace(0.2, 1.0, trajectory.shape[1]),
        )
        plt.scatter(start[0], start[1], color="red", marker="x", label="start")
        plt.scatter(goal[0], goal[1], color="green", marker="x", label="goal")
        plt.gca().add_patch(
            plt.Circle(
                (obs_center[0], obs_center[1]),
                obs_radius,
                color="red",
                alpha=0.6,
            )
        )
        plt.legend()
        plt.savefig("diffusion.png")
        images.append(imageio.v2.imread("diffusion.png"))
    os.remove("diffusion.png")
    plt.close()

    # Make a gif of the diffusion
    imageio.mimsave("results/diffusion.gif", images, fps=10)


visualize_diffusion(trajectory[:, 0], test_start, test_goal)
# env.visualize_environment(test_start, test_goal, final_trajectories)

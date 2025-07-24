import matplotlib.pyplot as plt
import numpy as np
import torch

import toyenv
from architecture.traj_unet import TemporalUnet
from sampling.cbf import CBFGuidedSampler
from sampling.cbf_util import (
    CBFQP,
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

num_samples = 1
env.add_circular_obstacle(torch.tensor([5.0, 5.0]), 0.5)
env.add_circular_obstacle(torch.tensor([6.0, 4.0]), 0.5)

hs, jac_hs = env.generate_barrier()
alphas = [lambda h: 30 * (torch.exp(h) - 1)] * len(hs)
cbf = single_integrator_cbf(hs, 2, jac_hs, alphas)
cbf_qp = CBFQP(cbf)

sampler = CBFGuidedSampler(model, cbf_qp, per_state_barrier=True)
# Trajectory is of shape (num_diffusion_steps, num_samples, num_steps, dim_x)
sampling_trajectories = (
    sampler.sample(
        torch.randn(num_samples, num_steps, 2), conditions, ts=1, num_steps=num_steps
    )
    .detach()
    .cpu()
    .numpy()
)
final_trajectories = sampling_trajectories[-1]


def visualize_diffusion(trajectory, start, goal):
    """
    trajectory is of shape (num_diffusion_steps, num_samples, dim_x)
    """
    import os

    import imageio

    # Make a gif of the diffusion
    images = []
    for t in range(trajectory.shape[0]):
        fig, ax = plt.subplots(figsize=(10, 8))
        env.visualize_environment(ax, start, goal, trajectories=[trajectory[t]])
        plt.savefig("diffusion.png")
        images.append(imageio.v2.imread("diffusion.png"))
        plt.close(fig)
    os.remove("diffusion.png")

    # Make a gif of the diffusion
    imageio.mimsave("results/diffusion.gif", images, fps=10)


visualize_diffusion(sampling_trajectories[:, 0], test_start, test_goal)
# env.visualize_environment(test_start, test_goal, final_trajectories)

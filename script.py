import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import toyenv
import training as train
from architecture.mlp import MLP
from architecture.traj_unet import TemporalUnet
from guidance import NominalSampler

# Gather trajectories for training
env = toyenv.Environment2D(bounds=(0.0, 10.0, 0.0, 10.0))

# Parameters
data_num_samples = 50  # number of (start, goal) pairs
num_trajectories_per_pair = 100
num_steps = 32
batch_size = 64
n_epochs = 200
sigma = 0.0

# Generate dataset
starts = []
goals = []
trajectories = []
for _ in range(data_num_samples):
    start = env.sample_start_position()
    goal = env.sample_goal_position()
    trajs = env.generate_multiple_trajectories(
        start, goal, num_steps, num_trajectories_per_pair, noise_std=0.1
    )
    for traj in trajs:
        starts.append(start)
        goals.append(goal)
        trajectories.append(traj)

starts = np.array(starts, dtype=np.float32)  # (N, 2)
goals = np.array(goals, dtype=np.float32)  # (N, 2)
trajectories = np.array(trajectories, dtype=np.float32)  # (N, num_steps, 2)

env.visualize_environment(starts[0], goals[0], trajectories[:10])


class TrajectoryDataset(Dataset):
    def __init__(self, starts, goals, trajectories):
        self.starts = starts
        self.goals = goals
        self.trajectories = trajectories
        self.num_steps = trajectories.shape[1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.trajectories[idx]),
            torch.from_numpy(
                np.concatenate([self.starts[idx], self.goals[idx]], axis=-1)
            ),
        )


dataset = TrajectoryDataset(starts, goals, trajectories)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

models = [
    MLP(
        input_dim=num_steps * 2,
        cond_dim=4,
        embed_dim=32,
        hidden_dim=128,
        num_layers=4,
    ),
    TemporalUnet(
        horizon=num_steps,
        transition_dim=2,
        cond_dim=4,
        dim=32,
        dim_mults=(1, 2, 4),
        attention=False,
    ),
]

model = models[1]
conditioned = True

train.train_fm_model(
    model,
    train_loader,
    n_epochs,
    sigma,
    conditioned=conditioned,
    val_loader=test_loader,
)


# Visualize samples for both models
test_start = starts[0]
test_goal = goals[0]
conditions = (
    (
        torch.from_numpy(np.concatenate([test_start, test_goal], axis=-1))
        .unsqueeze(0)
        .float()
    )
    if conditioned
    else None
)

sampler = NominalSampler(model)
trajectory = sampler.sample(
    torch.randn(10, num_steps, 2), conditions, ts=1, num_steps=num_steps
)

env.visualize_environment(test_start, test_goal, trajectory[-1].detach().cpu().numpy())

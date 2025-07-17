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

# Add some obstacles for diversity
env.add_obstacle((3.0, 3.0), 1.0)
env.add_obstacle((7.0, 7.0), 0.8)
env.add_rectangular_obstacle(1.0, 2.0, 6.0, 8.0)

# Parameters
data_num_samples = 200  # number of (start, goal) pairs
num_trajectories_per_pair = 3
num_steps = 64
batch_size = 64
n_epochs = 100
sigma = 1.0

# Generate dataset
starts = []
goals = []
trajectories = []
for _ in range(data_num_samples):
    start = env.sample_start_position()
    goal = env.sample_goal_position()
    trajs = env.generate_multiple_trajectories(
        start, goal, num_steps, num_trajectories_per_pair
    )
    for traj in trajs:
        starts.append(start)
        goals.append(goal)
        trajectories.append(traj)

starts = np.array(starts, dtype=np.float32)  # (N, 2)
goals = np.array(goals, dtype=np.float32)  # (N, 2)
trajectories = np.array(trajectories, dtype=np.float32)  # (N, num_steps, 2)


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
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ),
]

x, y = dataset[0:1]
t = torch.ones(x.shape[0]) * 0.5

train.train_fm_model(models[0], train_loader, n_epochs, sigma, conditioned=True)
train.train_fm_model(models[1], train_loader, n_epochs, sigma, conditioned=True)

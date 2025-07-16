import toyenv
import training as train
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import ConditionalMLP, ConditionalUNet1D

# Gather trajectories for training
env = toyenv.Environment2D(bounds=(0.0, 10.0, 0.0, 10.0))

# Add some obstacles for diversity
env.add_obstacle((3.0, 3.0), 1.0)
env.add_obstacle((7.0, 7.0), 0.8)
env.add_rectangular_obstacle(1.0, 2.0, 6.0, 8.0)

# Parameters
data_num_samples = 200  # number of (start, goal) pairs
num_trajectories_per_pair = 3
num_steps = 25
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
    trajs = env.generate_multiple_trajectories(start, goal, num_steps, num_trajectories_per_pair)
    for traj in trajs:
        starts.append(start)
        goals.append(goal)
        trajectories.append(traj)

starts = np.array(starts, dtype=np.float32)  # (N, 2)
goals = np.array(goals, dtype=np.float32)    # (N, 2)
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
            torch.from_numpy(self.starts[idx]),
            torch.from_numpy(self.goals[idx]),
            torch.from_numpy(self.trajectories[idx])
        )

dataset = TrajectoryDataset(starts, goals, trajectories)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_trajectory_diffusion(model_type='mlp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'mlp':
        # MLP operates on flattened trajectory
        input_dim = 1 + num_steps * 2 + 2 + 2  # t, traj_flat, start, goal
        model = ConditionalMLP(input_dim=input_dim, hidden_dim=256, num_layers=4, output_dim=num_steps*2)
    else:
        # UNet operates on (B, 2, T)
        model = ConditionalUNet1D(in_ch=2, base_ch=32, num_layers=3, cond_dim=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        for batch in train_loader:
            start, goal, traj = batch  # (B, 2), (B, 2), (B, T, 2)
            B, T, _ = traj.shape
            noise = torch.randn_like(traj)
            t = torch.rand(B, 1)
            t_broadcast = t.view(B, 1, 1)
            x_t = t_broadcast * noise + (1 - t_broadcast) * traj
            start = start.to(device)
            goal = goal.to(device)
            traj = traj.to(device)
            noise = noise.to(device)
            t = t.to(device)
            x_t = x_t.to(device)
            if model_type == 'mlp':
                # Flatten trajectory for MLP
                x_t_flat = x_t.view(B, -1)
                # Call model with separate arguments as expected by the interface
                pred = model(t, x_t_flat, start, goal)
                target = (traj - noise).view(B, -1)
                loss = torch.mean((pred - target) ** 2)
            else:
                # UNet expects (B, 2, T)
                x_t_unet = x_t.permute(0, 2, 1)  # (B, 2, T)
                pred = model(t, x_t_unet, start, goal)  # (B, 2, T)
                target = (traj - noise).permute(0, 2, 1)  # (B, 2, T)
                loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[{model_type.upper()}] Epoch {epoch+1}/{n_epochs} Loss: {loss.item():.4f}")
    return model

def sample_trajectory(model, start, goal, model_type='mlp', num_steps=50, num_sampling_steps=30, device='cpu'):
    # DDIM-like deterministic sampling
    start = np.array(start, dtype=np.float32)
    goal = np.array(goal, dtype=np.float32)
    B = 1
    traj = torch.randn((B, num_steps, 2), device=device)  # start from noise
    t_steps = np.linspace(1, 0, num_sampling_steps)
    for t in t_steps:
        t_tensor = torch.full((B, 1), t, device=device)
        if model_type == 'mlp':
            x_t_flat = traj.view(B, -1)
            # Call model with separate arguments
            flow = model(t_tensor, x_t_flat, torch.from_numpy(start).to(device).unsqueeze(0), torch.from_numpy(goal).to(device).unsqueeze(0))
            flow = flow.view(B, num_steps, 2)
        else:
            x_t_unet = traj.permute(0, 2, 1)
            flow = model(t_tensor, x_t_unet, torch.from_numpy(start).to(device).unsqueeze(0), torch.from_numpy(goal).to(device).unsqueeze(0))
            flow = flow.permute(0, 2, 1)
        traj = traj + flow / num_sampling_steps  # Euler step
    return traj[0].detach().cpu().numpy()

if __name__ == "__main__":
    print("Training trajectory diffusion model (MLP)...")
    mlp_model = train_trajectory_diffusion(model_type='mlp')
    print("Training trajectory diffusion model (UNet)...")
    unet_model = train_trajectory_diffusion(model_type='unet')
    print("Training complete.")

    # Visualization of generated trajectories
    import matplotlib.pyplot as plt
    num_visualize_pairs = 2
    num_gen_traj = 5
    env_vis = toyenv.Environment2D(bounds=(0.0, 10.0, 0.0, 10.0))
    env_vis.obstacles = env.obstacles  # Copy obstacles for visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model.to(device)
    unet_model.to(device)
    mlp_model.eval()
    unet_model.eval()

    for i in range(num_visualize_pairs):
        start = env_vis.sample_start_position()
        goal = env_vis.sample_goal_position()
        mlp_trajs = [sample_trajectory(mlp_model, start, goal, model_type='mlp', num_steps=num_steps, device=device) for _ in range(num_gen_traj)]
        unet_trajs = [sample_trajectory(unet_model, start, goal, model_type='unet', num_steps=num_steps, device=device) for _ in range(num_gen_traj)]
        print(f"Visualizing MLP-generated trajectories for pair {i+1}")
        env_vis.visualize_environment(start, goal, mlp_trajs)
        print(f"Visualizing UNet-generated trajectories for pair {i+1}")
        env_vis.visualize_environment(start, goal, unet_trajs)

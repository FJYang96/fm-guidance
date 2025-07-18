import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class Environment2D:
    """
    A 2D environment for navigation from start to goal positions with trajectory generation.
    """

    def __init__(
        self, bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    ):
        """
        Initialize the 2D environment.

        Args:
            bounds: (x_min, x_max, y_min, y_max) defining the square boundaries
        """
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.obstacles = []

    def sample_start_position(self) -> Tuple[float, float]:
        """
        Sample a random start position within the environment bounds.

        Returns:
            Tuple of (x, y) coordinates
        """
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        return (x, y)

    def sample_goal_position(self) -> Tuple[float, float]:
        """
        Sample a random goal position within the environment bounds.

        Returns:
            Tuple of (x, y) coordinates
        """
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        return (x, y)

    def add_obstacle(self, center: Tuple[float, float], radius: float):
        """
        Add a circular obstacle to the environment.

        Args:
            center: (x, y) coordinates of obstacle center
            radius: radius of the circular obstacle
        """
        self.obstacles.append({"type": "circle", "center": center, "radius": radius})

    def add_rectangular_obstacle(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ):
        """
        Add a rectangular obstacle to the environment.

        Args:
            x_min, x_max, y_min, y_max: bounds of the rectangular obstacle
        """
        self.obstacles.append(
            {"type": "rectangle", "bounds": (x_min, x_max, y_min, y_max)}
        )

    def is_point_in_obstacle(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside any obstacle.

        Args:
            point: (x, y) coordinates to check

        Returns:
            True if point is inside an obstacle, False otherwise
        """
        x, y = point

        for obstacle in self.obstacles:
            if obstacle["type"] == "circle":
                center_x, center_y = obstacle["center"]
                radius = obstacle["radius"]
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance <= radius:
                    return True

            elif obstacle["type"] == "rectangle":
                x_min, x_max, y_min, y_max = obstacle["bounds"]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True

        return False

    def generate_smooth_trajectory(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        num_steps: int,
        noise_std: float = 0.1,
    ) -> np.ndarray:
        """
        Generate a smooth trajectory from start to goal using cubic interpolation.

        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            num_steps: number of time steps in the trajectory
            noise_std: standard deviation of noise to add for variety

        Returns:
            Array of shape (num_steps, 2) containing trajectory points
        """
        # Create control points for smooth interpolation
        # Add some intermediate points to create smooth curves
        start_x, start_y = start
        goal_x, goal_y = goal

        # Generate intermediate control points
        num_control_points = 5
        control_points = []

        for i in range(num_control_points):
            t = i / (num_control_points - 1)
            # Linear interpolation between start and goal
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            # Add some noise for variety
            x += np.random.normal(0, noise_std)
            y += np.random.normal(0, noise_std)

            control_points.append([x, y])

        # Create smooth interpolation
        t_control = np.linspace(0, 1, num_control_points)
        t_trajectory = np.linspace(0, 1, num_steps)

        # Interpolate x and y separately
        x_control = [p[0] for p in control_points]
        y_control = [p[1] for p in control_points]

        x_interp = interp1d(
            t_control,
            x_control,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_interp = interp1d(
            t_control,
            y_control,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        x_traj = x_interp(t_trajectory)
        y_traj = y_interp(t_trajectory)

        # Ensure start and end points are exact
        x_traj[0] = start_x
        y_traj[0] = start_y
        x_traj[-1] = goal_x
        y_traj[-1] = goal_y

        return np.column_stack([x_traj, y_traj])

    def generate_multiple_trajectories(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        num_steps: int,
        num_trajectories: int = 5,
        noise_std: float = 0.01,
    ) -> List[np.ndarray]:
        """
        Generate multiple smooth trajectories from start to goal.

        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            num_steps: number of time steps in each trajectory
            num_trajectories: number of trajectories to generate

        Returns:
            List of trajectory arrays, each of shape (num_steps, 2)
        """
        trajectories = []

        for i in range(num_trajectories):
            # Vary the noise for different trajectories
            trajectory = self.generate_smooth_trajectory(
                start, goal, num_steps, noise_std
            )
            trajectories.append(trajectory)

        return trajectories

    def visualize_environment(
        self,
        start: Optional[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        trajectories: Optional[List[np.ndarray]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Visualize the environment with obstacles, start/goal positions, and trajectories.

        Args:
            start: (x, y) start position to plot
            goal: (x, y) goal position to plot
            trajectories: list of trajectory arrays to plot
            figsize: figure size for the plot
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot environment bounds
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Plot obstacles
        for obstacle in self.obstacles:
            if obstacle["type"] == "circle":
                center_x, center_y = obstacle["center"]
                radius = obstacle["radius"]
                circle = plt.Circle(
                    (center_x, center_y), radius, color="red", alpha=0.6
                )
                ax.add_patch(circle)

            elif obstacle["type"] == "rectangle":
                x_min, x_max, y_min, y_max = obstacle["bounds"]
                rect = plt.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min, color="red", alpha=0.6
                )
                ax.add_patch(rect)

        # Plot start position
        if start is not None:
            ax.plot(start[0], start[1], "go", markersize=10, label="Start", zorder=5)

        # Plot goal position
        if goal is not None:
            ax.plot(goal[0], goal[1], "ro", markersize=10, label="Goal", zorder=5)

        # Plot trajectories
        if trajectories is not None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
            for i, trajectory in enumerate(trajectories):
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=f"Trajectory {i + 1}" if i < 5 else None,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Environment with Trajectories")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def clear_obstacles(self):
        """Clear all obstacles from the environment."""
        self.obstacles = []

    def get_environment_info(self) -> dict:
        """
        Get information about the current environment state.

        Returns:
            Dictionary containing environment bounds and obstacle information
        """
        return {
            "bounds": (self.x_min, self.x_max, self.y_min, self.y_max),
            "num_obstacles": len(self.obstacles),
            "obstacles": self.obstacles.copy(),
        }


# Example usage and testing functions
def test_environment():
    """Test the Environment2D class with various functionalities."""

    # Create environment
    env = Environment2D(bounds=(0.0, 10.0, 0.0, 10.0))

    # Add some obstacles
    env.add_obstacle((3.0, 3.0), 1.0)  # Circular obstacle
    env.add_obstacle((7.0, 7.0), 0.8)  # Another circular obstacle
    env.add_rectangular_obstacle(1.0, 2.0, 6.0, 8.0)  # Rectangular obstacle

    # Sample start and goal positions
    start = env.sample_start_position()
    goal = env.sample_goal_position()

    print(f"Start position: {start}")
    print(f"Goal position: {goal}")

    # Generate multiple trajectories
    trajectories = env.generate_multiple_trajectories(
        start, goal, num_steps=100, num_trajectories=8
    )

    # Visualize the environment
    env.visualize_environment(start, goal, trajectories)

    return env, start, goal, trajectories


def generate_training_data(
    env: Environment2D,
    num_samples: int = 100,
    num_steps: int = 50,
    num_trajectories_per_sample: int = 3,
) -> dict:
    """
    Generate training data for flow matching model.

    Args:
        env: Environment2D instance
        num_samples: number of start-goal pairs to generate
        num_steps: number of time steps per trajectory
        num_trajectories_per_sample: number of trajectories per start-goal pair

    Returns:
        Dictionary containing training data
    """
    training_data = {"starts": [], "goals": [], "trajectories": []}

    for i in range(num_samples):
        start = env.sample_start_position()
        goal = env.sample_goal_position()

        trajectories = env.generate_multiple_trajectories(
            start, goal, num_steps, num_trajectories_per_sample
        )

        training_data["starts"].append(start)
        training_data["goals"].append(goal)
        training_data["trajectories"].extend(trajectories)

    return training_data


if __name__ == "__main__":
    # Run the test
    env, start, goal, trajectories = test_environment()

    # Generate some training data
    print("\nGenerating training data...")
    training_data = generate_training_data(env, num_samples=10)
    print(f"Generated {len(training_data['starts'])} start-goal pairs")
    print(f"Generated {len(training_data['trajectories'])} trajectories")

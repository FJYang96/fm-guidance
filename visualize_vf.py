import functools

import matplotlib.pyplot as plt
import numpy as np


########################################################
# Define the Gaussian probability path
########################################################
def alpha(t):
    return t


def beta(t):
    return 1 - t


def alpha_dot(t):
    return 1


def beta_dot(t):
    return -1


########################################################
# FM related quantities
########################################################
def conditional_vf(t, x_t, z):
    z_component = (alpha_dot(t) - beta_dot(t) / beta(t) * alpha(t)) * z
    x_component = (beta_dot(t) / beta(t)) * x_t
    return x_component + z_component


def posterior_pdf_binary(t, z, x_t):
    """
    Posterior pdf of p_t(z | x_t): which z is the most likely to be the z we want to generate given x_t
    """
    pz0 = 1 / np.sqrt(2 * np.pi * beta(t)) * np.exp(-(x_t**2) / (2 * beta(t) ** 2))
    pz1 = (
        1
        / np.sqrt(2 * np.pi * beta(t))
        * np.exp(-((x_t - alpha(t)) ** 2) / (2 * beta(t) ** 2))
    )
    if z == 0:
        return pz0 / (pz0 + pz1)
    else:
        return pz1 / (pz0 + pz1)


def marginal_vf_binary(t, x_t):
    sum = 0
    for z in [0, 1]:
        sum += posterior_pdf_binary(t, z, x_t) * conditional_vf(t, x_t, z)
    return sum


########################################################
# ODE solver
########################################################
def ode_solver(x_0, dfdt, t_min=0, t_max=1, num_points=100):
    t_space = np.linspace(t_min, t_max, num_points)
    x_space = np.zeros((len(x_0), len(t_space)))
    x_space[:, 0] = x_0
    for i in range(1, num_points):
        x_space[:, i] = x_space[:, i - 1] + dfdt(t_space[i - 1], x_space[:, i - 1]) * (
            t_space[i] - t_space[i - 1]
        )
    return t_space, x_space


########################################################
# Plot the vector field
########################################################
def visualize_conditional_vector_field(
    z, x_min=-2, x_max=2, t_min=0, t_max=0.99, num_points=20
):
    """
    Visualize the conditional vector field for a given z value.
    """
    # Create meshgrid for t and x
    t_space = np.linspace(t_min, t_max, num_points)
    x_space = np.linspace(x_min, x_max, num_points)
    T, X = np.meshgrid(t_space, x_space)

    # Initialize arrays to store vector components
    U = np.zeros_like(T)  # t-component (will be 1 for all points)
    V = np.zeros_like(T)  # x-component (from conditional_vf)

    # Calculate vector field values
    for i in range(num_points):
        for j in range(num_points):
            t, x = T[i, j], X[i, j]
            vf_value = conditional_vf(t, x, z)
            U[i, j] = 1  # Movement in t direction is always 1
            V[i, j] = vf_value

    # Create the plot
    plt.quiver(T, X, U, V, scale=80, width=0.002, alpha=0.5)


def visualize_marginal_vector_field(
    x_min=-2, x_max=2, t_min=0, t_max=0.99, num_points=20
):
    """
    Visualize the marginal vector field.
    """
    t_space = np.linspace(t_min, t_max, num_points)
    x_space = np.linspace(x_min, x_max, num_points)
    T, X = np.meshgrid(t_space, x_space)

    # Initialize arrays to store vector components
    U = np.zeros_like(T)  # t-component (will be 1 for all points)
    V = np.zeros_like(T)  # x-component (from conditional_vf)

    for i in range(num_points):
        for j in range(num_points):
            t, x = T[i, j], X[i, j]
            vf_value = marginal_vf_binary(t, x)
            U[i, j] = 1  # Movement in t direction is always 1
            V[i, j] = vf_value

    # Create the plot
    plt.quiver(T, X, U, V, scale=80, width=0.002, alpha=0.5)


def visualize_trajectory(x_0, dfdt, t_min=0, t_max=1, num_points=100):
    t_space, x_space = ode_solver(x_0, dfdt, t_min, t_max, num_points)
    for i in range(x_space.shape[0]):
        plt.plot(t_space, x_space[i, :], color="grey", alpha=0.3)


def visualize_posterior_pdf(
    x_0, t_points=5, x_points=100, x_min=-2, x_max=2, t_min=0, t_max=0.99, scale=0.1
):
    """
    Visualize posterior distributions p(z|x_t) at different time points.
    Args:
        x_0: Initial samples to generate trajectories
        t_points: Number of time points to visualize
        x_points: Number of points to use for x-axis of each distribution
        x_min, x_max: Range for x-axis
        t_min, t_max: Range for time axis
    """
    t_space = np.linspace(t_min, t_max, t_points)
    x_space = np.linspace(x_min, x_max, x_points)
    T, X = np.meshgrid(t_space, x_space)

    for i in range(t_points):
        t = t_space[i]
        pz0 = posterior_pdf_binary(t, 0, x_space)
        pz1 = posterior_pdf_binary(t, 1, x_space)
        plt.plot(t - pz0 * scale, x_space, color="C0", alpha=0.5)
        plt.plot(t - pz1 * scale, x_space, color="C1", alpha=0.5)
        plt.fill_betweenx(x_space, t - pz0 * scale, t, color="C0", alpha=0.1)
        plt.fill_betweenx(x_space, t - pz1 * scale, t, color="C1", alpha=0.1)


########################################################
# Main
########################################################
x_0 = np.random.randn(100)
plot_marginal_vector_field = True

plt.figure(figsize=(10, 8))
if plot_marginal_vector_field:
    visualize_marginal_vector_field()
    # visualize_trajectory(x_0, marginal_vf_binary)
    visualize_posterior_pdf(x_0)
else:
    z = 3
    visualize_conditional_vector_field(z)
    visualize_trajectory(x_0, functools.partial(conditional_vf, z=z))
    plt.title(f"z={z}", fontsize=12)
plt.xlabel("t")
plt.ylabel("x")
plt.grid(True)
plt.show()

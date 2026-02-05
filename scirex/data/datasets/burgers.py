import numpy as np
import torch

def generate_burgers_data(num_samples=1200, nx=256, t_final=1.0, nu=0.01 / np.pi):
    """
    Generate synthetic data for the Burgers equation:
    u_t + u*u_x = nu*u_xx
    with random initial conditions
    """
    x = np.linspace(0, 2 * np.pi, nx)
    dx = x[1] - x[0]

    input_data = np.zeros((num_samples, 2, nx), dtype=np.float32)
    output_data = np.zeros((num_samples, 1, nx), dtype=np.float32)

    # Use smaller timesteps for more stable integration
    dt = 0.0005
    nt = int(t_final / dt)

    for i in range(num_samples):
        # Generate random initial conditions (smooth functions)
        n_modes = 4
        a_n = np.random.normal(0, 1, (n_modes,)) * 0.1
        b_n = np.random.normal(0, 1, (n_modes,)) * 0.1

        u0 = np.zeros_like(x)
        for n in range(n_modes):
            u0 += a_n[n] * np.sin((n + 1) * x) + b_n[n] * np.cos((n + 1) * x)
        
        # Solve Burgers equation
        u = u0.copy()
        for _ in range(nt):
            # Spatial derivatives using central differences
            u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
            u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)

            # Forward Euler step with stability check
            du = -dt * (u * u_x - nu * u_xx)
            u = u + np.clip(du, -1.0, 1.0)
        
        # Format data: Initial condition + Grid
        input_data[i, 0, :] = u0
        input_data[i, 1, :] = x
        output_data[i, 0, :] = u

    # Normalize the data
    input_data[:, 0, :] = (input_data[:, 0, :] - np.mean(input_data[:, 0, :])) / (np.std(input_data[:, 0, :]) + 1e-8)
    output_data = (output_data - np.mean(output_data)) / (np.std(output_data) + 1e-8)

    return torch.from_numpy(input_data), torch.from_numpy(output_data)

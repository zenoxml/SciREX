import numpy as np
import torch

def generate_wave_data(n_samples=1200, nx=64, nt=100, c=1.0):
    """Generate data for the 1D wave equation:
    ∂²u/∂t² = c² * ∂²u/∂x²
    """
    # Spatial domain
    L = 2 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx)

    # Time domain
    T = 2.0
    dt = T / nt

    input_data = np.zeros((n_samples, 3, nx), dtype=np.float32)
    output_data = np.zeros((n_samples, 1, nx), dtype=np.float32)

    for i in range(n_samples):
        # Generate initial displacement (sum of Gaussian pulses)
        max_pulses = 2
        u0 = np.zeros(nx)
        for _ in range(max_pulses):
            pos = np.random.uniform(0, L)
            width = np.random.uniform(0.1, 0.3)
            amp = np.random.uniform(0.2, 1.0)
            u0 += amp * np.exp(-((x - pos) ** 2) / (2 * width**2))

        # Initial velocity (smooth random function)
        v0 = np.random.normal(0, 0.1, nx)
        v0 = np.convolve(v0, np.ones(10) / 10, mode="same")

        # Solve wave equation using central differences in time and space
        u_prev = u0.copy()
        u_curr = u0 + dt * v0 # First timestep
        
        for _ in range(2, nt):
            u_next = 2 * u_curr - u_prev + (c * dt / dx) ** 2 * (
                np.roll(u_curr, 1) - 2 * u_curr + np.roll(u_curr, -1)
            )
            u_prev = u_curr
            u_curr = u_next
        
        # input: initial displacement + initial velocity + grid
        input_data[i, 0, :] = u0
        input_data[i, 1, :] = v0
        input_data[i, 2, :] = x
        output_data[i, 0, :] = u_curr

    return torch.from_numpy(input_data), torch.from_numpy(output_data), torch.from_numpy(x)

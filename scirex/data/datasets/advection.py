import numpy as np
import torch

def generate_advection_data(n_samples=1200, nx=64, nt=100, v=1.0):
    """Generate data for the 1D advection equation:
    ∂u/∂t + v * ∂u/∂x = 0
    """
    # Spatial domain: periodic boundary conditions [0, 2π]
    L = 2 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx)

    # Time domain
    T = 2.0
    dt = T / nt

    input_data = np.zeros((n_samples, 2, nx), dtype=np.float32)
    output_data = np.zeros((n_samples, 1, nx), dtype=np.float32)

    for i in range(n_samples):
        # Generate initial condition: sum of sinusoidal waves
        n_waves = 3
        amplitudes = np.random.uniform(0.1, 1.0, n_waves)
        frequencies = np.random.randint(1, 4, n_waves)

        u0 = np.zeros(nx)
        for amp, freq in zip(amplitudes, frequencies):
            u0 += amp * np.sin(freq * x)
        u0 = u0 / n_waves # Normalize

        # Solve using upwind scheme
        u = u0.copy()
        for _ in range(nt):
            if v > 0:
                u = u - v * dt / dx * (u - np.roll(u, 1))
            else:
                u = u - v * dt / dx * (np.roll(u, -1) - u)
        
        # Prepare input (init + grid) and output (final solution)
        input_data[i, 0, :] = u0
        input_data[i, 1, :] = x
        output_data[i, 0, :] = u

    return torch.from_numpy(input_data), torch.from_numpy(output_data), torch.from_numpy(x)

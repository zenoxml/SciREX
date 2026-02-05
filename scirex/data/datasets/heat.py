import numpy as np
import torch

def generate_heat_data_torch(n_samples=1200, nx=64, nt=100, D=0.1):
    """
    Generate synthetic data for the 1D Heat equation:
    ∂u/∂t = D * ∂²u/∂x²
    
    Returns:
        input_data : (N, 2, nx)  (Initial u + spatial grid x)
        output_data: (N, 1, nx)  (Final u at T)
        grid       : (nx,)       (Spatial grid x)
    """
    print(f"Generating synthetic 1D Heat Wave data (samples={n_samples}, grid={nx})...")
    
    # Spatial domain
    L = 2 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx)
    
    # Time domain
    T = 1.0
    dt = T / nt
    
    input_data = np.zeros((n_samples, 2, nx), dtype=np.float32)
    output_data = np.zeros((n_samples, 1, nx), dtype=np.float32)
    
    for i in range(n_samples):
        # Generate random initial condition (sum of Gaussian pulses)
        k1 = np.random.randint(0, 10000)
        # We use numpy random for reproducibility within the loop if we set seed globally
        
        n_pulses = 3
        u0 = np.zeros(nx)
        
        for _ in range(n_pulses):
            pos = np.random.uniform(0, L)
            width = np.random.uniform(0.1, 0.3)
            amp = np.random.uniform(0.2, 1.0)
            u0 += amp * np.exp(-((x - pos) ** 2) / (2 * width**2))
            
        # Add to input: Channel 0 is u0, Channel 1 is x (spatial grid)
        input_data[i, 0, :] = u0
        input_data[i, 1, :] = x
        
        # Solve Heat Equation using explicit Finite Difference
        # u^{n+1}_i = u^n_i + D * dt/dx^2 * (u^n_{i+1} - 2u^n_i + u^n_{i-1})
        # We assume periodic boundary conditions for simplicity in FD indices (roll)
        
        u = u0.copy()
        alpha = D * dt / (dx**2)
        
        for _ in range(nt):
            u_left = np.roll(u, 1)
            u_right = np.roll(u, -1)
            u = u + alpha * (u_left - 2*u + u_right)
            
        output_data[i, 0, :] = u

    return torch.from_numpy(input_data), torch.from_numpy(output_data), torch.from_numpy(x)

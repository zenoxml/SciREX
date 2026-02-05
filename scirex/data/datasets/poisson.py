import numpy as np
import torch

def generate_poisson_data(n_samples=600, nx=64, ny=64, include_mesh=True):
    """
    Generate synthetic data for the 2D Poisson equation:
    -∇²u = f with Dirichlet boundary conditions
    
    Returns:
        input_f  : (N, C, nx, ny)  (Source term + optional mesh)
        output_u : (N, 1, nx, ny)  (Solution)
    """
    print(f"Generating synthetic Poisson 2D data (samples={n_samples}, grid={nx}x{ny})...")
    
    # 1. Define the spatial grid [0, 1] x [0, 1]
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Pre-allocate arrays
    # f is the source term, mesh adds (x, y) coordinates to the input
    n_channels = 3 if include_mesh else 1
    input_f = np.zeros((n_samples, n_channels, nx, ny), dtype=np.float32)
    output_u = np.zeros((n_samples, 1, nx, ny), dtype=np.float32)
    
    # 2. Setup Spectral Solver
    # In Fourier space, the Laplacian operator ∇² becomes -mesh_grid_of_frequencies^2.
    # The equation -∇²u = f becomes (kx^2 + ky^2) * U_hat = F_hat.
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # The denominator for inversion is (kx^2 + ky^2)
    denominator = (KX**2 + KY**2)
    denominator[0, 0] = 1.0  # Prevent division by zero for the DC component
    
    for i in range(n_samples):
        # 3. Generate Random Source Term 'f'
        # We create a random number of Gaussian "blobs" as the source term.
        n_sources = np.random.randint(2, 6)
        f = np.zeros((nx, ny))
        for _ in range(n_sources):
            cx, cy = np.random.uniform(0.2, 0.8, 2) # Random center
            width = np.random.uniform(0.05, 0.15)    # Random width
            amp = np.random.uniform(-50, 50)        # Random amplitude
            gaussian = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * width**2))
            f += gaussian
            
        input_f[i, 0] = f
        if include_mesh:
            input_f[i, 1] = X
            input_f[i, 2] = Y
        
        # 4. Solve the equation using FFT
        # U_hat = F_hat / (kx^2 + ky^2)
        f_hat = np.fft.fftn(f)
        u_hat = f_hat / denominator
        
        # Set the DC component (mean) of u to zero to ensure a unique solution
        u_hat[0, 0] = 0.0
        
        # Inverse transform to get back to physical space
        u = np.real(np.fft.ifftn(u_hat))
        output_u[i, 0] = u

    return torch.from_numpy(input_f), torch.from_numpy(output_u)


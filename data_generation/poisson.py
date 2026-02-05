import torch
import numpy as np

def generate_poisson_data(n_samples=1200, nx=64, ny=64, device='cpu', include_mesh=True):
    """Generate data for the 2D Poisson equation:
    -∇²u = f with Dirichlet boundary conditions
    
    The equation: -∂²u/∂x² - ∂²u/∂y² = f(x,y)
    Domain: [0,1] × [0,1]
    Boundary conditions: u = 0 on the boundary
    """
    # Spatial domain [0,1] × [0,1]
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    def generate_source_term():
        """Generate random source term f(x,y) as a sum of Gaussian functions"""
        n_sources = 3
        # Random parameters for Gaussian sources
        amplitudes = (torch.rand(n_sources, device=device) * 2.0) - 1.0
        centers_x = (torch.rand(n_sources, device=device) * 0.6) + 0.2
        centers_y = (torch.rand(n_sources, device=device) * 0.6) + 0.2

        f = torch.zeros((nx, ny), device=device)
        for i in range(n_sources):
            f += amplitudes[i] * torch.exp(-50 * ((X - centers_x[i]) ** 2 + (Y - centers_y[i]) ** 2))
        return f

    def solve_poisson(f):
        """Solve Poisson equation using spectral method"""
        # Wave numbers
        kx = 2 * np.pi * torch.fft.fftfreq(nx, device=device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        # Compute solution in Fourier space
        f_hat = torch.fft.fft2(f)
        denominator = (KX**2 + KY**2)
        
        # Avoid division by zero at DC component
        denominator[0, 0] = 1.0
        u_hat = f_hat / denominator
        u_hat[0, 0] = 0.0  # Set mean to zero

        # Transform back to real space
        u = torch.real(torch.fft.ifft2(u_hat))

        # Enforce Dirichlet boundary conditions
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        return u

    # Generate dataset
    source_terms = []
    solutions = []
    
    for _ in range(n_samples):
        f = generate_source_term()
        u = solve_poisson(f)
        source_terms.append(f)
        solutions.append(u)

    source_terms = torch.stack(source_terms)
    solutions = torch.stack(solutions)

    # Prepare input-output pairs
    if include_mesh:
        # Include spatial coordinates in input
        mesh_x = X.unsqueeze(0).repeat(n_samples, 1, 1)
        mesh_y = Y.unsqueeze(0).repeat(n_samples, 1, 1)
        # Input shape: (N, 3, nx, ny)
        input_data = torch.stack([source_terms, mesh_x, mesh_y], dim=1)
    else:
        # Input shape: (N, 1, nx, ny)
        input_data = source_terms.unsqueeze(1)

    # Output shape: (N, 1, nx, ny)
    output_data = solutions.unsqueeze(1)

    return input_data, output_data


def generate_poisson_3d_data(n_samples=100, nx=32, ny=32, nz=32, n_sources=3, device='cpu', include_mesh=True):
    """Generate data for the 3D Poisson equation:
    -∇²u = f with Dirichlet boundary conditions
    """
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    z = torch.linspace(0, 1, nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    def generate_source_term():
        # n_sources is now from outer scope argument
        amplitudes = (torch.rand(n_sources, device=device) * 2.0) - 1.0
        centers_x = (torch.rand(n_sources, device=device) * 0.6) + 0.2
        centers_y = (torch.rand(n_sources, device=device) * 0.6) + 0.2
        centers_z = (torch.rand(n_sources, device=device) * 0.6) + 0.2

        f = torch.zeros((nx, ny, nz), device=device)
        for i in range(n_sources):
            f += amplitudes[i] * torch.exp(-50 * ((X - centers_x[i]) ** 2 + (Y - centers_y[i]) ** 2 + (Z - centers_z[i]) ** 2))
        return f

    def solve_poisson(f):
        kx = 2 * np.pi * torch.fft.fftfreq(nx, device=device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, device=device)
        kz = 2 * np.pi * torch.fft.fftfreq(nz, device=device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')

        f_hat = torch.fft.fftn(f)
        denominator = (KX**2 + KY**2 + KZ**2)
        denominator[0, 0, 0] = 1.0
        u_hat = f_hat / denominator
        u_hat[0, 0, 0] = 0.0

        u = torch.real(torch.fft.ifftn(u_hat))
        
        # Dirichlet BCs
        u[0, :, :] = 0; u[-1, :, :] = 0
        u[:, 0, :] = 0; u[:, -1, :] = 0
        u[:, :, 0] = 0; u[:, :, -1] = 0
        return u

    source_terms = []
    solutions = []
    for _ in range(n_samples):
        f = generate_source_term()
        u = solve_poisson(f)
        source_terms.append(f)
        solutions.append(u)

    source_terms = torch.stack(source_terms)
    solutions = torch.stack(solutions)

    if include_mesh:
        mesh_x = X.unsqueeze(0).repeat(n_samples, 1, 1, 1)
        mesh_y = Y.unsqueeze(0).repeat(n_samples, 1, 1, 1)
        mesh_z = Z.unsqueeze(0).repeat(n_samples, 1, 1, 1)
        input_data = torch.stack([source_terms, mesh_x, mesh_y, mesh_z], dim=1)
    else:
        input_data = source_terms.unsqueeze(1)

    output_data = solutions.unsqueeze(1)
    return input_data, output_data

# FastVPINNs - Training Tutorial - Helmholtz Problem

In this notebook, we will explore how to use FastVPINNs to solve the Helmholtz equation using custom loss functions and neural networks.

## hp-Variational Physics-Informed Neural Networks (hp-VPINNs)

Variational Physics-Informed Neural Networks (VPINNs) are a specialized class of Physics-Informed Neural Networks (PINNs) that are trained using the variational formulation of governing equations. The hp-VPINNs extend this concept by incorporating hp-FEM principles such as h- and p-refinement to enhance solution accuracy.

## Mathematical Formulation

The 2D Helmholtz equation is given by:

$$
-\nabla^2 u + ku = f
$$

where:
- $u$ is the solution
- $k$ is the Helmholtz coefficient
- $f$ is the source term
- $\nabla^2$ is the Laplacian operator

To obtain the weak form, we multiply by a test function $v$ and integrate over the domain $\Omega$:

$$
\int_{\Omega} (-\nabla^2 u + ku)v \, dx = \int_{\Omega} f v \, dx
$$

Applying integration by parts to the Laplacian term:

$$
\int_{\Omega} \nabla u \cdot \nabla v \, dx + \int_{\Omega} kuv \, dx - \int_{\partial \Omega} \nabla u \cdot n v \, ds = \int_{\Omega} f v \, dx
$$

With $v = 0$ on $\partial \Omega$, the boundary integral vanishes, giving us the weak form:

$$
\int_{\Omega} \nabla u \cdot \nabla v \, dx + \int_{\Omega} kuv \, dx = \int_{\Omega} f v \, dx
$$

## Problem Setup

In this example, we solve the Helmholtz equation with the following exact solution:

$$
u(x,y) = (x + y)\sin(\pi x)\sin(\pi y)
$$

### Key Parameters

#### Geometry Parameters:
- Mesh type: Quadrilateral
- Domain: [0,1] × [0,1]
- Cells: 2×2 grid
- Boundary points: 400
- Test points: 100×100 grid

#### Finite Element Parameters:
- Order: 6 (Legendre)
- Quadrature: Gauss-Jacobi (order 10)
- Transformation: Bilinear

#### Neural Network Parameters:
- Architecture: [2, 30, 30, 30, 1]
- Activation: tanh
- Learning rate: 0.002
- Boundary loss penalty (β): 10
- Training epochs: 20,000

### Boundary Conditions

The boundary conditions are implemented through four functions defining values on each boundary:

```python
def left_boundary(x, y):
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)

def right_boundary(x, y):
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)

def top_boundary(x, y):
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)

def bottom_boundary(x, y):
    return (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)
```

### Source Term

The source term f(x,y) is derived through the method of manufactured solutions:

```python
def rhs(x, y):
    term1 = 2 * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)
    term2 = 2 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    term3 = (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)
    term4 = -2 * (np.pi**2) * (x + y) * np.sin(np.pi * x) * np.sin(np.pi * y)
    return term1 + term2 + term3 + term4
```

## Implementation Steps

1. **Mesh Generation**:
   - Create a 2×2 quadrilateral mesh
   - Generate boundary points
   - Set up test points for solution evaluation

2. **Finite Element Space**:
   - Define Legendre basis functions
   - Set up quadrature rules
   - Configure boundary conditions

3. **Neural Network Model**:
   - Create a dense neural network with 3 hidden layers
   - Configure the Helmholtz loss function
   - Set up the training parameters

4. **Training Process**:
   - Train for 20,000 epochs
   - Track loss and timing metrics
   - Evaluate solution accuracy

5. **Visualization and Analysis**:
   - Plot training loss
   - Compare exact and predicted solutions
   - Generate error plots
   - Calculate error metrics (L2, L1, L∞ norms)

## Key Differences from Poisson Problem

1. **PDE Structure**:
   - Addition of the ku term
   - Modified variational form
   - Different source term computation

2. **Loss Function**:
   - Uses `pde_loss_helmholtz` instead of `pde_loss_poisson`
   - Includes Helmholtz coefficient in bilinear parameters

3. **Boundary Conditions**:
   - Non-zero Dirichlet conditions
   - More complex boundary value functions

## Results Analysis

The implementation generates four key visualizations:
1. Training loss convergence
2. Exact solution contour
3. Predicted solution contour
4. Absolute error distribution

Error metrics are computed in various norms:
- L2 norm (global accuracy)
- L1 norm (average absolute error)
- L∞ norm (maximum error)
- Relative versions of each norm

These metrics help assess the solution quality and convergence characteristics of the method.
# FastVPINNs - Training Tutorial - Inverse Parameter Identification

In this notebook, we will demonstrate how to use FastVPINNs for solving inverse problems with custom loss functions and neural networks.

Author: Thivin Anandh [LinkedIn](https://linkedin.com/in/thivinanandh)
[GitHub](https://github.com/thivinanandh)
[Portfolio](https://thivinanandh.github.io)

Paper: [FastVPINNs: Tensor-driven acceleration of VPINNs for complex geometries](https://arxiv.org/abs/2404.12063)

## hp-Variational Physics-Informed Neural Networks for Inverse Problems

Variational Physics-Informed Neural Networks (VPINNs) can be extended to solve inverse problems where we need to identify parameters in the governing equations. The hp-VPINNs framework combines the variational formulation with parameter identification using sensor data.

## Mathematical Formulation

Let's consider the 2D Poisson equation with a parameter ε (epsilon) that we want to identify:

$$
-\nabla^2 u = f(x,y,\varepsilon)
$$

where $u$ is the solution, $f$ is the source term that depends on the parameter $\varepsilon$, and $\nabla^2$ is the Laplacian operator. The variational form is:

$$
\int_{\Omega} \nabla u \cdot \nabla v \, dx - \int_{\Omega} f(\varepsilon) v \, dx = 0
$$

For this example, we consider an exact solution:

$$
u(x,y) = \sin(x)\tanh(x)\exp(-\varepsilon x^2) \cdot 10
$$

The corresponding source term $f$ is derived using the method of manufactured solutions.

## Setting up Problem Parameters

### Geometry Parameters
- i_mesh_type: Type of mesh elements (quadrilateral)
- i_mesh_generation_method: Internal mesh generation
- i_n_test_points_x, i_n_test_points_y: Number of test points (100×100)
- i_output_path: Output path for results
- i_x_min, i_x_max: Domain x-limits (-1, 1)
- i_y_min, i_y_max: Domain y-limits (-1, 1)
- i_n_cells_x, i_n_cells_y: Number of cells (2×2)
- i_n_boundary_points: Number of boundary points (400)
- i_num_sensor_points: Number of sensor points (50)

### Finite Element Parameters
- i_fe_order: Order of basis functions (10)
- i_fe_type: Type of basis functions (legendre)
- i_quad_order: Quadrature order (20)
- i_quad_type: Quadrature rule (gauss-jacobi)

### Neural Network Parameters
- Architecture: [2, 30, 30, 30, 1]
- Learning rate: 0.002
- Activation: tanh
- Boundary loss penalty (β): 10
- Training epochs: 5000

## Inverse Problem Setup

### Target Parameter
The true value of the parameter is set to:
```python
EPS = 0.3  # True parameter value
```

### Initial Guess
We start with an initial guess:
```python
eps = 2.0  # Initial parameter guess
```

### Sensor Data
We use randomly distributed sensor points in the domain to collect measurements of the solution. These measurements are used to identify the parameter:
```python
i_num_sensor_points = 50  # Number of sensor locations
```

## Model Training

The model combines three types of losses:
1. PDE loss (variational form)
2. Boundary condition loss
3. Sensor data mismatch loss

During training, we monitor:
- Total loss evolution
- Individual loss components
- Parameter convergence
- Solution accuracy

## Visualization and Results

The code generates a comprehensive visualization with six subplots:
1. Loss evolution over epochs
2. Exact solution with sensor locations
3. Predicted solution
4. Pointwise error distribution
5. Parameter convergence history
6. Sensor loss evolution

## Error Metrics

We compute and report several error metrics:
- L2 error norm
- L1 error norm
- L∞ error norm
- Relative versions of these errors


## Usage

1. Set up the problem parameters
```python
# Geometry and mesh parameters
i_mesh_type = "quadrilateral"
i_mesh_generation_method = "internal"
...

# FE parameters
i_fe_order = 10
i_fe_type = "legendre"
...

# Neural network parameters
i_learning_rate_dict = {
    "initial_learning_rate": 0.002,
    ...
}
```

2. Define the boundary conditions and exact solution

3. Create the model and train
```python
model = DenseModel_Inverse(
    layer_dims=[2, 30, 30, 30, 1],
    ...
)

for epoch in range(i_num_epochs):
    loss = model.train_step(beta=i_beta, bilinear_params_dict=bilinear_params_dict)
    ...
```

4. Analyze results and visualize
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
-\varepsilon \nabla^2 u = f(x,y)
$$

where $u$ is the solution, $f$ is the source term, and $\nabla^2$ is the Laplacian operator. The variational form is:

$$
\int_{\Omega} \varepsilon \nabla u \cdot \nabla v \, dx - \int_{\Omega} f v \, dx = 0
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


## Detailed Implementation Guide

### 1. Import Required Libraries
```python
# Common libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import time
from tqdm import tqdm

# FastvPINNs specific modules
from scirex.core.sciml.geometry.geometry_2d import Geometry_2D
from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fastvpinns.data.datahandler2d import DataHandler2D
```

### 2. Problem Configuration
Set up all necessary parameters for the geometry, finite elements, and neural network:

```python
# Geometry Configuration
i_mesh_type = "quadrilateral"
i_mesh_generation_method = "internal"
i_x_min, i_x_max = -1, 1
i_y_min, i_y_max = -1, 1
i_n_cells_x = i_n_cells_y = 2
i_n_boundary_points = 400
i_n_test_points_x = i_n_test_points_y = 100
i_num_sensor_points = 50

# Finite Element Configuration
i_fe_order = 10
i_fe_type = "legendre"
i_quad_order = 20
i_quad_type = "gauss-jacobi"

# Neural Network Configuration
i_learning_rate_dict = {
    "initial_learning_rate": 0.002,
    "use_lr_scheduler": False,
    "decay_steps": 1000,
    "decay_rate": 0.96,
    "staircase": True,
}

i_dtype = tf.float32
i_activation = "tanh"
i_beta = 10
i_num_epochs = 5000
```

### 3. Define Problem-Specific Functions

#### 3.1 Boundary Conditions and Exact Solution
```python
# Set the true parameter value
EPS = 0.3

def exact_solution(x, y):
    """Exact solution function"""
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val

# Define boundary conditions
def left_boundary(x, y):
    return exact_solution(x, y)

def right_boundary(x, y):
    return exact_solution(x, y)

def top_boundary(x, y):
    return exact_solution(x, y)

def bottom_boundary(x, y):
    return exact_solution(x, y)
```

#### 3.2 Source Term (RHS)
```python
def rhs(x, y):
    """Right-hand side function derived from the exact solution"""
    X, Y = x, y
    eps = EPS
    return (-EPS * (
        40.0 * X * eps * (np.tanh(X) ** 2 - 1) * np.sin(X)
        - 40.0 * X * eps * np.cos(X) * np.tanh(X)
        + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X)
        + 20 * (np.tanh(X) ** 2 - 1) * np.sin(X) * np.tanh(X)
        - 20 * (np.tanh(X) ** 2 - 1) * np.cos(X)
        - 10 * np.sin(X) * np.tanh(X)
    ) * np.exp(-1.0 * X**2 * eps))
```

#### 3.3 Parameter Configuration Functions
```python
def get_boundary_function_dict():
    """Map boundary IDs to boundary condition functions"""
    return {
        1000: bottom_boundary,
        1001: right_boundary,
        1002: top_boundary,
        1003: left_boundary,
    }

def get_bound_cond_dict():
    """Define boundary condition types"""
    return {1000: "dirichlet", 1001: "dirichlet", 
            1002: "dirichlet", 1003: "dirichlet"}

def get_inverse_params_dict():
    """Initial guess for the inverse parameter"""
    return {"eps": 2.0}

def get_inverse_params_actual_dict():
    """Actual parameter value for comparison"""
    return {"eps": EPS}
```

### 4. Geometry and Mesh Setup
```python
# Create geometry object
domain = Geometry_2D(
    i_mesh_type, i_mesh_generation_method,
    i_n_test_points_x, i_n_test_points_y,
    i_output_path
)

# Generate mesh
cells, boundary_points = domain.generate_quad_mesh_internal(
    x_limits=[i_x_min, i_x_max],
    y_limits=[i_y_min, i_y_max],
    n_cells_x=i_n_cells_x,
    n_cells_y=i_n_cells_y,
    num_boundary_points=i_n_boundary_points
)
```

### 5. Finite Element Space Setup
```python
fespace = Fespace2D(
    mesh=domain.mesh,
    cells=cells,
    boundary_points=boundary_points,
    cell_type=domain.mesh_type,
    fe_order=i_fe_order,
    fe_type=i_fe_type,
    quad_order=i_quad_order,
    quad_type=i_quad_type,
    fe_transformation_type="bilinear",
    bound_function_dict=bound_function_dict,
    bound_condition_dict=bound_condition_dict,
    forcing_function=rhs,
    output_path=i_output_path,
    generate_mesh_plot=True
)
```

### 6. Data Handler Setup
```python
# Initialize data handler
datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

# Get training data
train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

# Get sensor data
points, sensor_values = datahandler.get_sensor_data(
    exact_solution, 
    num_sensor_points=i_num_sensor_points, 
    mesh_type=i_mesh_generation_method
)
```

### 7. Model Creation and Training
```python
# Create inverse model
model = DenseModel_Inverse(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    params_dict={"n_cells": fespace.n_cells},
    loss_function=pde_loss_poisson_inverse,
    input_tensors_list=[
        datahandler.x_pde_list,
        train_dirichlet_input,
        train_dirichlet_output,
    ],
    orig_factor_matrices=[
        datahandler.shape_val_mat_list,
        datahandler.grad_x_mat_list,
        datahandler.grad_y_mat_list,
    ],
    force_function_list=datahandler.forcing_function_list,
    sensor_list=[points, sensor_values],
    inverse_params_dict=inverse_params_dict,
    tensor_dtype=i_dtype,
    activation=i_activation,
)

# Training loop
for epoch in tqdm(range(i_num_epochs), desc="Training Model"):
    loss = model.train_step(beta=i_beta, bilinear_params_dict=bilinear_params_dict)
    
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}")
        print(f"Variational Losses   || Pde Loss: {loss_pde:.3e} "
              f"Dirichlet Loss: {loss_dirichlet:.3e} Total Loss: {total_loss:.3e}")
        print(f"Predicted Parameter  || {loss['inverse_params']['eps'].numpy():.3e} "
              f"Actual Parameter: {actual_epsilon:.3e} "
              f"Sensor Loss: {float(loss['sensor_loss'].numpy()):.3e}")
```

### 8. Visualization and Analysis
```python
# Get predictions
test_points = domain.get_test_points()
y_exact = exact_solution(test_points[:, 0], test_points[:, 1])
y_pred = model(test_points).numpy().reshape(-1)
error = np.abs(y_exact - y_pred)

# Create visualization subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Plot loss evolution, solutions, errors, and parameter convergence
# ... (detailed plotting code as in the original implementation)

# Calculate error metrics
l2_error = np.sqrt(np.mean(error**2))
l1_error = np.mean(np.abs(error))
l_inf_error = np.max(np.abs(error))

# Create error report
error_df = pd.DataFrame({
    "L2 Error": [l2_error],
    "L1 Error": [l1_error],
    "L_inf Error": [l_inf_error],
    "Relative L2 Error": [l2_error / np.sqrt(np.mean(y_exact**2))],
    "Relative L1 Error": [l1_error / np.mean(np.abs(y_exact))],
    "Relative L_inf Error": [l_inf_error / np.max(np.abs(y_exact))],
})
print(error_df)
```

### Key Implementation Notes:

1. **Mesh Generation**: The code uses a simple 2×2 quadrilateral mesh, but this can be modified by adjusting `i_n_cells_x` and `i_n_cells_y`.

2. **Sensor Points**: 50 random sensor points are used for parameter identification. This number can be adjusted via `i_num_sensor_points`.

3. **Neural Network**: The architecture uses three hidden layers with 30 neurons each. This can be modified in the `layer_dims` parameter of `DenseModel_Inverse`.

4. **Training Process**: 
   - Uses a fixed learning rate of 0.002
   - Monitors both solution accuracy and parameter convergence
   - Prints detailed statistics every 1000 epochs

5. **Visualization**:
   - Loss evolution
   - Exact and predicted solutions
   - Error distribution
   - Parameter convergence
   - Sensor loss evolution

6. **Error Analysis**:
   - Computes both absolute and relative errors
   - Uses L1, L2, and L∞ norms
   - Provides comprehensive error statistics in tabular form
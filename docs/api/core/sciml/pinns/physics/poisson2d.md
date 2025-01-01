# Tensor-Based Loss Calculation for 2D Poisson Equation

## Overview
This module implements an efficient tensor-based approach for calculating variational residuals in 2D Poisson problems. It leverages TensorFlow's tensor operations for fast computation of weak form terms.

## Key Functions
- **`pde_loss_poisson2d`**: Computes the domain-based PDE loss.

> **Note**:  
> The implementation is based on the FastVPINNs methodology for efficient computation of variational residuals of PDEs.

---

## Function: `pde_loss_poisson2d`

### Description
Calculates residuals for the 2D Poisson problem using the Physics-Informed Neural Networks (PINNs) methodology.  
The weak form includes:
- Diffusion term: `-ε∇²(u)`
  - where `ε` is a known diffusion coefficient.

### Arguments
- **`pred_nn`** (`tf.Tensor`): Neural network solution at quadrature points.  
  **Shape**: `(N_points, 1)`
- **`pred_grad_x_nn`** (`tf.Tensor`): x-derivative of the neural network solution at quadrature points.  
  **Shape**: `(N_points, 1)`
- **`pred_grad_y_nn`** (`tf.Tensor`): y-derivative of the neural network solution at quadrature points.  
  **Shape**: `(N_points, 1)`
- **`pred_grad_xx_nn`** (`tf.Tensor`): Second-order x-derivative of the neural network solution at quadrature points.  
  **Shape**: `(N_points, 1)`
- **`pred_grad_yy_nn`** (`tf.Tensor`): Second-order y-derivative of the neural network solution at quadrature points.  
  **Shape**: `(N_points, 1)`
- **`forcing_function`** (`callable`): Right-hand side forcing term.
- **`bilinear_params`** (`dict`): A dictionary containing:
  - **`eps`**: Diffusion coefficient.

### Returns
- **`tf.Tensor`**: Cell-wise residuals averaged over test functions.  
  **Shape**: `(1,)`

### Notes
- The diffusion term is computed as `-ε(∇²u)` using second-order derivatives in the x and y directions.

# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

# Author: Thivin Anandh (https://thivinanandh.github.io)
# Version Info: 27/Dec/2024: Initial version - Thivin Anandh

"""Loss Function Implementation for 2D Poisson Inverse Problem.

This module implements the loss function for solving the inverse Poisson
equation with constant coefficient using neural networks. It focuses on
computing residuals in the weak form of the PDE for diffusion coefficient
identification.


Key functions:
    - pde_loss_poisson_inverse: Computes domain-based PDE loss for
      constant coefficient identification

Note:
    The implementation is based on the FastVPINNs methodology [1] for efficient
    computation of Variational residuals of PDEs.

References:
    [1] FastVPINNs: Tensor-Driven Acceleration of VPINNs for Complex Geometries
        DOI: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


def pde_loss_poisson_inverse(
    test_shape_val_mat: tf.Tensor,
    test_grad_x_mat: tf.Tensor,
    test_grad_y_mat: tf.Tensor,
    pred_nn: tf.Tensor,
    pred_grad_x_nn: tf.Tensor,
    pred_grad_y_nn: tf.Tensor,
    forcing_function: callable,
    bilinear_params: dict,
    inverse_params_dict: dict,
) -> tf.Tensor:
    """Calculates residual for Poisson inverse problem with constant coefficient.

    Implements the FastVPINNs methodology for computing variational residuals
    in 2D Poisson inverse problems (-∇·(ε∇u) = f) with unknown constant
    diffusion coefficient.

    Args:
        test_shape_val_mat: Test function values at quadrature points
            Shape: (n_elements, n_test_functions, n_quad_points)
        test_grad_x_mat: Test function x-derivatives at quadrature points
            Shape: (n_elements, n_test_functions, n_quad_points)
        test_grad_y_mat: Test function y-derivatives at quadrature points
            Shape: (n_elements, n_test_functions, n_quad_points)
        pred_nn: Neural network solution at quadrature points
            Shape: (n_elements, n_quad_points)
        pred_grad_x_nn: x-derivative of NN solution at quadrature points
            Shape: (n_elements, n_quad_points)
        pred_grad_y_nn: y-derivative of NN solution at quadrature points
            Shape: (n_elements, n_quad_points)
        forcing_function: Right-hand side forcing term
        bilinear_params: Additional bilinear form parameters (if any)
        inverse_params_dict: Dictionary containing:
            eps: Diffusion coefficient to be identified

    Returns:
        Cell-wise residuals averaged over test functions
            Shape: (n_cells,)

    Note:
        The weak form includes:
        - Diffusion term: ∫ε∇u·∇v dΩ
        where ε is the constant diffusion coefficient to be identified.
    """
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = inverse_params_dict["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells

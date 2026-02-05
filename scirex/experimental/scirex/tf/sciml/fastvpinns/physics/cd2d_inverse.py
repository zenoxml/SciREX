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

"""Loss Function Implementation for Convection-Diffusion 2D Inverse Problems.

This module implements the loss function for solving inverse problems in 2D
convection-diffusion equations with constant coefficients using neural networks.
It focuses on computing residuals in the weak form of the PDE for parameter
identification.

Key functions:
    - pde_loss_cd2d: Computes domain-based PDE loss for constant coefficients

Note:
    The implementation is based on the FastVPINNs methodology [1] for efficient
    computation of Variational residuals of PDEs.

References:
    [1] FastVPINNs: Tensor-Driven Acceleration of VPINNs for Complex Geometries
        DOI: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


def pde_loss_cd2d(
    test_shape_val_mat: tf.Tensor,
    test_grad_x_mat: tf.Tensor,
    test_grad_y_mat: tf.Tensor,
    pred_nn: tf.Tensor,
    pred_grad_x_nn: tf.Tensor,
    pred_grad_y_nn: tf.Tensor,
    forcing_function: callable,
    bilinear_params_dict: dict,
    inverse_param_dict: dict,
) -> tf.Tensor:
    """Calculates residual for convection-diffusion inverse problem with constant coefficients.

    Implements the FastVPINNs methodology for computing variational residuals in 2D
    convection-diffusion inverse problems. Handles constant coefficient identification
    through efficient tensor operations.

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
        bilinear_params_dict: Dictionary containing:
            b_x: x-direction convection coefficient
            b_y: y-direction convection coefficient
            c: reaction coefficient
        inverse_param_dict: Dictionary containing:
            eps: Diffusion coefficient to be identified

    Returns:
        Cell-wise residuals averaged over test functions
            Shape: (n_cells,)

    Note:
        The weak form includes:
        - Diffusion term: ∫ε∇u·∇v dΩ
        - Convection term: ∫(b·∇u)v dΩ
        - Reaction term: ∫cuv dΩ
        where ε is the constant diffusion coefficient to be identified.
    """

    # Loss Function : ∫du/dx. dv/dx  +  ∫du/dy. dv/dy - ∫f.v

    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = inverse_param_dict["eps"] * (pde_diffusion_x + pde_diffusion_y)

    # ∫du/dx. v dΩ
    conv_x = tf.transpose(tf.linalg.matvec(test_shape_val_mat, pred_grad_x_nn))

    # # ∫du/dy. v dΩ
    conv_y = tf.transpose(tf.linalg.matvec(test_shape_val_mat, pred_grad_y_nn))

    # # b(x) * ∫du/dx. v dΩ + b(y) * ∫du/dy. v dΩ
    conv = bilinear_params_dict["b_x"] * conv_x + bilinear_params_dict["b_y"] * conv_y

    # reaction term
    # ∫c.u.v dΩ
    reaction = bilinear_params_dict["c"] * tf.transpose(
        tf.linalg.matvec(test_shape_val_mat, pred_nn)
    )

    residual_matrix = (pde_diffusion + conv + reaction) - forcing_function

    # Perform Reduce mean along the axis 0
    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells

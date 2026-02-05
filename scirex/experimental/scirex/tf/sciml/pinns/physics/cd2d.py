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


"""Loss Function Implementation for 2D Convection-Diffusion Problem.

This module implements the loss function for solving 2D convection-diffusion 
equations using Physics Informed Neural Networks. It focuses on computing 
residuals of the PDE with known coefficients.


Key functions:
    - pde_loss_cd2d: Computes  PDE loss

Note:
References:

"""

import tensorflow as tf


def pde_loss_cd2d(
    pred_nn: tf.Tensor,
    pred_grad_x_nn: tf.Tensor,
    pred_grad_y_nn: tf.Tensor,
    pred_grad_xx_nn: tf.Tensor,
    pred_grad_yy_nn: tf.Tensor,
    forcing_function: callable,
    bilinear_params: dict,
) -> tf.Tensor:
    """Calculates residual for 2D convection-diffusion problem.

    Implements the PINNs methodology for computing residuals
    in 2D convection-diffusion equations with known coefficients.

    Args:
        pred_nn: Neural network solution at quadrature points
            Shape: (N_points, 1)
        pred_grad_x_nn: x-derivative of NN solution at quadrature points
            Shape: (N_points, 1)
        pred_grad_y_nn: y-derivative of NN solution at quadrature points
            Shape: (N_points, 1)
        pred_grad_xx_nn: second order x-derivative of NN solution at quadrature points
            Shape: (N_points, 1)
        pred_grad_yy_nn: second order y-derivative of NN solution at quadrature points
            Shape: (N_points, 1)
        forcing_function: Right-hand side forcing term
        bilinear_params: Dictionary containing:
            eps: Diffusion coefficient
            b_x: x-direction convection coefficient
            b_y: y-direction convection coefficient
            c: reaction coefficient

    Returns:
        Cell-wise residuals averaged over test functions
            Shape: (1,)

    Note:
        The loss includes:
        - Diffusion term: -ε∇^2(u)
        - Convection term: b·∇u
        - Reaction term: cu
        where ε, b, and c are known coefficients.
    """

    pde_diffusion_x = pred_grad_xx_nn

    pde_diffusion_y = pred_grad_yy_nn

    pde_diffusion = -1.0 * bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    conv_x = pred_grad_x_nn

    conv_y = pred_grad_y_nn

    pde_conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    pde_reaction = bilinear_params["c"] * pred_nn

    residual = (pde_diffusion + pde_conv + pde_reaction) - forcing_function

    # Perform Reduce mean along the axis 0
    pde_residual = tf.reduce_mean(tf.square(residual), axis=0)

    return pde_residual

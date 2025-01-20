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


"""Loss Function Implementation for 2D Poisson Problem.

This module implements the loss function for solving 2D Poisson 
equations using Physics Informed Neural Networks. It focuses on computing 
residuals of the PDE with known coefficients.


Key functions:
    - pde_loss_poisson2d: Computes  PDE loss

Note:
References:

"""

import tensorflow as tf


def pde_loss_maxwell(
    pred_nn_Ez: tf.Tensor,
    pred_nn_Hx: tf.Tensor,
    pred_nn_Hy: tf.Tensor,
    pred_grad_x_nn_Ez: tf.Tensor,
    pred_grad_x_nn_Hx: tf.Tensor,
    pred_grad_x_nn_Hy: tf.Tensor,
    pred_grad_y_nn_Ez: tf.Tensor,
    pred_grad_y_nn_Hx: tf.Tensor,
    pred_grad_y_nn_Hy: tf.Tensor,
    pred_grad_t_nn_Ez: tf.Tensor,
    pred_grad_t_nn_Hx: tf.Tensor,
    pred_grad_t_nn_Hy: tf.Tensor,
    forcing_function_1: callable,
    forcing_function_2: callable,
    forcing_function_3: callable,
    bilinear_params: dict,
) -> tf.Tensor:
    """Calculates residual for 2D Poisson problem.

    Implements the PINNs methodology for computing variational residuals
    in 2D Poisson equation with known coefficients.

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

    Returns:
        Cell-wise residuals averaged over test functions
            Shape: (1,)

    Note:
        The weak form includes:
        - Diffusion term: -ε∇^2(u)
        where ε is a known coefficient.
    """
    epsilon = bilinear_params["epsilon"]
    mew = bilinear_params["mu"]

    residual_1 = tf.reduce_mean(
        tf.square(epsilon * pred_grad_t_nn_Ez - (pred_grad_x_nn_Hy - pred_grad_y_nn_Hx))
    )

    residual_2 = tf.reduce_mean(tf.square(mew * pred_grad_t_nn_Hx + pred_grad_y_nn_Ez))

    residual_3 = tf.reduce_mean(tf.square(mew * pred_grad_t_nn_Hy - pred_grad_x_nn_Ez))

    return residual_1 + residual_2 + residual_3

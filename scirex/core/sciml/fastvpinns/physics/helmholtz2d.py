# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>

# Author: Thivin Anandh D
# URL: https://thivinanandh.github.io
# Loss functions for the Helmholtz2D problem


import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_helmholtz(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params,
):  # pragma: no cover
    """
    Calculates and returns the loss for the helmholtz problem

    :param test_shape_val_mat: The test shape value matrix.
    :type test_shape_val_mat: tf.Tensor
    :param test_grad_x_mat: The x-gradient of the test matrix.
    :type test_grad_x_mat: tf.Tensor
    :param test_grad_y_mat: The y-gradient of the test matrix.
    :type test_grad_y_mat: tf.Tensor
    :param pred_nn: The predicted neural network output.
    :type pred_nn: tf.Tensor
    :param pred_grad_x_nn: The x-gradient of the predicted neural network output.
    :type pred_grad_x_nn: tf.Tensor
    :param pred_grad_y_nn: The y-gradient of the predicted neural network output.
    :type pred_grad_y_nn: tf.Tensor
    :param forcing_function: The forcing function used in the PDE.
    :type forcing_function: function
    :param bilinear_params: The parameters for the bilinear form.
    :type bilinear_params: list


    :return: The calculated loss.
    :rtype: tf.Tensor
    """
    #  ∫ (du/dx. dv/dx ) dΩ
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_mat, pred_grad_x_nn))

    #  ∫ (du/dy. dv/dy ) dΩ
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_mat, pred_grad_y_nn))

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    # \int(k^2 (u).v) dw
    helmholtz_additional = (bilinear_params["k"] ** 2) * tf.transpose(
        tf.linalg.matvec(test_shape_val_mat, pred_nn)
    )

    residual_matrix = -1.0 * (pde_diffusion) + helmholtz_additional - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells

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

import tensorflow as tf


# PDE loss function for the CD2D problem Inverse (constant)
# @tf.function  #- Commented due to compatibility issues
def pde_loss_cd2d(
    test_shape_val_mat,
    test_grad_x_mat,
    test_grad_y_mat,
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    forcing_function,
    bilinear_params_dict,
    inverse_param_dict,
):  # pragma: no cover
    """
    Calculates and returns the loss for the  CD2D problem Inverse (constant)

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
    :param bilinear_params_dict: The dictionary containing the bilinear parameters.
    :type bilinear_params_dict: dict
    :param inverse_param_dict: The dictionary containing the parameters for the inverse problem.
    :type inverse_param_dict: dict

    :return: The calculated loss.
    :rtype: tf.Tensor
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

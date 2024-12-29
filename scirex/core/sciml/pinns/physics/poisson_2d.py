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


"""
This file `poisson2d.py` is implementation of our efficient tensor-based loss calculation for poisson equation

Author: Divij Ghose
URL: https://divijghose.github.io

Date: 21/Sep/2023

History: Initial implementation

Refer: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_poisson(
    pred_nn,
    pred_grad_x_nn,
    pred_grad_y_nn,
    pred_grad_xx_nn,
    pred_grad_yy_nn,
    forcing_function,
    bilinear_params,
): 
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = pred_grad_xx_nn

    # ∫du/dy. dv/dy dΩ
    pde_diffusion_y = pred_grad_yy_nn

    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = -1.0* bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells

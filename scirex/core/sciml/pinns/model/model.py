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

# Author: Divij Ghose
# URL: https://divijghose.github.io
# The file `model.py` hosts the Neural Network (NN) model and the training loop for 
# Physics-Informed Neural Networks (PINNs).
# The focus is on the model architecture and the training loop, and not on the loss functions.
"""Neural Network model for Physics-Informed Neural Networks (PINNs).

This module implements a fully connected neural network architecture along with,
its training loop for solving partial differential equations (PDEs) using the
physics-informed neural networks (PINNs) framework.


Key classes:
    - DenseModel: Base Neural Network class

Key functions:
    - train_step: The training routine for PINNs

References:
    [1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). 
    Physics-informed neural networks: A deep learning framework for solving 
    forward and inverse problems involving nonlinear partial differential equations. 
    Journal of Computational Physics, 378, 686-707.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy


# Custom Model
class DenseModel(tf.keras.Model):
    """ Defines the Dense Model for the Neural Network for solving PINNs.

    Attributes:
        :param layer_dims: List of dimensions of the dense layers.
        :type layer_dims: list
        :param learning_rate_dict: The dictionary containing the learning rate parameters.
        :type learning_rate_dict: dict
        :param params_dict: The dictionary containing the parameters.
        :type params_dict: dict
        :param loss_function: The loss function for the PDE.
        :type loss_function: function
        :param input_tensors_list: The list containing the input tensors.
        :type input_tensors_list: list
        :param force_function_list: The force function matrix.
        :type force_function_list: tf.Tensor
        :param tensor_dtype: The tensorflow dtype to be used for all the tensors.
        :type tensor_dtype: tf.DType
        :param use_attention: Flag to use attention layer after input, defaults to False.
        :type use_attention: bool, optional
        :param activation: The activation function to be used for the dense layers, defaults to "tanh".
        :type activation: str, optional
        :param hessian: Flag to use hessian loss, defaults to False.
        :type hessian: bool, optional
            
    Example:
        >>> model = DenseModel(layer_dims=[2, 20, 20, 20, 1], 
        >>>                     learning_rate_dict=learning_rate_dict,
        >>>                     params_dict=params_dict,
        >>>                     loss_function=loss_function,
        >>>                     input_tensors_list=input_tensors_list,
        >>>                     force_function_list=force_function_list,
        >>>                     tensor_dtype=tf.float32,
        >>>                     use_attention=False,
        >>>                     activation="tanh",
        >>>                     hessian=False)
    """

    def __init__(
        self,
        layer_dims,
        learning_rate_dict,
        params_dict,
        loss_function,
        input_tensors_list,
        force_function_list,
        tensor_dtype,
        use_attention=False,
        activation="tanh",
        hessian=False,
    ):
        super(DenseModel, self).__init__()
        self.layer_dims = layer_dims
        self.use_attention = use_attention
        self.activation = activation
        self.layer_list = []
        self.loss_function = loss_function
        self.hessian = hessian

        self.tensor_dtype = tensor_dtype

        # if dtype is not a valid tensorflow dtype, raise an error
        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")


        self.force_function_list = force_function_list

        self.input_tensors_list = input_tensors_list
        self.input_tensor = copy.deepcopy(input_tensors_list[0])
        self.dirichlet_input = copy.deepcopy(input_tensors_list[1])
        self.dirichlet_actual = copy.deepcopy(input_tensors_list[2])

        self.params_dict = params_dict

        self.force_matrix = self.force_function_list

        print(f"{'-'*74}")
        print(f"| {'PARAMETER':<25} | {'SHAPE':<25} |")
        print(f"{'-'*74}")
        print(
            f"| {'input_tensor':<25} | {str(self.input_tensor.shape):<25} | {self.input_tensor.dtype}"
        )
        print(
            f"| {'force_matrix':<25} | {str(self.force_matrix.shape):<25} | {self.force_matrix.dtype}"
        )
        print(
            f"| {'dirichlet_input':<25} | {str(self.dirichlet_input.shape):<25} | {self.dirichlet_input.dtype}"
        )
        print(
            f"| {'dirichlet_actual':<25} | {str(self.dirichlet_actual.shape):<25} | {self.dirichlet_actual.dtype}"
        )
        print(f"{'-'*74}")

        self.n_cells = params_dict["n_cells"]

        ## ----------------------------------------------------------------- ##
        ## ---------- LEARNING RATE AND OPTIMISER FOR THE MODEL ------------ ##
        ## ----------------------------------------------------------------- ##

        # parse the learning rate dictionary
        self.learning_rate_dict = learning_rate_dict
        initial_learning_rate = learning_rate_dict["initial_learning_rate"]
        use_lr_scheduler = learning_rate_dict["use_lr_scheduler"]
        decay_steps = learning_rate_dict["decay_steps"]
        decay_rate = learning_rate_dict["decay_rate"]
        # staircase = learning_rate_dict["staircase"]

        if use_lr_scheduler:
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=True
            )
        else:
            learning_rate_fn = initial_learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        ## ----------------------------------------------------------------- ##
        ## --------------------- MODEL ARCHITECTURE ------------------------ ##
        ## ----------------------------------------------------------------- ##

        # Build dense layers based on the input list
        for dim in range(len(self.layer_dims) - 2):
            self.layer_list.append(
                layers.Dense(
                    self.layer_dims[dim + 1],
                    activation=self.activation,
                    kernel_initializer="glorot_uniform",
                    dtype=self.tensor_dtype,
                    bias_initializer="zeros",
                )
            )

        # Add a output layer with no activation
        self.layer_list.append(
            layers.Dense(
                self.layer_dims[-1],
                activation=None,
                kernel_initializer="glorot_uniform",
                dtype=self.tensor_dtype,
                bias_initializer="zeros",
            )
        )

        # Add attention layer if required
        if self.use_attention:
            self.attention_layer = layers.Attention()

        # Compile the model
        self.compile(optimizer=self.optimizer)
        self.build(input_shape=(None, self.layer_dims[0]))

        # print the summary of the model
        self.summary()

    # def build(self, input_shape):
    #     super(DenseModel, self).build(input_shape)

    def call(self, inputs):
        """The call method for the model.

        Attributes:
        :param inputs: The input tensor for the model.
        :type inputs: tf.Tensor
        :return: The output tensor of the model.
        :rtype: tf.Tensor
        """
        x = inputs

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        return x

    def get_config(self):
        """Get the configuration of the model.

        Returns:
            dict: The configuration of the model.
        """
        # Get the base configuration
        base_config = super().get_config()

        # Add the non-serializable arguments to the configuration
        base_config.update(
            {
                "learning_rate_dict": self.learning_rate_dict,
                "loss_function": self.loss_function,
                "input_tensors_list": self.input_tensors_list,
                "force_function_list": self.force_function_list,
                "params_dict": self.params_dict,
                "use_attention": self.use_attention,
                "activation": self.activation,
                "hessian": self.hessian,
                "layer_dims": self.layer_dims,
                "tensor_dtype": self.tensor_dtype,
            }
        )

        return base_config
    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None):  
        """The trraining step for the neural network model.

        Calculates the required derivatives using autograd and computes the loss.

        Args:
            :param beta: The beta parameter for the training step, defaults to 10.
            :type beta: int, optional
            :param bilinear_params_dict: The dictionary containing the bilinear parameters, defaults to None.
            :type bilinear_params_dict: dict, optional


        Returns:
            :return: The output of the training step.
            :rtype: varies based on implementation

        Raises:

        Notes:

        References:
        """

        with tf.GradientTape(persistent=True) as tape:
            # Predict the values for dirichlet boundary conditions
            predicted_values_dirichlet = self(self.dirichlet_input)

            # initialize total loss as a tensor with shape (1,) and value 0.0
            total_pde_loss = 0.0

            with tf.GradientTape(persistent=True) as tape1:
                # tape gradient
                tape1.watch(self.input_tensor)
                # Compute the predicted values from the model
                predicted_values = self(self.input_tensor)

            # compute the gradients of the predicted values wrt the input which is (x, y)
            gradients = tape1.gradient(predicted_values, self.input_tensor)

            pred_grad_x = gradients[:, 0]  # shape : (N_points, 1)
            pred_grad_y = gradients[:, 1]  # shape : (N_points, 1)



            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(self.input_tensor)
                # Compute the second order derivatives
                second_order_gradients = tape2.gradient(gradients, self.input_tensor)

            pred_grad_xx = second_order_gradients[:, 0]  # shape : (N_points, 1)
            pred_grad_yy = second_order_gradients[:, 1]  # shape : (N_points, 1)

            pinns_residual = self.loss_function(
                pred_nn=predicted_values,
                pred_grad_x_nn=pred_grad_x,
                pred_grad_y_nn=pred_grad_y,
                pred_grad_xx_nn=pred_grad_xx,
                pred_grad_yy_nn=pred_grad_yy,
                forcing_function=self.force_matrix,
                bilinear_params=bilinear_params_dict,
            )


            # Compute the total loss for the PDE
            total_pde_loss = pinns_residual

            # print shapes of the predicted values and the actual values
            boundary_loss = tf.reduce_mean(
                tf.square(predicted_values_dirichlet - self.dirichlet_actual), axis=0
            )

            # Compute Total Loss
            total_loss = total_pde_loss + beta * boundary_loss

        trainable_vars = self.trainable_variables
        self.gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return {
            "loss_pde": total_pde_loss,
            "loss_dirichlet": boundary_loss,
            "loss": total_loss,
        }

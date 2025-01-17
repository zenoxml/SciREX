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

"""Neural Network Model Implementation for Variational Physics-Informed Neural Networks.

This module implements the neural network architecture and training loop for
solving PDEs using variational physics-informed neural networks (VPINNs).
It provides a flexible framework for handling various PDEs through custom
loss functions.

The implementation supports:
    - Flexible neural network architectures
    - Dirichlet boundary conditions
    - Custom loss function composition
    - Adaptive learning rate scheduling
    - Attention mechanisms (optional)
    - Efficient tensor operations
    - Automatic differentiation for gradients

Key classes:
    - DenseModel: Neural network model for VPINN implementation

Authors:
    - Thivin Anandh (https://thivinanandh.github.io/)

Versions:
    - 27-Dec-2024 (Version 0.1): Initial Implementation
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy

# import tensorflow wrapper
from ....dl.tensorflow_wrapper import TensorflowDense


# Custom Model
class DenseModel(tf.keras.Model):
    """Neural network model for solving PDEs using variational formulation.

    This class implements a custom neural network architecture for solving
    partial differential equations using the variational form. It supports
    flexible layer configurations and various loss components.

    Attributes:
        layer_dims: List of neurons per layer including input/output
        learning_rate_dict: Learning rate configuration containing:
            - initial_learning_rate: Starting learning rate
            - use_lr_scheduler: Whether to use learning rate decay
            - decay_steps: Steps between learning rate updates
            - decay_rate: Factor for learning rate decay
        params_dict: Model parameters including:
            - n_cells: Number of cells in the domain
        loss_function: Custom loss function for PDE residuals
        input_tensors_list: List containing:
            [0]: input_tensor - Main computation points
            [1]: dirichlet_input - Boundary points
            [2]: dirichlet_actual - Boundary values
        orig_factor_matrices: List containing:
            [0]: Shape function values
            [1]: x-derivative of shape functions
            [2]: y-derivative of shape functions
        tensor_dtype: TensorFlow data type for computations
        use_attention: Whether to use attention mechanism
        activation: Activation function for hidden layers
        optimizer: Adam optimizer with optional learning rate schedule

    Example:
        >>> model = DenseModel(
        ...     layer_dims=[2, 64, 64, 1],
        ...     learning_rate_dict={'initial_learning_rate': 0.001},
        ...     params_dict={'n_cells': 100},
        ...     loss_function=custom_loss,
        ...     tensor_dtype=tf.float32
        ... )
        >>> history = model.fit(x_train, epochs=1000)

    Note:
        The training process balances PDE residuals and boundary conditions
        through a weighted loss function. The implementation uses efficient
        tensor operations for computing variational residuals.
    """

    def __init__(
        self,
        layer_dims: list,
        learning_rate_dict: dict,
        params_dict: dict,
        loss_function,
        input_tensors_list: list,
        orig_factor_matrices: list,
        force_function_list: list,
        tensor_dtype,
        use_attention=False,
        activation="tanh",
        hessian=False,
    ):
        """
        Initialize the DenseModel class.

        Args:
            layer_dims (list): List of neurons per layer including input/output.
            learning_rate_dict (dict): Learning rate configuration containing:
                - initial_learning_rate: Starting learning rate
                - use_lr_scheduler: Whether to use learning rate decay
                - decay_steps: Steps between learning rate updates
                - decay_rate: Factor for learning rate decay
            params_dict (dict): Model parameters including:
                - n_cells: Number of cells in the domain
            loss_function: Custom loss function for PDE residuals
            input_tensors_list: List containing:
                [0]: input_tensor - Main computation points
                [1]: dirichlet_input - Boundary points
                [2]: dirichlet_actual - Boundary values
            orig_factor_matrices: List containing:
                [0]: Shape function values
                [1]: x-derivative of shape functions
                [2]: y-derivative of shape functions
            force_function_list: List containing:
                - forcing_function: Forcing function values
            tensor_dtype: TensorFlow data type for computations
            use_attention (bool): Whether to use attention mechanism, defaults to False.
            activation (str): Activation function for hidden layers, defaults to "tanh".
            hessian (bool): Whether to compute Hessian matrix, defaults to False.

        Returns:
            None
        """
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

        self.orig_factor_matrices = orig_factor_matrices
        self.shape_function_mat_list = copy.deepcopy(orig_factor_matrices[0])
        self.shape_function_grad_x_factor_mat_list = copy.deepcopy(
            orig_factor_matrices[1]
        )
        self.shape_function_grad_y_factor_mat_list = copy.deepcopy(
            orig_factor_matrices[2]
        )

        self.force_function_list = force_function_list

        self.input_tensors_list = input_tensors_list
        self.input_tensor = copy.deepcopy(input_tensors_list[0])
        self.dirichlet_input = copy.deepcopy(input_tensors_list[1])
        self.dirichlet_actual = copy.deepcopy(input_tensors_list[2])

        self.params_dict = params_dict

        self.pre_multiplier_val = self.shape_function_mat_list
        self.pre_multiplier_grad_x = self.shape_function_grad_x_factor_mat_list
        self.pre_multiplier_grad_y = self.shape_function_grad_y_factor_mat_list

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
            f"| {'pre_multiplier_grad_x':<25} | {str(self.pre_multiplier_grad_x.shape):<25} | {self.pre_multiplier_grad_x.dtype}"
        )
        print(
            f"| {'pre_multiplier_grad_y':<25} | {str(self.pre_multiplier_grad_y.shape):<25} | {self.pre_multiplier_grad_y.dtype}"
        )
        print(
            f"| {'pre_multiplier_val':<25} | {str(self.pre_multiplier_val.shape):<25} | {self.pre_multiplier_val.dtype}"
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
                TensorflowDense.create_layer(
                    units=self.layer_dims[dim + 1],
                    activation=self.activation,
                    dtype=self.tensor_dtype,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )

        # Add a output layer with no activation
        self.layer_list.append(
            TensorflowDense.create_layer(
                units=self.layer_dims[-1],
                activation=None,
                dtype=self.tensor_dtype,
                kernel_initializer="glorot_uniform",
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

    def call(self, inputs) -> tf.Tensor:
        """
        The call method for the model.

        Args:
            inputs: The input tensor for the model.

        Returns:
            tf.Tensor: The output tensor from the model.
        """
        x = inputs

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        return x

    def get_config(self) -> dict:
        """
        Get the configuration of the model.

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
                "orig_factor_matrices": self.orig_factor_matrices,
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
    def train_step(
        self, beta=10, bilinear_params_dict=None
    ) -> dict:  # pragma: no cover
        """
        The train step method for the model.

        Args:
            beta (int): The weight for the boundary loss, defaults to 10.
            bilinear_params_dict (dict): The bilinear parameters dictionary, defaults to None.

        Returns:
            dict: The loss values for the model.
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

            # Split the gradients into x and y components and reshape them to (-1, 1)
            # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_grad_x = tf.reshape(
                gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_grad_y = tf.reshape(
                gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            pred_val = tf.reshape(
                predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            cells_residual = self.loss_function(
                test_shape_val_mat=self.pre_multiplier_val,
                test_grad_x_mat=self.pre_multiplier_grad_x,
                test_grad_y_mat=self.pre_multiplier_grad_y,
                pred_nn=pred_val,
                pred_grad_x_nn=pred_grad_x,
                pred_grad_y_nn=pred_grad_y,
                forcing_function=self.force_matrix,
                bilinear_params=bilinear_params_dict,
            )

            residual = tf.reduce_sum(cells_residual)

            # Compute the total loss for the PDE
            total_pde_loss = total_pde_loss + residual

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

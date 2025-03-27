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
    - Divij Ghose (https://divijghose.github.io/)

Versions:
    - 27-Dec-2024 (Version 0.1): Initial Implementation
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
import copy

# import tensorflow wrapper
from ....dl.tensorflow_wrapper import TensorflowDense


class CustomActivation(layers.Layer):
    def __init__(self, a, dtype=tf.float64, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.a = tf.Variable(a, dtype=dtype)  # Use the provided value of a
        self.constant = tf.constant(10.0, dtype=dtype)
        self.dtype_val = dtype
        # self.b = b

    @tf.function
    def call(self, inputs):
        input_tensor = tf.cast(inputs, tf.float64)
        val1 = tf.tanh(self.constant * self.a * input_tensor)

        return val1


class PolynomialActivation(tf.keras.layers.Layer):
    """
    Polynomial activation function for neural networks implemented with TensorFlow,
    based on the ICLR 2020 paper.

    PolynomialActivation(x) = fn(g(x))/sqrt(2^n)

    Where:
    - fn(x) is an n-th order polynomial with trainable weights
    - g(x) is the dynamic input scaling function
    - weights are normalized when their L2 norm exceeds 3
    """

    def __init__(self, coefficients=None, degree=1, dtype=tf.float32, name=None):
        """
        Initialize the polynomial activation function.

        Args:
            coefficients: Optional list of initial coefficients [w0, w1, w2, ..., wn]
                         If provided, degree will be set to len(coefficients)-1
            degree: The degree of the polynomial (n), used only if coefficients is None
            dtype: Data type for the coefficients
            name: Optional name for the activation function
        """
        super(PolynomialActivation, self).__init__(name=name)

        # Set degree and initial coefficients
        if coefficients is not None:
            self._initial_coefficients = list(
                coefficients
            )  # Convert to list to be safe
            self._degree = len(self._initial_coefficients) - 1
        else:
            self._initial_coefficients = None
            self._degree = degree

        # Set data type
        self._dtype_value = dtype

    def build(self, input_shape):
        """
        Build the layer, creating the trainable weights.
        """
        # Create trainable variables for the coefficients [w_0, w_1, ..., w_n]
        if self._initial_coefficients is not None:
            # Use provided coefficients as initializer
            initializer = tf.constant_initializer(self._initial_coefficients)
        else:
            # Default to Glorot uniform initializer
            initializer = tf.keras.initializers.GlorotUniform()

        self.coefficients = self.add_weight(
            name="coefficients",  # Use a simple name, TensorFlow will prefix with layer name
            shape=(self._degree + 1,),
            initializer=initializer,
            trainable=True,
            dtype=self._dtype_value,
        )

        # Store constraint norm value (3 as mentioned in the paper)
        self.max_norm = tf.constant(3.0, dtype=self._dtype_value)

        self.built = True

    def dynamic_input_scaling(self, inputs):
        """
        Implement the g(x) dynamic input scaling function:

        g(x_i) = sqrt(2) * x_i / max_1≤j≤k|x_j|

        As defined in the paper to constrain max(g(x_i)) = sqrt(2)
        """
        # Calculate the maximum absolute value in each sample
        # Keep dims to ensure proper broadcasting
        abs_inputs = tf.abs(inputs)
        max_abs = tf.reduce_max(abs_inputs, axis=-1, keepdims=True)

        # Avoid division by zero
        max_abs = tf.maximum(max_abs, tf.keras.backend.epsilon())

        # Apply the scaling: sqrt(2) * x_i / max|x_j|
        sqrt_2 = tf.sqrt(tf.constant(2.0, dtype=inputs.dtype))
        scaled_inputs = sqrt_2 * inputs / max_abs

        return scaled_inputs

    def normalize_weights(self):
        """
        Normalize weights when their L2 norm exceeds 3,
        as specified in the paper: w_j * 3/||w_j||_2
        """
        # Calculate L2 norm of weights
        weights_norm = tf.norm(self.coefficients, ord=2)
        # Add epsilon to avoid division by zero
        weights_norm = tf.maximum(weights_norm, tf.keras.backend.epsilon())

        # Create a condition to apply normalization only when norm > 3
        condition = tf.greater(weights_norm, self.max_norm)

        # When condition is true, normalize weights to have norm = 3
        normalized_weights = self.coefficients * (self.max_norm / weights_norm)

        # Apply normalization only when condition is true
        self.coefficients.assign(
            tf.cond(condition, lambda: normalized_weights, lambda: self.coefficients)
        )

    def compute_polynomial(self, x):
        """
        Compute the polynomial: w_0 + w_1*x + w_2*x^2 + ... + w_n*x^n
        """
        # Get coefficients with the right dtype
        w = tf.cast(self.coefficients, x.dtype)

        # Initialize with the constant term (w[0])
        result = tf.ones_like(x) * w[0]

        # Compute higher-order terms
        x_power = x
        for i in range(1, self._degree + 1):
            result = result + w[i] * x_power
            # Prepare for the next power
            if i < self._degree:
                x_power = x_power * x

        return result

    def call(self, inputs):
        """
        Apply the polynomial activation function with dynamic scaling and normalization.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor after applying polynomial activation
        """
        # First normalize weights if needed
        self.normalize_weights()

        # Apply dynamic input scaling g(x)
        scaled_inputs = self.dynamic_input_scaling(inputs)

        # Compute polynomial function f_n(g(x))
        poly_output = self.compute_polynomial(scaled_inputs)

        # Scale the output by 1/sqrt(2^n) as defined in the paper
        scaling_factor = tf.sqrt(
            tf.pow(
                tf.constant(2.0, dtype=inputs.dtype),
                tf.cast(self._degree, inputs.dtype),
            )
        )
        final_output = poly_output / scaling_factor

        return final_output

    def get_config(self):
        """
        Return configuration for serialization.
        """
        config = super(PolynomialActivation, self).get_config()
        config.update(
            {
                "degree": self._degree,
                "dtype": self._dtype_value,
                # Get current coefficients if available, otherwise use initial ones
                "coefficients": (
                    self.coefficients.numpy().tolist()
                    if hasattr(self, "coefficients")
                    else self._initial_coefficients
                ),
            }
        )
        return config

    def get_coefficients(self):
        """
        Get the current values of the coefficients.

        Returns:
            List of current coefficient values
        """
        if hasattr(self, "coefficients"):
            return self.coefficients.numpy().tolist()
        return self._initial_coefficients


class MagnetisationModel(tf.keras.Model):
    """Neural Network for interpolating magnetisation data."""

    def __init__(self, b_tensor, h_tensor, dtype=tf.float32):
        super(MagnetisationModel, self).__init__()
        self.layer_dims = [1, 64, 64, 1]
        self.layer_list = []
        self.activation = "relu"
        self.b_tensor = b_tensor
        self.h_tensor = h_tensor
        self.tensor_dtype = dtype

        self.mean_b = tf.reduce_mean(self.b_tensor)
        self.std_b = tf.math.reduce_std(self.b_tensor)
        self.mean_h = tf.reduce_mean(self.h_tensor)
        self.std_h = tf.math.reduce_std(self.h_tensor)
        self.b = (self.b_tensor - self.mean_b) / self.std_b
        self.h = (self.h_tensor - self.mean_h) / self.std_h

        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        self.learning_rate = 0.002
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

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

        self.layer_list.append(
            TensorflowDense.create_layer(
                units=self.layer_dims[-1],
                activation=None,
                dtype=self.tensor_dtype,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
            )
        )

        self.compile(optimizer=self.optimizer)
        self.build(input_shape=(None, self.layer_dims[0]))
        self.summary()

    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

    @tf.function
    def train_step(self):
        # mean squared error loss
        with tf.GradientTape() as tape:
            predicted_h = self(self.b)
            loss = tf.reduce_mean(tf.square(predicted_h - self.h))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}


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
        use_adaptive=False,
        use_polynomial=False,
        polynomial_coeffs=[0.0, 1.0],
        hessian=False,
        trained_magnetisation_model=None,
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
        self.use_adaptive = use_adaptive
        self.use_polynomial = use_polynomial
        self.polynomial_coeffs = polynomial_coeffs
        self.degree = len(self.polynomial_coeffs) - 1
        self.layer_list = []
        self.polynomial_activations = []  # Track polynomial activation layers
        self.loss_function = loss_function
        self.hessian = hessian
        self.a = 0.1

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

        self.trained_magnetisation_model = trained_magnetisation_model

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

        if self.use_adaptive:

            adaptive_activation = CustomActivation(self.a, self.tensor_dtype)

            # Build dense layers based on the i/p list
            for dim in range(len(self.layer_dims) - 2):
                self.layer_list.append(
                    layers.Dense(
                        self.layer_dims[dim + 1],
                        activation=None,
                        kernel_initializer="glorot_uniform",
                        dtype=self.tensor_dtype,
                        bias_initializer="zeros",
                    )
                )
                self.layer_list.append(adaptive_activation)

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

        elif self.use_polynomial:

            # Build dense layers based on the i/p list
            for dim in range(len(self.layer_dims) - 2):
                self.layer_list.append(
                    layers.Dense(
                        self.layer_dims[dim + 1],
                        activation=None,
                        kernel_initializer="glorot_uniform",
                        dtype=self.tensor_dtype,
                        bias_initializer="zeros",
                    )
                )
                # Create a NEW instance of PolynomialActivation for each layer with a unique name to avoid variable conflicts
                poly_name = f"poly_act_{dim}"
                poly_act = PolynomialActivation(
                    coefficients=self.polynomial_coeffs,
                    degree=self.degree,
                    dtype=self.tensor_dtype,
                    name=poly_name,
                )
                self.layer_list.append(poly_act)
                self.polynomial_activations.append(
                    poly_act
                )  # Keep track of polynomial activations

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

        else:

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

        x = x

        # Apply attention layer after input if flag is True
        if self.use_attention:
            x = self.attention_layer([x, x])

        # Loop through the dense layers
        for layer in self.layer_list:
            x = layer(x)

        x = (
            tf.cast(
                (tf.sqrt(tf.square(inputs[:, 0:1]) + tf.square(inputs[:, 1:2])) - 1),
                dtype=self.tensor_dtype,
            )
            * x
        )
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

    @property
    def trainable_variables(self):
        """Get all trainable variables including polynomial activation coefficients."""
        # Get all standard Keras trainable variables
        keras_vars = super().trainable_variables

        # Add polynomial activation coefficients
        poly_vars = []
        for layer in self.layer_list:
            if (
                hasattr(layer, "coefficients")
                and hasattr(layer.coefficients, "trainable")
                and layer.coefficients.trainable
            ):
                poly_vars.append(layer.coefficients)

        # Return combined list
        return keras_vars + poly_vars

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
                predicted_Az = self(self.input_tensor)

            # compute the gradients of the predicted values wrt the input which is (x, y)
            gradients = tape1.gradient(predicted_Az, self.input_tensor)

            # Split the gradients into x and y components and reshape them to (-1, 1)
            # the reshaping is done for the tensorial operations purposes (refer Notebook)
            pred_Az_grad_x = tf.reshape(
                gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)
            pred_Az_grad_y = tf.reshape(
                gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            pred_Az_val = tf.reshape(
                predicted_Az, [self.n_cells, self.pre_multiplier_val.shape[-1]]
            )  # shape : (N_cells , N_quadrature_points)

            predicted_Bx = gradients[:, 0]
            predicted_By = gradients[:, 1]

            calculated_B = tf.sqrt(tf.square(predicted_Bx) + tf.square(predicted_By))
            calculated_B = tf.reshape(calculated_B, [-1, 1])
            normalized_B = (
                calculated_B - self.trained_magnetisation_model.mean_b
            ) / self.trained_magnetisation_model.std_b
            predicted_H = self.trained_magnetisation_model(normalized_B)
            calculated_H = (
                predicted_H * self.trained_magnetisation_model.std_h
                + self.trained_magnetisation_model.mean_h
            )
            calculated_permeability = calculated_B / calculated_H
            calculated_permeability = tf.reshape(
                calculated_permeability,
                [self.n_cells, self.pre_multiplier_val.shape[-1]],
            )

            cells_residual = self.loss_function(
                test_shape_val_mat=self.pre_multiplier_val,
                test_grad_x_mat=self.pre_multiplier_grad_x,
                test_grad_y_mat=self.pre_multiplier_grad_y,
                pred_nn=pred_Az_val,
                pred_grad_x_nn=pred_Az_grad_x,
                pred_grad_y_nn=pred_Az_grad_y,
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
            "loss_dirichlet": beta * boundary_loss,
            "loss": total_loss,
        }

    # method to get polynomial coefficients
    def get_polynomial_coefficients(self):
        """
        Returns the current values of all polynomial activation coefficients in the model.
        Call this method after training to check if coefficients have been updated.
        """
        coeffs = []
        for i, layer in enumerate(self.layer_list):
            if hasattr(layer, "get_coefficients"):
                coeffs.append({f"layer_{i}": layer.get_coefficients()})
        return coeffs

    def inference(self, test_tensor):
        """
        The inference method for the model.

        Returns:
            dict: The predicted values from the model.
        """
        test_tensor = tf.convert_to_tensor(test_tensor, dtype=self.tensor_dtype)
        test_tensor_x, test_tensor_y = test_tensor[:, 0], test_tensor[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            # tape gradient
            tape.watch(test_tensor_x)
            tape.watch(test_tensor_y)
            # Compute the predicted values from the model

            coords = tf.stack([test_tensor_x, test_tensor_y], axis=-1)

            predicted_Az = self(coords)

        # gradients = tape.gradient(predicted_Az, test_tensor)

        Bx = tape.gradient(predicted_Az, test_tensor_y)
        By = -1.0 * tape.gradient(predicted_Az, test_tensor_x)

        B = tf.sqrt(tf.square(Bx) + tf.square(By))

        del tape

        return {
            "Az": predicted_Az,
            "Bx": Bx,
            "By": By,
            "B": B,
        }

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

"""Two-Dimensional Data Handler Implementation for PDE Solvers.

This module implements data handling functionality for 2D PDE problems,
focusing on efficient tensor conversions and management of finite element
data structures. It provides methods for converting numpy arrays to
tensorflow tensors and handling various aspects of the PDE solution process.

The implementation supports:
    - Shape function and gradient tensor management
    - Dirichlet boundary data processing
    - Test point generation and handling
    - Sensor data management for inverse problems
    - Bilinear parameter tensor conversion
    - Forcing function data handling

Key classes:
    - DataHandler2D: Implementation for 2D PDE data handling

Dependencies:
    - tensorflow: For tensor operations
    - numpy: For numerical arrays
    - FESpace2D: For finite element space handling
    - Domain2D: For domain management

Note:
    The implementation follows FastVPINNs methodology [1] for efficient
    handling of finite element data structures.

References:
    [1] FastVPINNs: Tensor-Driven Acceleration of VPINNs for Complex Geometries
        DOI: https://arxiv.org/abs/2404.12063
"""

from ...fe.fespace2d import *
from ...geometry.geometry_2d import *
import tensorflow as tf

from .datahandler import DataHandler


class DataHandler2D(DataHandler):
    """Handles data conversion and management for 2D PDE problems.

    This class implements the DataHandler interface for 2D problems,
    providing methods for converting finite element data to tensorflow
    tensors and managing various aspects of the PDE solution process.

    Attributes:
        fespace: Finite element space object for mesh and element info
        domain: Domain object for geometric information
        dtype: TensorFlow data type for tensor conversion
        shape_val_mat_list: Tensor of shape function values
            Shape: List of matrices of shape (n_test_functions, n_quad_points) with length n_elements
        grad_x_mat_list: Tensor of x-derivatives
            Shape: List of matrices of shape (n_test_functions, n_quad_points) with length n_elements
        grad_y_mat_list: Tensor of y-derivatives
            Shape: List of matrices of shape (n_test_functions, n_quad_points) with length n_elements
        x_pde_list: Tensor of quadrature point coordinates
        forcing_function_list: Tensor of forcing function values
        test_points: Tensor of test point coordinates

    Example:
        >>> fespace = FESpace2D(mesh, elements)
        >>> domain = Domain2D(bounds)
        >>> handler = DataHandler2D(fespace, domain, tf.float32)
        >>> dirichlet_input, dirichlet_vals = handler.get_dirichlet_input()
        >>> test_points = handler.get_test_points()

    Note:
        All input numpy arrays are assumed to be float64. The class handles
        conversion to the specified tensorflow dtype (typically float32)
        for computational efficiency.
    """

    def __init__(self, fespace, domain, dtype):
        """
        Constructor for the DataHandler2D class

        Args:
            fespace (FESpace2D): The FESpace2D object.
            domain (Domain2D): The Domain2D object.
            dtype (tf.DType): The tensorflow dtype to be used for all the tensors.

        Returns:
            None
        """
        # call the parent class constructor
        super().__init__(fespace=fespace, domain=domain, dtype=dtype)

        self.shape_val_mat_list = []
        self.grad_x_mat_list = []
        self.grad_y_mat_list = []
        self.x_pde_list = []
        self.forcing_function_list = []

        # check if the given dtype is a valid tensorflow dtype
        if not isinstance(self.dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        for cell_index in range(self.fespace.n_cells):
            shape_val_mat = tf.constant(
                self.fespace.get_shape_function_val(cell_index), dtype=self.dtype
            )
            grad_x_mat = tf.constant(
                self.fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype
            )
            grad_y_mat = tf.constant(
                self.fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype
            )
            x_pde = tf.constant(
                self.fespace.get_quadrature_actual_coordinates(cell_index),
                dtype=self.dtype,
            )
            forcing_function = tf.constant(
                self.fespace.get_forcing_function_values(cell_index), dtype=self.dtype
            )
            self.shape_val_mat_list.append(shape_val_mat)
            self.grad_x_mat_list.append(grad_x_mat)
            self.grad_y_mat_list.append(grad_y_mat)
            self.x_pde_list.append(x_pde)
            self.forcing_function_list.append(forcing_function)

        # now convert all the shapes into 3D tensors for easy multiplication
        # input tensor - x_pde_list
        self.x_pde_list = tf.reshape(self.x_pde_list, [-1, 2])

        self.forcing_function_list = tf.concat(self.forcing_function_list, axis=1)

        self.shape_val_mat_list = tf.stack(self.shape_val_mat_list, axis=0)
        self.grad_x_mat_list = tf.stack(self.grad_x_mat_list, axis=0)
        self.grad_y_mat_list = tf.stack(self.grad_y_mat_list, axis=0)

        # test points
        self.test_points = None

    def get_dirichlet_input(self) -> tuple:
        """
        This function will return the input for the Dirichlet boundary data

        Args:
            None

        Returns:
            The Dirichlet boundary data as a tuple of tensors
        """
        input_dirichlet, actual_dirichlet = (
            self.fespace.generate_dirichlet_boundary_data()
        )

        # convert to tensors
        input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

        return input_dirichlet, actual_dirichlet

    def get_test_points(self) -> tf.Tensor:
        """
        Get the test points for the given domain.

        Args:
            None

        Returns:
            The test points as a tensor
        """
        self.test_points = self.domain.get_test_points()
        self.test_points = tf.constant(self.test_points, dtype=self.dtype)
        return self.test_points

    def get_bilinear_params_dict_as_tensors(self, function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            function (function): The function from the example file which returns the bilinear parameters dictionary

        Returns:
            The bilinear parameters dictionary with all the values converted to tensors
        """
        # get the dictionary of bilinear parameters
        bilinear_params_dict = function()

        # loop over all keys and convert the values to tensors
        for key in bilinear_params_dict.keys():
            bilinear_params_dict[key] = tf.constant(
                bilinear_params_dict[key], dtype=self.dtype
            )

        return bilinear_params_dict

    # to be used only in inverse problems
    def get_sensor_data(
        self, exact_sol, num_sensor_points, mesh_type, file_name=None
    ) -> tuple:
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            exact_sol (function): The exact solution function
            num_sensor_points (int): The number of sensor points
            mesh_type (str): The type of mesh
            file_name (str): The file name to save the sensor data

        Returns:
            The sensor data as a tensor

        Raises:
            ValueError: If the mesh type is not internal or external
        """
        print(f"mesh_type = {mesh_type}")
        if mesh_type == "internal":
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data(
                exact_sol, num_sensor_points
            )
        elif mesh_type == "external":
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data_external(
                exact_sol, num_sensor_points, file_name
            )
        # convert the points and sensor values into tensors
        points = tf.constant(points, dtype=self.dtype)
        sensor_values = tf.constant(sensor_values, dtype=self.dtype)

        sensor_values = tf.reshape(sensor_values, [-1, 1])
        points = tf.reshape(points, [-1, 2])

        return points, sensor_values

    # get inverse param dict as tensors
    def get_inverse_params(self, inverse_params_dict_function) -> dict:
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            inverse_params_dict_function (function): The function from the example file which returns the inverse parameters dictionary

        Returns:
            The inverse parameters dictionary with all the values converted to tensors
        """
        # loop over all keys and convert the values to tensors

        inverse_params_dict = inverse_params_dict_function()

        for key in inverse_params_dict.keys():
            inverse_params_dict[key] = tf.constant(
                inverse_params_dict[key], dtype=self.dtype
            )

        return inverse_params_dict

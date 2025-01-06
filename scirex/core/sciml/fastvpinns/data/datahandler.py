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

"""
Abstract Base Interface for Neural Network Data Handling in PDEs.

This module provides the base interface for handling data transformations
and tensor conversions required for neural network-based PDE solvers. It
defines the essential structure for managing various data types involved
in finite element computations.

The implementation supports:
    - Finite element data processing
    - Dirichlet boundary condition handling
    - Test point generation and management
    - Bilinear parameter tensor conversion
    - Sensor data generation and handling
    - Parameter management for inverse problems

Key classes:
    - DataHandler: Abstract base class for PDE data handling

Note:
    All implementations assume double precision (float64) numpy arrays
    as inputs, with optional conversion to float32 for computational
    efficiency.

Dependencies:
    - abc: For abstract base class functionality
    - tensorflow: For tensor operations
    - numpy: For numerical arrays

Authors:
    - Thivin Anandh (https://thivinanandh.github.io/)

Versions:
    - 27-Dec-2024 (Version 0.1): Initial Implementation
"""

from abc import abstractmethod


class DataHandler:
    """Abstract base class for PDE solution data handling and tensor conversion.

    This class defines the interface for managing and converting various data types
    required in neural network-based PDE solvers. It handles transformation between
    numpy arrays and tensorflow tensors, and manages different aspects of the finite
    element computation data.

    Attributes:
        fespace: Finite element space object containing mesh and element info
        domain: Domain object containing geometric and boundary information
        dtype: TensorFlow data type for tensor conversion (float32/float64)

    Example:
        >>> class MyDataHandler(DataHandler):
        ...     def __init__(self, fespace, domain):
        ...         super().__init__(fespace, domain, tf.float32)
        ...
        ...     def get_dirichlet_input(self):
        ...         # Implementation
        ...         pass
        ...
        ...     def get_test_points(self):
        ...         # Implementation
        ...         pass

    Note:
        Concrete implementations must override:
        - get_dirichlet_input()
        - get_test_points()
        - get_bilinear_params_dict_as_tensors()
        - get_sensor_data()
        - get_inverse_params()

        All methods should handle type conversion between numpy arrays
        and tensorflow tensors consistently.
    """

    def __init__(self, fespace, domain, dtype):
        """
        Constructor for the DataHandler class

        Args:
            fespace (FESpace2D): The FESpace2D object.
            domain (Domain2D): The Domain2D object.
            dtype (tf.DType): The tensorflow dtype to be used for all the tensors.

        Returns:
            None
        """

        self.fespace = fespace
        self.domain = domain
        self.dtype = dtype

    @abstractmethod
    def get_dirichlet_input(self) -> tuple:
        """
        This function will return the input for the Dirichlet boundary data

        Args:
            None

        Returns:
            The Dirichlet boundary data as a tuple of tensors
        """

    @abstractmethod
    def get_test_points(self):
        """
        Get the test points for the given domain.

        Args:
            None

        Returns:
            The test points as a tensor
        """

    @abstractmethod
    def get_bilinear_params_dict_as_tensors(self, function) -> dict:
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            function (function): The function from the example file which returns the bilinear parameters dictionary

        Returns:
            The bilinear parameters dictionary with all the values converted to tensors
        """

    @abstractmethod
    def get_sensor_data(self, exact_sol, num_sensor_points, mesh_type, file_name=None):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            exact_sol (function): The exact solution function
            num_sensor_points (int): The number of sensor points
            mesh_type (str): The type of mesh
            file_name (str): The file name to save the sensor data

        Returns:
            The sensor data as a tensor
        """

    @abstractmethod
    def get_inverse_params(self, inverse_params_dict_function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Args:
            inverse_params_dict_function (function): The function from the example file which returns the inverse parameters dictionary

        Returns:
            The inverse parameters dictionary with all the values converted to tensors
        """

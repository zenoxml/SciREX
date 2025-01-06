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
    Module: fespace.py

    This module provides abstract base functionality for finite element space 
    implementations, defining core interfaces for fe computations and analysis.

    Classes:
        Fespace: Abstract base class for finite element space implementations

    Dependencies:
        - numpy: For numerical computations
        - abc: For abstract base class functionality

    Key Features:
        - Mesh handling and cell management
        - Shape function evaluation and manipulation
        - Boundary condition implementation
            * Dirichlet boundary handling
            * Boundary function management
        - Quadrature point management
        - Forcing function integration
        - Sensor data handling for inverse problems
        - Reference and physical space transformations
        - Gradient computation in both spaces

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
        27/Dec/2024: Initial version - Thivin Anandh D

    Notes:
        This module serves as the foundation for specific finite element
        space implementations. It defines the minimum interface required
        for fe computations while allowing flexibility in concrete
        implementations.
"""

import numpy as np
from abc import abstractmethod


class Fespace:
    """
    Abstract base class defining the interface for finite element spaces.

    This class provides the foundation for implementing finite element spaces,
    including mesh handling, element operations, and solution computations.

    Attributes:
        mesh: Mesh object containing geometric information
        cells (ndarray): Array of cell indices
        boundary_points (dict): Dictionary of boundary point information
        cell_type (str): Type of finite element cell
        fe_order (int): Order of finite element approximation
        fe_type (str): Type of finite element basis
        quad_order (int): Order of quadrature rule
        quad_type (str): Type of quadrature formula
        fe_transformation_type (str): Type of geometric transformation
        bound_function_dict (dict): Dictionary of boundary condition functions
        bound_condition_dict (dict): Dictionary of boundary condition types
        forcing_function (callable): Source term function
        output_path (str): Path for output files

    Example:
        >>> class MyFespace(Fespace):
        ...     def set_finite_elements(self):
        ...         # Implementation
        ...         pass
        ...     def generate_dirichlet_boundary_data(self):
        ...         # Implementation
        ...         pass
        ...     # Implement other abstract methods

    Notes:
        - All coordinate transformations must be implemented
        - Shape function values and gradients are available in both
        reference and physical spaces
        - Supports both internal and external sensor data for
        inverse problems
        - Boundary conditions must be properly specified through
        the boundary dictionaries
    """

    def __init__(
        self,
        mesh,
        cells,
        boundary_points,
        cell_type: str,
        fe_order: int,
        fe_type: str,
        quad_order: int,
        quad_type: str,
        fe_transformation_type: str,
        bound_function_dict: dict,
        bound_condition_dict: dict,
        forcing_function,
        output_path: str,
    ) -> None:
        """
        The constructor of the Fespace2D class.

        Args:
            mesh: The mesh object.
            cells: The cells of the mesh.
            boundary_points: The boundary points of the mesh.
            cell_type: The type of the cell.
            fe_order: The order of the finite element.
            fe_type: The type of the finite element.
            quad_order: The order of the quadrature.
            quad_type: The type of the quadrature.
            fe_transformation_type: The type of the finite element transformation.
            bound_function_dict: The dictionary of the boundary functions.
            bound_condition_dict: The dictionary of the boundary conditions.
            forcing_function: The forcing function.
            output_path: The path to the output directory.

        Returns:
            None
        """
        self.mesh = mesh
        self.boundary_points = boundary_points
        self.cells = cells
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type

        self.fe_transformation_type = fe_transformation_type
        self.output_path = output_path
        self.bound_function_dict = bound_function_dict
        self.bound_condition_dict = bound_condition_dict
        self.forcing_function = forcing_function

    @abstractmethod
    def set_finite_elements(self) -> None:
        """
        Assigns the finite elements to each cell.

        This method initializes the finite element objects for each cell in the mesh.
        It creates an instance of the `FE2D_Cell` class for each cell, passing the necessary parameters.
        The finite element objects store information about the basis functions, gradients, Jacobians,
        quadrature points, weights, actual coordinates, and forcing functions associated with each cell.

        After initializing the finite element objects, this method prints the shape details of various matrices
        and updates the total number of degrees of freedom (dofs) for the entire mesh.

        Args:
            None

        Returns:
            None
        """

    @abstractmethod
    def generate_dirichlet_boundary_data(self) -> np.ndarray:
        """
        Generate Dirichlet boundary data.

        This function returns the boundary points and their corresponding values.

        Args:
            None

        Returns:
            np.ndarray: The boundary points and their values.

        Notes:
            The boundary points and values are stored in the `boundary_points` attribute of the `Fespace` object.
        """

    @abstractmethod
    def get_shape_function_val(self, cell_index: int) -> np.ndarray:
        """
        Get the actual values of the shape functions on a given cell.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the actual values of the shape functions.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_x(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the gradient of the shape function with respect to the x-coordinate.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_x_ref(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate on the reference element.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the gradient of the shape function with respect to the x-coordinate on the reference element.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_y(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to y at the given cell index.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the gradient of the shape function with respect to y.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_shape_function_grad_y_ref(self, cell_index: int):
        """
        Get the gradient of the shape function with respect to y at the reference element.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the gradient of the shape function with respect to y at the reference element.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.

        Notes:
            This function returns the gradient of the shape function with respect to y at the reference element
            for a given cell. The shape function gradient values are stored in the `basis_grady_at_quad_ref` array
            of the corresponding finite element cell. The `cell_index` parameter specifies the index of the cell
            for which the shape function gradient is required. If the `cell_index` is greater than the total number
            of cells, a `ValueError` is raised. The returned gradient values are copied from the `basis_grady_at_quad_ref` array to ensure immutability.
        """

    @abstractmethod
    def get_quadrature_actual_coordinates(self, cell_index: int) -> np.ndarray:
        """
        Get the actual coordinates of the quadrature points for a given cell.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the actual coordinates of the quadrature points.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """

    @abstractmethod
    def get_forcing_function_values(self, cell_index: int) -> np.ndarray:
        """
        Get the forcing function values at the quadrature points.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: An array containing the forcing function values at the quadrature points.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.

        Notes:
            This function computes the forcing function values at the quadrature points for a given cell.
            It loops over all the basis functions and computes the integral using the actual coordinates
            and the basis functions at the quadrature points. The resulting values are stored in the
            `forcing_at_quad` attribute of the corresponding `fe_cell` object. The forcing function is evaluated using the `forcing_function` method of the `fe_cell`
            object.
        """

    @abstractmethod
    def get_sensor_data(self, exact_solution, num_points: int) -> np.ndarray:
        """
        Obtain sensor data (actual solution) at random points.

        Args:
            exact_solution (ndarray): The exact solution values.
            num_points (int): The number of points to sample from the domain.

        Returns:
            np.ndarray: The sensor data at the given points.

        Notes:
            This method is used in the inverse problem to obtain the sensor data at random points within the domain. Currently, it only works for problems with an analytical solution.
            Methodologies to obtain sensor data for problems from a file are not implemented yet.
            It is also not implemented for external or complex meshes.
        """

    @abstractmethod
    def get_sensor_data_external(
        self, exact_sol, num_points: int, file_name: str
    ) -> np.ndarray:
        """
        This method is used to obtain the sensor data from an external file when there is no analytical solution available.

        Args:
            exact_sol: The exact solution values.
            num_points: The number of points to sample from the domain.
            file_name: The name of the file containing the sensor data.

        Returns:
            np.ndarray: The sensor data at the given points based on the external file.

        """

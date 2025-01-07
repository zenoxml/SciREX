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
    Module: fe_transformation_3d.py

    This module provides the abstract base class for all 3D finite element transformations.
    It defines the interface for mapping between reference and physical coordinates in 
    three-dimensional finite element analysis.

    Classes:
        FETransformation3D: Abstract base class for 3D finite element transformations

    Dependencies:
        - abc: For abstract base class functionality

    Key Features:
        - Abstract interface for 3D coordinate transformations
        - Reference to physical domain mapping
        - 3D Jacobian matrix computation
        - Support for different 3D element geometries
        - Cell geometry specification interface
        - Systematic transformation validation
        - Three-dimensional coordinate mapping (xi, eta, zeta)

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
        27/Dec/2024: Initial version - Thivin Anandh D

    References:
        None
"""

from abc import abstractmethod
import numpy as np


class FETransforamtion3D:  # pragma: no cover
    """
    A base class for 3D finite element transformations.

    This abstract class defines the interface for mapping between reference and physical
    coordinates in 3D finite element analysis. Implementations must provide specific
    transformation rules for different three-dimensional element types.

    Attributes:
        None

    Methods:
        set_cell():
            Sets the physical coordinates of the element vertices.
            Must be implemented by derived classes.

        get_original_from_ref(xi, eta, zeta):
            Maps coordinates from reference to physical domain.
            Must be implemented by derived classes.
            Args: xi, eta, zeta - Reference coordinates
            Returns: (x, y, z) physical coordinates

        get_jacobian(xi, eta, zeta):
            Computes the Jacobian matrix of the transformation.
            Must be implemented by derived classes.
            Args: xi, eta, zeta - Reference coordinates
            Returns: determinant of the Jacobian at all the quadrature points

    Example:
        >>> class HexTransform(FETransformation3D):
        ...     def set_cell(self, vertices):
        ...         self.vertices = vertices
        ...     def get_original_from_ref(self, xi, eta, zeta):
        ...         # Implementation for hexahedral element
        ...         pass
        ...     def get_jacobian(self, xi, eta, zeta):
        ...         # Implementation for hexahedral element
        ...         pass

    Notes:
        - Reference domain is typically [-1,1] × [-1,1] × [-1,1]
        - Transformations must be invertible
        - Implementations should handle element distortion
        - Jacobian is used for both mapping and integration
        - Coordinate order is consistently (xi, eta, zeta) -> (x, y, z)
    """

    def __init__(self):
        """
        Constructor for the FETransformation3D class.
        """
        pass

    @abstractmethod
    def set_cell(self):
        """
        Set the cell co-ordinates, which will be used to calculate the Jacobian and actual values.

        Args:
            None
        """

    @abstractmethod
    def get_original_from_ref(
        self, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray
    ) -> np.ndarray:
        """
        This method returns the original coordinates from the reference coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: Returns the transformed original coordinates from the reference coordinates.
        """

    @abstractmethod
    def get_original_from_ref(
        self, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray
    ) -> np.ndarray:
        """
        This method returns the original co-ordinates from the reference co-ordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: Returns the transformed original co-ordinates from the reference co-ordinates.
        """

    @abstractmethod
    def get_jacobian(
        self, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray
    ) -> np.ndarray:
        """
        This method returns the Jacobian of the transformation.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: Returns the Jacobian of the transformation.
        """

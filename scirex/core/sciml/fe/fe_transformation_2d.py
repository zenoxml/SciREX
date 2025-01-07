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
    Module: fe_transformation_2d.py

    This module provides the abstract base class for all 2D finite element transformations. 
    It defines the interface for mapping between reference and physical coordinates in 
    two-dimensional finite element analysis.

    Classes:
        FETransformation2D: Abstract base class for 2D finite element transformations

    Dependencies:
        - abc: For abstract base class functionality
        - quad_affine: For affine transformation implementations
        - quad_bilinear: For bilinear transformation implementations

    Key Features:
        - Abstract interface for coordinate transformations
        - Reference to physical domain mapping
        - Jacobian matrix computation
        - Support for different element geometries
        - Cell geometry specification interface
        - Systematic transformation validation

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
       27/Dec/2024: Initial version - Thivin Anandh D

    References:
        None
"""

from abc import abstractmethod
import numpy as np


class FETransforamtion2D:
    """
    A base class for 2D finite element transformations.

    This abstract class defines the interface for mapping between reference and physical
    coordinates in 2D finite element analysis. Implementations must provide specific
    transformation rules for different element types.

    Attributes:
        None

    Methods:
        set_cell():
            Sets the physical coordinates of the element vertices.
            Must be implemented by derived classes.

        get_original_from_ref(xi, eta):
            Maps coordinates from reference to physical domain.
            Must be implemented by derived classes.

        get_jacobian(xi, eta):
            Computes the Jacobian matrix of the transformation.
            Must be implemented by derived classes.

    Example:
        >>> class QuadTransform(FETransformation2D):
        ...     def set_cell(self, vertices):
        ...         self.vertices = vertices
        ...     def get_original_from_ref(self, xi:np.ndarray, eta:np.ndarray) -> np.ndarray:
        ...         # Implementation for quad element
        ...         pass
        ...     def get_jacobian(self, xi: np.ndarray, eta:np.ndarray) -> np.ndarray:
        ...         # Implementation for quad element
        ...         pass

    Notes:
        - Reference domain is typically [-1,1] × [-1,1]
        - Transformations must be invertible
        - Implementations should handle element distortion
        - Jacobian is used for both mapping and integration
    """

    def __init__(self):
        """
        Constructor for the FETransforamtion2D class.
        """

    @abstractmethod
    def set_cell(self):
        """
        Set the cell coordinates, which will be used to calculate the Jacobian and actual values.

        :return: None
        """

    @abstractmethod
    def get_original_from_ref(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the original coordinates from the reference coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: Returns the transformed original coordinates from the reference coordinates.
        """

    @abstractmethod
    def get_jacobian(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the Jacobian of the transformation.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: Returns the Jacobian of the transformation.
        """


## Mandatory, Import all the basis functions here (Quad element Transformations)
from .quad_affine import *
from .quad_bilinear import *

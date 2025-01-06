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

"""Abstract Base Class for Numerical Quadrature Formulas.

This module provides the base interface for implementing various numerical 
quadrature schemes. It defines the common structure and required methods 
that all quadrature implementations must follow.

Key functionalities:
    - Abstract interface for quadrature rule implementations
    - Standard methods for accessing quadrature points and weights
    - Flexible framework for different element types and dimensions

The module serves as a foundation for:
    - Multiple quadrature rule implementations
    - Different element type integrations
    - Various quadrature orders and types
    - Custom quadrature scheme implementations

Key classes:
    - Quadratureformulas: Abstract base class for all quadrature implementations

Dependencies:
    - abc: For abstract base class functionality

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version Info:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

from abc import abstractmethod
import numpy as np


class Quadratureformulas:
    """Abstract base class for numerical quadrature formulas.

    This class defines the interface that all quadrature implementations must
    follow. It provides the basic structure for implementing various quadrature
    rules while ensuring consistent access to quadrature data.

    Attributes:
        quad_order: Order of the quadrature rule
        quad_type: Type of quadrature (e.g., 'gauss-legendre', 'gauss-jacobi')
        num_quad_points: Total number of quadrature points

    Example:
        >>> class MyQuadrature(Quadratureformulas):
        ...     def __init__(self):
        ...         super().__init__(quad_order=3,
        ...                         quad_type='custom',
        ...                         num_quad_points=9)
        ...     def get_quad_values(self):
        ...         # Implementation
        ...         pass
        ...     def get_num_quad_points(self):
        ...         return self.num_quad_points

    Note:
        This is an abstract base class. Concrete implementations must override:
        - get_quad_values()
        - get_num_quad_points()

        The implementation should ensure proper initialization of:
        - Quadrature points
        - Quadrature weights
        - Number of quadrature points
    """

    def __init__(self, quad_order: int, quad_type: str, num_quad_points: int):
        """
        Constructor for the Quadratureformulas_Quad2D class.

        Args:
            quad_order: Order of quadrature rule
            quad_type: Type of quadrature ('gauss-legendre' or 'gauss-jacobi')
            num_quad_points: Total number of quadrature points

        Returns:
            None
        """
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.num_quad_points = num_quad_points

    @abstractmethod
    def get_quad_values(self):
        """
        Returns the quadrature weights, xi and eta values.

        Args:
            None

        Returns:
            weights: Weights for each quadrature point
            xi: x-coordinates of quadrature points in reference element
            eta: y-coordinates of quadrature points in reference element
        """

    @abstractmethod
    def get_num_quad_points(self):
        """
        Returns the number of quadrature points.

        Args:
            None

        Returns:
            num_quad_points: Total number of quadrature points
        """

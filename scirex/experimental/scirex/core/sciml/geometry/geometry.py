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

"""Abstract Base Interface for Geometry and Mesh Operations.

This module provides the base interface for implementing geometry and mesh
handling capabilities in both 2D and 3D. It defines the essential structure
for mesh operations including reading, generation, and manipulation.

Key functionalities:
    - Abstract interface for mesh reading operations
    - Common mesh generation method definitions
    - VTK file generation specifications
    - Test point extraction framework
    - Mesh type and generation method standardization

The module serves as a foundation for:
    - Both 2D and 3D mesh implementations
    - Various element type support
    - Multiple mesh generation approaches
    - Consistent mesh handling interface

Key classes:
    - Geometry: Abstract base class for all geometry implementations

Dependencies:
    - numpy: For numerical operations
    - meshio: For mesh input/output operations
    - gmsh: For mesh generation capabilities
    - matplotlib: For visualization
    - pyDOE: For sampling methods
    - abc: For abstract base class functionality

Note:
    This module provides only the interface definitions. Concrete
    implementations must be provided by derived classes for specific
    dimensional and element type requirements.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import meshio
from pyDOE import lhs

import gmsh

from abc import abstractmethod


class Geometry:
    """Abstract base class for geometry and mesh operations.

    This class defines the interface that all geometry implementations must
    follow, providing the basic structure for mesh handling operations in
    both 2D and 3D contexts.

    Attributes:
        mesh_type: Type of mesh elements (e.g., 'quadrilateral', 'triangle')
        mesh_generation_method: Method for mesh generation ('internal'/'external')

    Example:
        >>> class Geometry2D(Geometry):
        ...     def __init__(self, mesh_type='quadrilateral',
        ...                  method='internal'):
        ...         super().__init__(mesh_type, method)
        ...
        ...     def read_mesh(self, mesh_file, boundary_level,
        ...                   sampling_method, refine_level):
        ...         # Implementation
        ...         pass
        ...
        ...     def generate_vtk_for_test(self):
        ...         # Implementation
        ...         pass
        ...
        ...     def get_test_points(self):
        ...         # Implementation
        ...         return points

    Note:
        This is an abstract base class. Concrete implementations must override:
        - read_mesh()
        - generate_vtk_for_test()
        - get_test_points()

        Each implementation should provide appropriate mesh handling for its
        specific dimensional and element type requirements.
    """

    def __init__(self, mesh_type: str, mesh_generation_method: str):
        """
        Constructor for the Geometry class.

        Args:
            mesh_type: Type of mesh elements (e.g., 'quadrilateral', 'triangle')
            mesh_generation_method: Method for mesh generation ('internal'/'external')

        Returns:
            None
        """
        self.mesh_type = mesh_type
        self.mesh_generation_method = mesh_generation_method

    @abstractmethod
    def read_mesh(
        self,
        mesh_file: str,
        boundary_point_refinement_level: int,
        bd_sampling_method: str,
        refinement_level: int,
    ):
        """
        Abstract method to read mesh from Gmsh. This method should be implemented by the derived classes.

        Args:
            mesh_file (str): Path to the mesh file
            boundary_point_refinement_level (int): Level of refinement for boundary points
            bd_sampling_method (str): Sampling method for boundary points
            refinement_level (int): Level of mesh refinement

        Returns:
            None
        """

    @abstractmethod
    def generate_vtk_for_test(self):
        """
        Generates a VTK from Mesh file (External) or using gmsh (for Internal).

        Args:
        None

        Returns:
        None
        """

    @abstractmethod
    def get_test_points(self):
        """
        This function is used to extract the test points from the given mesh

        Args:
            None

        Returns:
            points (np.ndarray): Test points extracted from the mesh
        """

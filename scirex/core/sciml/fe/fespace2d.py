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


"""Finite Element Space Implementation for 2D Problems.

This module implements finite element space functionality for 2D domains, providing
a framework for handling mesh elements, boundary conditions, and numerical integration.

Key classes:
    - Fespace2D: Main class for managing 2D finite element spaces
    - FE2D_Cell: Implementation of individual finite element cells

Key functionalities:
    - Finite element space construction and management
    - Boundary condition handling (Dirichlet)
    - Shape function and gradient computations
    - Quadrature point and weight management
    - Forcing function evaluation
    - Sensor data generation for inverse problems

The implementation supports:
    - Various element types (currently focused on quadrilateral elements)
    - Different orders of finite elements
    - Custom quadrature rules
    - Multiple boundary conditions
    - Forcing function integration
    - Mesh visualization

Note:
    Triangle mesh support is currently not implemented.

Dependencies:
    - numpy: For numerical computations
    - meshio: For mesh handling
    - matplotlib: For visualization
    - tensorflow: For optimization tasks
    - pyDOE: For Latin Hypercube Sampling
    - pandas: For data handling

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version Info:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

import numpy as np
import meshio
from .fe2d_cell import FE2D_Cell

# from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm

# import plotting
import matplotlib.pyplot as plt

# import path
from pathlib import Path

# import tensorflow
import tensorflow as tf

from ..utils.print_utils import print_table

from pyDOE import lhs
import pandas as pd

from matplotlib import rc
from cycler import cycler


plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20

plt.rcParams["legend.fontsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "darkblue",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#bcbd22",
        "#8c564b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#7f7f7f",
    ]
)

from .fespace import Fespace


class Fespace2D(Fespace):
    """
    Represents a finite element space in 2D. This class provides functionality for handling 2D finite element spaces,
    including mesh generation, basis function evaluation, and boundary condition handling.

    Args:
        mesh (meshio.Mesh): The mesh object containing the mesh information.
        cells (np.ndarray): The cell information from the mesh.
        boundary_points (dict): The boundary points information from the mesh.
        cell_type (str): The type of the cell (e.g., 'quadrilateral').
        fe_order (int): The order of the finite element basis functions.
        fe_type (str): The type of the finite element basis functions (e.g., 'legendre').
        quad_order (int): The order of the quadrature rule.
        quad_type (str): The type of the quadrature rule (e.g., 'gauss-legendre').
        fe_transformation_type (str): The type of the finite element transformation (e.g., 'affine').
        bound_function_dict (dict): A dictionary containing the boundary functions.
        bound_condition_dict (dict): A dictionary containing the boundary conditions.
        forcing_function (function): The forcing function for the problem.
        output_path (str): The path to save the output files.
        generate_mesh_plot (bool): Flag to generate the mesh plot (default: False).

    Raises:
        ValueError: If the cell type is not supported.

    Returns:
        None
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
        generate_mesh_plot: bool = False,
    ) -> None:
        """
        The constructor of the Fespace2D class.
        """
        # call the constructor of the parent class
        super().__init__(
            mesh=mesh,
            cells=cells,
            boundary_points=boundary_points,
            cell_type=cell_type,
            fe_order=fe_order,
            fe_type=fe_type,
            quad_order=quad_order,
            quad_type=quad_type,
            fe_transformation_type=fe_transformation_type,
            bound_function_dict=bound_function_dict,
            bound_condition_dict=bound_condition_dict,
            forcing_function=forcing_function,
            output_path=output_path,
        )

        if self.cell_type == "triangle":
            raise ValueError(
                "Triangle Mesh is not supported yet"
            )  # added by thivin - to remove support for triangular mesh

        self.generate_mesh_plot = generate_mesh_plot

        # to be calculated in the plot function
        self.total_dofs = 0
        self.total_boundary_dofs = 0

        # to be calculated on get_boundary_data_dirichlet function
        self.total_dirichlet_dofs = 0

        # get the number of cells
        self.n_cells = self.cells.shape[0]

        self.fe_cell = []

        # Function which assigns the fe_cell for each cell
        self.set_finite_elements()

        # generate the plot of the mesh
        if self.generate_mesh_plot:
            self.generate_plot(self.output_path)
        # self.generate_plot(self.output_path)

        # Obtain boundary Data
        self.dirichlet_boundary_data = self.generate_dirichlet_boundary_data()

        title = [
            "Number of Cells",
            "Number of Quadrature Points",
            "Number of Dirichlet Boundary Points",
            "Quadrature Order",
            "fe Order",
            "fe Type",
            "fe Transformation Type",
        ]
        values = [
            self.n_cells,
            self.total_dofs,
            self.total_dirichlet_dofs,
            self.quad_order,
            self.fe_order,
            self.fe_type,
            self.fe_transformation_type,
        ]
        # print the table
        print_table("fe Space Information", ["Property", "Value"], title, values)

    def set_finite_elements(self) -> None:
        """
        Assigns the finite elements to each cell.

        This method initializes the finite element objects for each cell in the mesh.
        It creates an instance of the `FE2D_Cell` class for each cell, passing the necessary parameters.
        The finite element objects store information about the basis functions, gradients, Jacobians,
        quadrature points, weights, actual coordinates, and forcing functions associated with each cell.

        After initializing the finite element objects, this method prints the shape details of various matrices
        and updates the total number of degrees of freedom (dofs) for the entire mesh.

        :return: None
        """
        progress_bar = tqdm(
            total=self.n_cells,
            desc="Fe2D_cell Setup",
            unit="cells_assembled",
            bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
            colour="blue",
            ncols=100,
        )

        dof = 0
        for i in range(self.n_cells):
            self.fe_cell.append(
                FE2D_Cell(
                    self.cells[i],
                    self.cell_type,
                    self.fe_order,
                    self.fe_type,
                    self.quad_order,
                    self.quad_type,
                    self.fe_transformation_type,
                    self.forcing_function,
                )
            )

            # obtain the shape of the basis function (n_test, N_quad)
            dof += self.fe_cell[i].basis_at_quad.shape[1]

            progress_bar.update(1)
        # print the Shape details of all the matrices from cell 0 using print_table function
        title = [
            "Shape function Matrix Shape",
            "Shape function Gradient Matrix Shape",
            "Jacobian Matrix Shape",
            "Quadrature Points Shape",
            "Quadrature Weights Shape",
            "Quadrature Actual Coordinates Shape",
            "Forcing Function Shape",
        ]
        values = [
            self.fe_cell[0].basis_at_quad.shape,
            self.fe_cell[0].basis_gradx_at_quad.shape,
            self.fe_cell[0].jacobian.shape,
            self.fe_cell[0].quad_xi.shape,
            self.fe_cell[0].quad_weight.shape,
            self.fe_cell[0].quad_actual_coordinates.shape,
            self.fe_cell[0].forcing_at_quad.shape,
        ]
        print_table("fe Matrix Shapes", ["Matrix", "Shape"], title, values)

        # update the total number of dofs
        self.total_dofs = dof

    def generate_plot(self, output_path) -> None:
        """
        Generate a plot of the mesh.

        Args:
            output_path (str): The path to save the output files.

        Returns:
            None
        """
        total_quad = 0
        marker_list = [
            "o",
            ".",
            ",",
            "x",
            "+",
            "P",
            "s",
            "D",
            "d",
            "^",
            "v",
            "<",
            ">",
            "p",
            "h",
            "H",
        ]

        print(f"[INFO] : Generating the plot of the mesh")
        # Plot the mesh
        plt.figure(figsize=(6.4, 4.8), dpi=300)

        # label flag ( to add the label only once)
        label_set = False

        # plot every cell as a quadrilateral
        # loop over all the cells
        for i in range(self.n_cells):
            # get the coordinates of the cell
            x = self.fe_cell[i].cell_coordinates[:, 0]
            y = self.fe_cell[i].cell_coordinates[:, 1]

            # add the first point to the end of the array
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            plt.plot(x, y, "k-", linewidth=0.5)

            # plot the quadrature points
            x_quad = self.fe_cell[i].quad_actual_coordinates[:, 0]
            y_quad = self.fe_cell[i].quad_actual_coordinates[:, 1]

            total_quad += x_quad.shape[0]

            if not label_set:
                plt.scatter(
                    x_quad, y_quad, marker="x", color="b", s=2, label="Quad Pts"
                )
                label_set = True
            else:
                plt.scatter(x_quad, y_quad, marker="x", color="b", s=2)

        self.total_dofs = total_quad

        bound_dof = 0
        # plot the boundary points
        # loop over all the boundary tags
        for i, (bound_id, bound_pts) in enumerate(self.boundary_points.items()):
            # get the coordinates of the boundary points
            x = bound_pts[:, 0]
            y = bound_pts[:, 1]

            # add the first point to the end of the array
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            bound_dof += x.shape[0]

            plt.scatter(
                x, y, marker=marker_list[i + 1], s=2, label=f"Bd-id : {bound_id}"
            )

        self.total_boundary_dofs = bound_dof

        plt.legend(bbox_to_anchor=(0.85, 1.02))
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(str(Path(output_path) / "mesh.png"), bbox_inches="tight")
        plt.savefig(str(Path(output_path) / "mesh.svg"), bbox_inches="tight")

        # print the total number of quadrature points
        print(f"Plots generated")
        print(f"[INFO] : Total number of cells = {self.n_cells}")
        print(f"[INFO] : Total number of quadrature points = {self.total_dofs}")
        print(f"[INFO] : Total number of boundary points = {self.total_boundary_dofs}")

    def generate_dirichlet_boundary_data(self) -> np.ndarray:
        """
        Generate Dirichlet boundary data. This function returns the boundary points and their corresponding values.

        Args:
            None

        Returns:
            tuple: The boundary points and their values as numpy arrays.
        """
        x = []
        y = []
        for bound_id, bound_pts in self.boundary_points.items():
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0], pt[1]], dtype=np.float64)
                x.append(pt_new)
                val = np.array(
                    self.bound_function_dict[bound_id](pt[0], pt[1]), dtype=np.float64
                ).reshape(-1, 1)
                y.append(val)

        print(f"[INFO] : Total number of Dirichlet boundary points = {len(x)}")
        self.total_dirichlet_dofs = len(x)
        print(f"[INFO] : Shape of Dirichlet-X = {np.array(x).shape}")
        print(f"[INFO] : Shape of Y = {np.array(y).shape}")

        return x, y

    def generate_dirichlet_boundary_data_vector(self, component: int) -> np.ndarray:
        """
        Generate the boundary data vector for the Dirichlet boundary condition. This function returns the boundary points and their corresponding values for a specific component.

        Args:
            component (int): The component of the boundary data vector.

        Returns:
            tuple: The boundary points and their values as numpy arrays.
        """
        x = []
        y = []
        for bound_id, bound_pts in self.boundary_points.items():
            # get the coordinates of the boundary points
            for pt in bound_pts:
                pt_new = np.array([pt[0], pt[1]], dtype=np.float64)
                x.append(pt_new)
                val = np.array(
                    self.bound_function_dict[bound_id](pt[0], pt[1])[component],
                    dtype=np.float64,
                ).reshape(-1, 1)
                y.append(val)

        return x, y

    def get_shape_function_val(self, cell_index: int) -> np.ndarray:
        """
        Get the actual values of the shape functions on a given cell.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The actual values of the shape functions on the given cell.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_at_quad.copy()

    def get_shape_function_grad_x(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The actual values of the shape functions on the given cell.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_gradx_at_quad.copy()

    def get_shape_function_grad_x_ref(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to the x-coordinate on the reference element.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The actual values of the shape functions on the given cell.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_gradx_at_quad_ref.copy()

    def get_shape_function_grad_y(self, cell_index: int) -> np.ndarray:
        """
        Get the gradient of the shape function with respect to y at the given cell index.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The actual values of the shape functions on the given cell.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_grady_at_quad.copy()

    def get_shape_function_grad_y_ref(self, cell_index: int):
        """
        Get the gradient of the shape function with respect to y at the reference element.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The actual values of the shape functions on the given cell.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.

        Note:
            This function returns the gradient of the shape function with respect to y at the reference element
            for a given cell. The shape function gradient values are stored in the `basis_grady_at_quad_ref` array
            of the corresponding finite element cell. The `cell_index` parameter specifies the index of the cell
            for which the shape function gradient is required. If the `cell_index` is greater than the total number
            of cells, a `ValueError` is raised. The returned gradient values are copied from the `basis_grady_at_quad_ref` array to ensure immutability.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].basis_grady_at_quad_ref.copy()

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
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].quad_actual_coordinates.copy()

    def get_quadrature_weights(self, cell_index: int) -> np.ndarray:
        """
        Return the quadrature weights for a given cell.

        Args:
            cell_index (int): The index of the cell for which the quadrature weights are needed.

        Returns:
            np.ndarray: The quadrature weights for the given cell  of dimension (N_Quad_Points, 1).

        Raises:
            ValueError: If cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        return self.fe_cell[cell_index].mult.copy()

    def get_forcing_function_values(self, cell_index: int) -> np.ndarray:
        """
        Get the forcing function values at the quadrature points.

        Args:
            cell_index (int): The index of the cell.

        Returns:
            np.ndarray: The forcing function values at the quadrature points.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.

        Note:
            This function computes the forcing function values at the quadrature points for a given cell.
            It loops over all the basis functions and computes the integral using the actual coordinates
            and the basis functions at the quadrature points. The resulting values are stored in the
            `forcing_at_quad` attribute of the corresponding `fe_cell` object.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        # Changed by Thivin: To assemble the forcing function at the quadrature points here in the fespace
        # so that it can be used to handle multiple dimensions on a vector valud problem

        # get number of shape functions
        n_shape_functions = self.fe_cell[cell_index].basis_function.num_shape_functions

        # Loop over all the basis functions and compute the integral
        f_integral = np.zeros((n_shape_functions, 1), dtype=np.float64)

        for i in range(n_shape_functions):
            val = 0
            for q in range(self.fe_cell[cell_index].basis_at_quad.shape[1]):
                x = self.fe_cell[cell_index].quad_actual_coordinates[q, 0]
                y = self.fe_cell[cell_index].quad_actual_coordinates[q, 1]
                # print("f_values[q] = ",f_values[q])

                # the Jacobian and the quadrature weights are pre multiplied to the basis functions
                val += (self.fe_cell[cell_index].basis_at_quad[i, q]) * self.fe_cell[
                    cell_index
                ].forcing_function(x, y)
                # print("val = ", val)

            f_integral[i] = val

        self.fe_cell[cell_index].forcing_at_quad = f_integral

        return self.fe_cell[cell_index].forcing_at_quad.copy()

    def get_forcing_function_values_vector(
        self, cell_index: int, component: int
    ) -> np.ndarray:
        """
        This function will return the forcing function values at the quadrature points
        based on the Component of the RHS Needed, for vector valued problems

        Args:
            cell_index (int): The index of the cell.
            component (int): The component of the forcing function.

        Returns:
            np.ndarray: The forcing function values at the quadrature points.

        Raises:
            ValueError: If the cell_index is greater than the number of cells.
        """
        if cell_index >= len(self.fe_cell) or cell_index < 0:
            raise ValueError(
                f"cell_index should be less than {self.n_cells} and greater than or equal to 0"
            )

        # get the coordinates
        x = self.fe_cell[cell_index].quad_actual_coordinates[:, 0]
        y = self.fe_cell[cell_index].quad_actual_coordinates[:, 1]

        # compute the forcing function values
        f_values = self.fe_cell[cell_index].forcing_function(x, y)[component]

        # compute the integral
        f_integral = np.sum(self.fe_cell[cell_index].basis_at_quad * f_values, axis=1)

        self.fe_cell[cell_index].forcing_at_quad = f_integral.reshape(-1, 1)

        return self.fe_cell[cell_index].forcing_at_quad.copy()

    def get_sensor_data(self, exact_solution, num_points: int):
        """
        Obtain sensor data (actual solution) at random points.

        This method is used in the inverse problem to obtain the sensor data at random points within the domain.
        Currently, it only works for problems with an analytical solution.
        Methodologies to obtain sensor data for problems from a file are not implemented yet.
        It is also not implemented for external or complex meshes.

        Args:
            exact_solution (function): The exact solution function.
            num_points (int): The number of points to sample.

        Returns:
            Tuple: A tuple containing two arrays: sensor points and the exact solution values.
        """
        # generate random points within the bounds of the domain
        # get the bounds of the domain
        x_min = np.min(self.mesh.points[:, 0])
        x_max = np.max(self.mesh.points[:, 0])
        y_min = np.min(self.mesh.points[:, 1])
        y_max = np.max(self.mesh.points[:, 1])
        # sample n random points within the bounds of the domain
        # Generate points in the unit square

        num_internal_points = int(num_points * 0.9)

        points = lhs(2, samples=num_internal_points)
        points[:, 0] = x_min + (x_max - x_min) * points[:, 0]
        points[:, 1] = y_min + (y_max - y_min) * points[:, 1]
        # get the exact solution at the points
        exact_sol = exact_solution(points[:, 0], points[:, 1])

        # print the shape of the points and the exact solution
        print(f"[INFO] : Number of sensor points = {points.shape[0]}")
        print(f"[INFO] : Shape of sensor points = {points.shape}")

        # plot the points
        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.scatter(points[:, 0], points[:, 1], marker="x", color="r", s=2)
        plt.axis("equal")
        plt.title("Sensor Points")
        plt.tight_layout()
        plt.savefig("sensor_points.png", bbox_inches="tight")

        return points, exact_sol

    def get_sensor_data_external(self, exact_sol, num_points: int, file_name: str):
        """
        This method is used to obtain the sensor data from an external file.

        Args:
            exact_sol (function): The exact solution function.
            num_points (int): The number of points to sample.
            file_name (str): The name of the file containing the sensor data.

        Returns:
            Tuple: A tuple containing two arrays: sensor points and the exact solution values.

        Note:
            This method reads the sensor data from a file and samples `num_points` from the data.
            The sensor data is then returned as a tuple containing the sensor points and the exact solution values.
        """
        # use pandas to read the file
        df = pd.read_csv(file_name)

        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        exact_sol = df.iloc[:, 2].values

        # now sample num_points from the data
        indices = np.random.randint(0, x.shape[0], num_points)

        x = x[indices]
        y = y[indices]
        exact_sol = exact_sol[indices]

        # stack them together
        points = np.stack((x, y), axis=1)

        return points, exact_sol

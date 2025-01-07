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


# Author : Thivin Anandh. D
# URL: https://thivinanandh.github.io
# Added test cases for validating Quadrature routines by computing the areas.
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf

from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fe.fe2d_setup_main import FE2DSetupMain

import pytest


def test_invalid_fe_type():
    """
    Test case to validate the behavior when an invalid finite element type is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "quadrilateral", 2, "invalid_fe_type", 1, "quad_type"
        )
        fe2d_setup_main.assign_basis_function()


@pytest.mark.parametrize(
    "fe_type", ["legendre", "jacobi", "legendre_special", "chebyshev_2", "jacobi_plain"]
)
def test_valid_fe_type(fe_type):
    """
    Test case to validate the behavior when a valid finite element type is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain("quadrilateral", 2, fe_type, 4, "quad_type")
    assert fe2d_setup_main.assign_basis_function() is not None


def test_invalid_cell_type():
    """
    Test case to validate the behavior when an invalid cell type is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "invalid_cell_type", 2, "legendre", 4, "gauss_jacobi"
        )
        fe2d_setup_main.assign_quadrature_rules()


def test_valid_cell_type():
    """
    Test case to validate the behavior when a valid cell type is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain("quadrilateral", 2, "legendre", 4, "gauss-jacobi")
    assert fe2d_setup_main.assign_quadrature_rules() is not None


@pytest.mark.parametrize("quad_type", ["gauss-legendre", "gauss-jacobi"])
def test_valid_quad_type(quad_type):
    """
    Test case to validate the behavior when a valid quadrature type is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain("quadrilateral", 2, "legendre", 3, quad_type)
    assert fe2d_setup_main.assign_quadrature_rules() is not None


def test_invalid_quad_type():
    """
    Test case to validate the behavior when an invalid quadrature type is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "quadrilateral", 3, "legendre", 4, "invalid_quad_type"
        )
        fe2d_setup_main.assign_quadrature_rules()


@pytest.mark.parametrize("quad_order", [0, 1, 2, 1e5])
def test_invalid_quad_order(quad_order):
    """
    Test case to validate the behavior when an invalid quadrature order is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "quadrilateral", 4, "legendre", quad_order, "gauss-jacobi"
        )
        fe2d_setup_main.assign_quadrature_rules()


@pytest.mark.parametrize("quad_order", [3, 4, 5])
def test_valid_quad_order(quad_order):
    """
    Test case to validate the behavior when a valid quadrature order is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain(
        "quadrilateral", 3, "legendre", quad_order, "gauss-jacobi"
    )
    assert fe2d_setup_main.assign_quadrature_rules() is not None


@pytest.mark.parametrize("fe_order", [2, 7, 20])
def test_valid_fe_order(fe_order):
    """
    Test case to validate the behavior when a valid finite element order is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain(
        "quadrilateral", fe_order, "legendre", 1, "gauss-jacobi"
    )
    assert fe2d_setup_main.assign_basis_function() is not None


@pytest.mark.parametrize("fe_order", [0, 1, 1001])
def test_invalid_fe_order(fe_order):
    """
    Test case to validate the behavior when an invalid finite element order is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "quadrilateral", fe_order, "legendre", 2, "gauss-jacobi"
        )
        fe2d_setup_main.assign_basis_function()


@pytest.mark.parametrize("transformation_type", ["affine", "bilinear"])
def test_valid_transformation_type(transformation_type):
    """
    Test case to validate the behavior when a valid transformation type is provided.
    It should return a non-None value.
    """
    fe2d_setup_main = FE2DSetupMain("quadrilateral", 3, "legendre", 3, "quad_type")
    assert (
        fe2d_setup_main.assign_fe_transformation(
            transformation_type, [[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        is not None
    )


def test_invalid_transformation_type():
    """
    Test case to validate the behavior when an invalid transformation type is provided.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fe2d_setup_main = FE2DSetupMain(
            "quadrilateral", 3, "legendre", 3, "gauss-jacobi"
        )
        fe2d_setup_main.assign_fe_transformation(
            "invalid_transformation", [[0, 0], [1, 0], [1, 1], [0, 1]]
        )

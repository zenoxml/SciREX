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
Poisson Equation Configuration

Configuration for training neural operators on the Poisson equation:
    -∇²u = f  (in 2D: -∂²u/∂x² - ∂²u/∂y² = f)

With Dirichlet boundary conditions: u = 0 on boundary.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal

from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO2DConfig, FNO3DConfig


@dataclass
class PoissonDataConfig:
    """Configuration for Poisson equation data.
    
    Attributes:
        data_path: Path to data directory.
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Training batch size.
        nx: Resolution in x direction.
        ny: Resolution in y direction.
        domain: Domain bounds ((x_min, x_max), (y_min, y_max)).
        source_type: Type of source term generation.
        n_source_features: Number of features for source term generation.
        boundary_type: Type of boundary conditions.
        normalize: Whether to normalize data.
        include_mesh: Whether to include mesh coordinates as input.
    """
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 50
    nx: int = 64
    ny: int = 64
    domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))
    source_type: Literal["gaussian", "random_fourier", "single_peak"] = "gaussian"
    n_source_features: int = 3  # Number of Gaussian peaks, etc.
    boundary_type: Literal["dirichlet", "neumann", "periodic"] = "dirichlet"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Poisson3DDataConfig:
    """Configuration for 3D Poisson equation data."""
    data_path: Optional[str] = None
    n_train: int = 500
    n_test: int = 100
    batch_size: int = 8
    nx: int = 32
    ny: int = 32
    nz: int = 32
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    source_type: Literal["gaussian", "random_fourier"] = "gaussian"
    n_source_features: int = 3
    boundary_type: Literal["dirichlet", "periodic"] = "dirichlet"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Poisson2DConfig:
    """Complete configuration for 2D Poisson equation.
    
    Problem: -∇²u = f with u = 0 on boundary
    Input: Source term f(x, y) + mesh coordinates
    Output: Solution u(x, y)
    
    Example usage:
        config = Poisson2DConfig()
        model = FNO2D(**config.model.__dict__)
    """
    name: str = "poisson_2d"
    description: str = "2D Poisson equation: -∇²u = f with Dirichlet BCs"
    
    # Model configuration
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3,      # Source term + mesh (x, y)
        out_channels=1,     # Solution u(x, y)
        hidden_channels=64,
        n_layers=4,
        n_modes=(12, 12),
        activation="gelu",
        use_channel_mlp=True
    ))
    
    # Data configuration
    data: PoissonDataConfig = field(default_factory=PoissonDataConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=50,
        batch_size=50,
        gradient_clip=None,
        early_stopping=True,
        early_stopping_patience=20,
        scheduler=SchedulerConfig(
            scheduler_type="StepLR",
            step_size=20,
            gamma=0.5
        ),
        loss=LossConfig(
            training_loss="mse",
            testing_loss="mse"
        )
    ))


@dataclass
class PoissonMultiscaleConfig:
    """Configuration for multiscale Poisson problem.
    
    This configuration is designed for problems with
    fine-scale variations in the source term.
    """
    name: str = "poisson_multiscale"
    description: str = "Multiscale Poisson equation with fine features"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3,
        out_channels=1,
        hidden_channels=128,
        n_layers=6,
        n_modes=(24, 24),  # More modes for multiscale
        activation="gelu",
        use_channel_mlp=True,
        channel_mlp_expansion=1.0
    ))
    
    data: PoissonDataConfig = field(default_factory=lambda: PoissonDataConfig(
        nx=128,
        ny=128,
        source_type="random_fourier",
        n_source_features=10
    ))
    
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=5e-4,
        n_epochs=200,
        batch_size=32,
        scheduler=SchedulerConfig(
            scheduler_type="CosineAnnealingLR"
        )
    ))


@dataclass
class Poisson3DConfig:
    """Complete configuration for 3D Poisson equation."""
    name: str = "poisson_3d"
    description: str = "3D Poisson equation: -∇²u = f"
    
    # Model configuration
    model: FNO3DConfig = field(default_factory=lambda: FNO3DConfig(
        in_channels=4,      # Source term + mesh (x, y, z)
        out_channels=1,     # Solution u(x, y, z)
        hidden_channels=32,
        n_layers=4,
        n_modes=(8, 8, 8),
        activation="gelu",
        use_channel_mlp=True
    ))
    
    # Data configuration
    data: Poisson3DDataConfig = field(default_factory=Poisson3DDataConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=50,
        batch_size=8,
        scheduler=SchedulerConfig(
            scheduler_type="StepLR",
            step_size=20,
            gamma=0.5
        ),
        loss=LossConfig(
            training_loss="mse"
        )
    ))


# Convenience alias
PoissonConfig = Poisson2DConfig

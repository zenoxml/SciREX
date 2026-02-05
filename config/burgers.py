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
Burgers Equation Configuration

Configuration for training neural operators on the Burgers equation:
    u_t + u * u_x = nu * u_xx

Supports both 1D and 2D Burgers equation setups.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO1DConfig, FNO2DConfig


@dataclass
class BurgersDataConfig:
    """Configuration for Burgers equation data.
    
    Attributes:
        data_path: Path to data directory.
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Training batch size.
        test_batch_size: Test batch size.
        spatial_resolution: Spatial resolution (nx for 1D, (nx, ny) for 2D).
        temporal_resolution: Temporal resolution (number of time steps).
        t_final: Final time.
        viscosity: Kinematic viscosity (nu).
        domain_size: Size of spatial domain.
        normalize: Whether to normalize data.
        include_mesh: Whether to include mesh coordinates as input.
    """
    data_path: Optional[str] = None
    n_train: int = 800
    n_test: int = 200
    batch_size: int = 32
    test_batch_size: int = 32
    spatial_resolution: int = 256
    temporal_resolution: int = 101
    t_final: float = 1.0
    viscosity: float = 0.01
    domain_size: Tuple[float, float] = (0.0, 6.283185307)  # 2*pi
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Burgers1DConfig:
    """Complete configuration for 1D Burgers equation.
    
    Example usage:
        config = Burgers1DConfig()
        model = FNO1D(**config.model.__dict__)
        optimizer = get_optimizer(config.optimization)
    """
    name: str = "burgers_1d"
    description: str = "1D Burgers equation: u_t + u*u_x = nu*u_xx"
    
    # Model configuration
    model: FNO1DConfig = field(default_factory=lambda: FNO1DConfig(
        in_channels=2,      # Initial condition + mesh
        out_channels=1,     # Solution at t=T
        hidden_channels=32,
        n_layers=4,
        n_modes=(16,),
        activation="relu"
    ))
    
    # Data configuration
    data: BurgersDataConfig = field(default_factory=BurgersDataConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-4,
        n_epochs=200,
        batch_size=100,
        gradient_clip=1.0,
        early_stopping=True,
        early_stopping_patience=10,
        scheduler=SchedulerConfig(
            scheduler_type="ReduceLROnPlateau",
            patience=20,
            gamma=0.5
        ),
        loss=LossConfig(
            training_loss="mse",
            regularization="l2",
            reg_weight=1e-5
        )
    ))


@dataclass 
class Burgers2DConfig:
    """Complete configuration for 2D Burgers equation."""
    name: str = "burgers_2d"
    description: str = "2D Burgers equation"
    
    # Model configuration
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3,      # Initial condition + mesh (x, y)
        out_channels=1,     # Solution at t=T
        hidden_channels=64,
        n_layers=4,
        n_modes=(16, 16),
        activation="gelu"
    ))
    
    # Data configuration
    data: BurgersDataConfig = field(default_factory=lambda: BurgersDataConfig(
        n_train=1000,
        n_test=200,
        batch_size=16,
        spatial_resolution=64
    ))
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=500,
        batch_size=16,
        gradient_clip=1.0,
        scheduler=SchedulerConfig(
            scheduler_type="CosineAnnealingLR"
        ),
        loss=LossConfig(
            training_loss="relative_l2"
        )
    ))


# Convenience alias
BurgersConfig = Burgers1DConfig

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
Heat Equation Configuration

Configuration for training neural operators on the Heat equation:
    u_t = α * ∇²u  (thermal diffusion)

Where α is the thermal diffusivity.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO1DConfig, FNO2DConfig


@dataclass
class HeatDataConfig:
    """Configuration for Heat equation data.
    
    Attributes:
        data_path: Path to data directory.
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Training batch size.
        spatial_resolution: Spatial resolution.
        temporal_resolution: Number of time steps.
        t_final: Final time.
        diffusivity: Thermal diffusivity (α).
        domain: Spatial domain bounds.
        initial_condition_type: Type of initial condition.
        boundary_type: Type of boundary conditions.
        normalize: Whether to normalize data.
        include_mesh: Whether to include mesh coordinates.
    """
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 32
    spatial_resolution: int = 64
    temporal_resolution: int = 50
    t_final: float = 0.5
    diffusivity: float = 0.01
    domain: Tuple[float, float] = (0.0, 1.0)
    initial_condition_type: str = "gaussian"  # "gaussian", "random_fourier", "step"
    boundary_type: str = "dirichlet"  # "dirichlet", "neumann", "periodic"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Heat1DConfig:
    """Complete configuration for 1D Heat equation.
    
    Problem: u_t = α * u_xx
    Input: Initial temperature distribution + mesh
    Output: Temperature at t = T
    """
    name: str = "heat_1d"
    description: str = "1D Heat equation: u_t = α * u_xx"
    
    # Model configuration
    model: FNO1DConfig = field(default_factory=lambda: FNO1DConfig(
        in_channels=2,      # Initial condition + mesh
        out_channels=1,     # Solution at t=T
        hidden_channels=32,
        n_layers=4,
        n_modes=(16,),
        activation="gelu"
    ))
    
    # Data configuration
    data: HeatDataConfig = field(default_factory=HeatDataConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=200,
        batch_size=32,
        scheduler=SchedulerConfig(
            scheduler_type="StepLR",
            step_size=50,
            gamma=0.7
        ),
        loss=LossConfig(
            training_loss="mse"
        )
    ))


@dataclass
class Heat2DConfig:
    """Complete configuration for 2D Heat equation.
    
    Problem: u_t = α * (u_xx + u_yy)
    Input: Initial temperature distribution + mesh (x, y)
    Output: Temperature at t = T
    """
    name: str = "heat_2d"
    description: str = "2D Heat equation: u_t = α * ∇²u"
    
    # Model configuration
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3,      # Initial condition + mesh (x, y)
        out_channels=1,     # Solution at t=T
        hidden_channels=64,
        n_layers=4,
        n_modes=(12, 12),
        activation="gelu"
    ))
    
    # Data configuration
    data: HeatDataConfig = field(default_factory=lambda: HeatDataConfig(
        spatial_resolution=64,
        temporal_resolution=100,
        batch_size=20
    ))
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=20,
        scheduler=SchedulerConfig(
            scheduler_type="CosineAnnealingLR"
        ),
        loss=LossConfig(
            training_loss="l2"
        )
    ))


@dataclass
class ThermalConductionConfig:
    """Configuration for thermal conduction with variable conductivity.
    
    Problem: ρc_p * u_t = ∇·(k(x)∇u) + f
    This is a more general heat equation with spatially varying
    thermal conductivity k(x).
    """
    name: str = "thermal_conduction"
    description: str = "Heat equation with variable thermal conductivity"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=4,      # Initial + conductivity field + mesh (x, y)
        out_channels=1,
        hidden_channels=64,
        n_layers=5,
        n_modes=(16, 16),
        activation="gelu",
        use_channel_mlp=True
    ))
    
    data: HeatDataConfig = field(default_factory=lambda: HeatDataConfig(
        n_train=2000,
        spatial_resolution=64,
        batch_size=16
    ))
    
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=5e-4,
        n_epochs=300,
        scheduler=SchedulerConfig(
            scheduler_type="ReduceLROnPlateau",
            patience=30
        )
    ))


# Convenience alias
HeatConfig = Heat1DConfig

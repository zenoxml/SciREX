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
Wave Equation Configuration

Configuration for training neural operators on the Wave equation:
    u_tt = c² * ∇²u  (wave propagation)

Where c is the wave speed.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO1DConfig, FNO2DConfig


@dataclass
class WaveDataConfig:
    """Configuration for Wave equation data.
    
    Attributes:
        data_path: Path to data directory.
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Training batch size.
        spatial_resolution: Spatial resolution.
        temporal_resolution: Number of time steps.
        t_final: Final time.
        wave_speed: Wave propagation speed (c).
        domain: Spatial domain bounds.
        initial_condition_type: Type of initial displacement.
        initial_velocity_type: Type of initial velocity.
        boundary_type: Type of boundary conditions.
        normalize: Whether to normalize data.
        include_mesh: Whether to include mesh coordinates.
    """
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 32
    spatial_resolution: int = 128
    temporal_resolution: int = 100
    t_final: float = 1.0
    wave_speed: float = 1.0
    domain: Tuple[float, float] = (0.0, 1.0)
    initial_condition_type: str = "gaussian"  # "gaussian", "plucked", "random"
    initial_velocity_type: str = "zero"  # "zero", "random"
    boundary_type: str = "dirichlet"  # "dirichlet", "neumann", "absorbing"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Wave1DConfig:
    """Complete configuration for 1D Wave equation.
    
    Problem: u_tt = c² * u_xx
    Input: Initial displacement + initial velocity + mesh
    Output: Displacement at t = T
    """
    name: str = "wave_1d"
    description: str = "1D Wave equation: u_tt = c² * u_xx"
    
    # Model configuration
    model: FNO1DConfig = field(default_factory=lambda: FNO1DConfig(
        in_channels=3,      # Initial displacement + velocity + mesh
        out_channels=1,     # Solution at t=T
        hidden_channels=32,
        n_layers=4,
        n_modes=(16,),
        activation="gelu"
    ))
    
    # Data configuration
    data: WaveDataConfig = field(default_factory=WaveDataConfig)
    
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
class Wave2DConfig:
    """Complete configuration for 2D Wave equation.
    
    Problem: u_tt = c² * (u_xx + u_yy)
    Input: Initial displacement + initial velocity + mesh (x, y)
    Output: Displacement at t = T
    """
    name: str = "wave_2d"
    description: str = "2D Wave equation: u_tt = c² * ∇²u"
    
    # Model configuration
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=4,      # Initial displacement + velocity + mesh (x, y)
        out_channels=1,     # Solution at t=T
        hidden_channels=64,
        n_layers=4,
        n_modes=(16, 16),
        activation="gelu"
    ))
    
    # Data configuration
    data: WaveDataConfig = field(default_factory=lambda: WaveDataConfig(
        spatial_resolution=64,
        temporal_resolution=100,
        batch_size=20
    ))
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=150,
        batch_size=20,
        scheduler=SchedulerConfig(
            scheduler_type="CosineAnnealingLR"
        ),
        loss=LossConfig(
            training_loss="l2"
        )
    ))


@dataclass
class AcousticWaveConfig:
    """Configuration for acoustic wave propagation.
    
    Problem: p_tt = c²(x) * ∇²p + f
    This models acoustic wave propagation with spatially varying
    sound speed c(x) and source term f.
    """
    name: str = "acoustic_wave"
    description: str = "Acoustic wave equation with variable sound speed"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=5,      # Initial + velocity + source + speed field + mesh
        out_channels=1,     # Pressure field
        hidden_channels=64,
        n_layers=5,
        n_modes=(20, 20),
        activation="gelu",
        use_channel_mlp=True
    ))
    
    data: WaveDataConfig = field(default_factory=lambda: WaveDataConfig(
        n_train=2000,
        spatial_resolution=128,
        temporal_resolution=200,
        batch_size=16
    ))
    
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=5e-4,
        n_epochs=300,
        gradient_clip=1.0,
        scheduler=SchedulerConfig(
            scheduler_type="ReduceLROnPlateau",
            patience=30
        )
    ))


# Convenience alias
WaveConfig = Wave1DConfig

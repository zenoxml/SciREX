# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited.
# Licensed under the Apache License, Version 2.0.

"""Navier-Stokes Equation Configuration."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal
from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO2DConfig, FNO3DConfig


@dataclass
class NavierStokesDataConfig:
    """Configuration for Navier-Stokes equation data."""
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 10
    nx: int = 64
    ny: int = 64
    nt: int = 20  # Number of time steps
    t_final: float = 1.0
    viscosity: float = 1e-3
    reynolds_number: Optional[float] = None
    flow_type: str = "decaying"  # "decaying", "forced", "channel"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class NavierStokes3DDataConfig:
    """Configuration for 3D Navier-Stokes equation data.
    
    Attributes:
        data_path: Path to data directory (optional).
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Training batch size.
        nx, ny, nz: Spatial resolution in each direction.
        nt: Number of time steps to store as input.
        t_final: Final simulation time.
        viscosity: Kinematic viscosity (ν).
        normalize: Whether to normalize data.
        include_mesh: Whether to include mesh coordinates as input.
    """
    data_path: Optional[str] = None
    n_train: int = 200
    n_test: int = 50
    batch_size: int = 4
    nx: int = 32
    ny: int = 32
    nz: int = 32
    nt: int = 5  # Number of time snapshots as input
    t_final: float = 1.0
    viscosity: float = 1e-3
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class NavierStokes2DConfig:
    """Configuration for 2D Navier-Stokes equations.
    
    Problem: ∂ω/∂t + u·∇ω = ν∇²ω + f (vorticity formulation)
    Input: Initial vorticity field + mesh coordinates
    Output: Vorticity field at t = T
    """
    name: str = "navier_stokes_2d"
    description: str = "2D Navier-Stokes in vorticity formulation"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=11,  # Initial vorticity + 10 time steps,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        n_modes=(12, 12),
        activation="gelu",
        use_channel_mlp=True
    ))
    data: NavierStokesDataConfig = field(default_factory=NavierStokesDataConfig)
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3, n_epochs=500, batch_size=10, gradient_clip=1.0,
        scheduler=SchedulerConfig(scheduler_type="StepLR", step_size=100, gamma=0.5),
        loss=LossConfig(training_loss="relative_l2")
    ))


@dataclass
class TurbulentFlowConfig:
    """Configuration for turbulent flow at higher Reynolds numbers."""
    name: str = "turbulent_flow"
    description: str = "Turbulent 2D Navier-Stokes"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=11, out_channels=1, hidden_channels=128,
        n_layers=6, n_modes=(20, 20), activation="gelu", use_channel_mlp=True
    ))
    data: NavierStokesDataConfig = field(default_factory=lambda: NavierStokesDataConfig(
        viscosity=1e-4, nx=128, ny=128, batch_size=4
    ))
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=5e-4, n_epochs=1000, gradient_clip=0.5
    ))


@dataclass
class NavierStokes3DConfig:
    """Complete configuration for 3D Navier-Stokes equations.
    
    Problem: ∂ω/∂t + (u·∇)ω - (ω·∇)u = ν∇²ω (3D vorticity formulation)
    Input: Vorticity magnitude snapshots + mesh coordinates
    Output: Vorticity magnitude at t = T
    
    Note: 3D Navier-Stokes includes vortex stretching term (ω·∇)u which
    makes the dynamics significantly more complex than 2D.
    """
    name: str = "navier_stokes_3d"
    description: str = "3D Navier-Stokes in vorticity formulation"
    
    # Model configuration: FNO3D for volumetric data
    model: FNO3DConfig = field(default_factory=lambda: FNO3DConfig(
        in_channels=8,      # nt=5 time snapshots + 3 mesh coordinates
        out_channels=1,     # Vorticity magnitude at final time
        hidden_channels=32,
        n_layers=4,
        n_modes=(8, 8, 8),
        activation="gelu",
        use_channel_mlp=True
    ))
    
    # Data configuration
    data: NavierStokes3DDataConfig = field(default_factory=NavierStokes3DDataConfig)
    
    # Optimization configuration
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=4,
        gradient_clip=1.0,
        early_stopping=True,
        early_stopping_patience=20,
        scheduler=SchedulerConfig(
            scheduler_type="StepLR",
            step_size=30,
            gamma=0.5
        ),
        loss=LossConfig(
            training_loss="mse",
            testing_loss="mse"
        )
    ))


NavierStokesConfig = NavierStokes2DConfig

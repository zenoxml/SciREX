# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited.
# Licensed under the Apache License, Version 2.0.

"""Navier-Stokes Equation Configuration."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal
from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO2DConfig


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


NavierStokesConfig = NavierStokes2DConfig

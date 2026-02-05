# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited.
# Licensed under the Apache License, Version 2.0.

"""Advection Equation Configuration."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal
from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO1DConfig, FNO2DConfig


@dataclass
class AdvectionDataConfig:
    """Configuration for Advection equation data."""
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 32
    spatial_resolution: int = 128
    temporal_resolution: int = 100
    t_final: float = 1.0
    advection_speed: float = 1.0
    advection_type: Literal["linear", "nonlinear"] = "linear"
    domain: Tuple[float, float] = (0.0, 6.283185)
    initial_condition_type: str = "sine"
    boundary_type: str = "periodic"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class Advection1DConfig:
    """Configuration for 1D linear Advection: u_t + c*u_x = 0."""
    name: str = "advection_1d"
    description: str = "1D Linear advection equation"
    
    model: FNO1DConfig = field(default_factory=lambda: FNO1DConfig(
        in_channels=2, out_channels=1, hidden_channels=32,
        n_layers=4, n_modes=(16,), activation="gelu"
    ))
    data: AdvectionDataConfig = field(default_factory=AdvectionDataConfig)
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3, n_epochs=150, batch_size=32
    ))


@dataclass
class Advection2DConfig:
    """Configuration for 2D Advection: u_t + a·∇u = 0."""
    name: str = "advection_2d"
    description: str = "2D Linear advection equation"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3, out_channels=1, hidden_channels=64,
        n_layers=4, n_modes=(12, 12), activation="gelu"
    ))
    data: AdvectionDataConfig = field(default_factory=lambda: AdvectionDataConfig(
        spatial_resolution=64, batch_size=20
    ))
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3, n_epochs=100, batch_size=20
    ))


AdvectionConfig = Advection1DConfig

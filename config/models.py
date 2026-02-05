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
Model Configuration Module

This module contains configuration classes for various neural operator models
including FNO, TFNO, WNO, and their variants.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Literal, Union


@dataclass
class ModelConfig:
    """Base configuration for all neural operator models.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of hidden channels.
        n_layers: Number of neural operator blocks.
        activation: Activation function to use.
    """
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 32
    n_layers: int = 4
    activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu"


@dataclass
class FNOConfig(ModelConfig):
    """Configuration for Fourier Neural Operator (FNO).
    
    Attributes:
        n_modes: Number of Fourier modes to keep per dimension.
        lifting_channel_ratio: Ratio for lifting layer hidden channels.
        projection_channel_ratio: Ratio for projection layer hidden channels.
        domain_padding: Padding fraction for domain padding.
        domain_padding_mode: Mode for domain padding.
        norm: Normalization type.
        skip_connection: Type of skip connection.
        use_channel_mlp: Whether to use channel MLP after each block.
        channel_mlp_expansion: Expansion ratio for channel MLP.
        factorization: Tensor factorization for spectral convolution weights.
        rank: Rank for factorization (fraction of original).
        implementation: Implementation strategy for spectral convolution.
        separable: Whether to use separable convolution.
        preactivation: Whether to use preactivation.
        complex_data: Whether input is complex-valued.
    """
    n_modes: Tuple[int, ...] = (16, 16)
    lifting_channel_ratio: float = 2.0
    projection_channel_ratio: float = 2.0
    domain_padding: Optional[float] = None
    domain_padding_mode: Literal["one-sided", "symmetric"] = "one-sided"
    norm: Optional[Literal["group_norm", "instance_norm", "layer_norm", "none"]] = None
    skip_connection: Literal["linear", "identity", "soft-gating"] = "soft-gating"
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 0.5
    factorization: Optional[str] = None
    rank: float = 1.0
    implementation: Literal["factorized", "reconstructed", "contractible"] = "factorized"
    separable: bool = False
    preactivation: bool = False
    complex_data: bool = False


@dataclass
class FNO1DConfig(FNOConfig):
    """Configuration for 1D FNO."""
    n_modes: Tuple[int] = (16,)
    hidden_channels: int = 32
    n_layers: int = 4


@dataclass
class FNO2DConfig(FNOConfig):
    """Configuration for 2D FNO."""
    n_modes: Tuple[int, int] = (16, 16)
    hidden_channels: int = 64
    n_layers: int = 4


@dataclass
class FNO3DConfig(FNOConfig):
    """Configuration for 3D FNO."""
    n_modes: Tuple[int, int, int] = (8, 8, 8)
    hidden_channels: int = 32
    n_layers: int = 4


@dataclass
class SimpleFNOConfig(FNOConfig):
    """Configuration for SimpleFNO (simplified implementation).
    
    This is a streamlined FNO configuration with commonly used defaults
    for quick experimentation.
    """
    hidden_channels: int = 32
    n_layers: int = 4
    use_channel_mlp: bool = False
    norm: Optional[str] = None


@dataclass
class TFNOConfig(FNOConfig):
    """Configuration for Tucker-factorized FNO (TFNO).
    
    TFNO uses Tucker decomposition for efficient weight representation.
    """
    factorization: str = "Tucker"
    rank: float = 0.1
    

@dataclass
class WNO2DConfig(ModelConfig):
    """Configuration for 2D Wavelet Neural Operator (WNO).
    
    Attributes:
        size: Spatial size of the input (nx, ny).
        level: Wavelet decomposition level.
        wavelet: Type of wavelet to use.
        padding: Amount of padding to apply.
        width: Width of the network (hidden channels).
    """
    size: Tuple[int, int] = (64, 64)
    level: int = 3
    wavelet: str = "db6"
    padding: int = 2
    hidden_channels: int = 32  # Called 'width' in the model
    n_layers: int = 4


# Preset model configurations for different scales
@dataclass
class FNO_Tiny2D(FNO2DConfig):
    """Tiny 2D FNO for quick testing."""
    n_modes: Tuple[int, int] = (8, 8)
    hidden_channels: int = 16
    n_layers: int = 2


@dataclass
class FNO_Small2D(FNO2DConfig):
    """Small 2D FNO for moderate problems."""
    n_modes: Tuple[int, int] = (12, 12)
    hidden_channels: int = 32
    n_layers: int = 4


@dataclass
class FNO_Medium2D(FNO2DConfig):
    """Medium 2D FNO for standard problems."""
    n_modes: Tuple[int, int] = (16, 16)
    hidden_channels: int = 64
    n_layers: int = 4


@dataclass
class FNO_Large2D(FNO2DConfig):
    """Large 2D FNO for complex problems."""
    n_modes: Tuple[int, int] = (24, 24)
    hidden_channels: int = 128
    n_layers: int = 6


@dataclass
class FNO_Tiny1D(FNO1DConfig):
    """Tiny 1D FNO for quick testing."""
    n_modes: Tuple[int] = (8,)
    hidden_channels: int = 16
    n_layers: int = 2


@dataclass
class FNO_Small1D(FNO1DConfig):
    """Small 1D FNO for moderate problems."""
    n_modes: Tuple[int] = (16,)
    hidden_channels: int = 32
    n_layers: int = 4


@dataclass
class WNO_Small2D(WNO2DConfig):
    """Small 2D WNO for moderate problems."""
    level: int = 3
    hidden_channels: int = 32
    n_layers: int = 4


@dataclass
class WNO_Medium2D(WNO2DConfig):
    """Medium 2D WNO for standard problems."""
    level: int = 4
    hidden_channels: int = 48
    n_layers: int = 4

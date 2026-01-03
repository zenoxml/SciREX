# Copyright (c) 2025 Zenteiq Aitech Innovations Private Limited and
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

# Author: Diya
# Version Info: 29/Dec/2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_conv_sr import SpectralConv1d

class FNOBlock1d(nn.Module):
    """
    Args:
        width: Number of channels in the input and output tensor
        modes: Number of Fourier modes to use in the spectral convolution   

    Returns:
        Output tensor of shape (batch, width, nx)
    """ 
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1)
        self.norm = nn.BatchNorm1d(width)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, width, nx)
        Returns:
            Output tensor of shape (batch, width, nx)
        """
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = self.norm(x1 + x2)
        x = F.gelu(x)
        return x
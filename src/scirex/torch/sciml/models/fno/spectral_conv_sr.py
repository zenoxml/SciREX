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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to use
        
        Returns:
            Output tensor of shape (batch, out_channels, nx)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize with smaller scale for stability
        self.scale = (1 / (in_channels * out_channels)) ** 0.5
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, nx)
        Returns:
            Output tensor of shape (batch, channels, nx)
        """
        batch = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1), 
                            device=x.device, dtype=torch.cfloat)
        
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :modes],
            self.weights[:, :, :modes]
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm='ortho')
        return x
import torch
import torch.nn as nn
import numpy as np
from pytorch_wavelets import DWTForward as DWT, DWTInverse as IDWT, DTCWTForward, DTCWTInverse

""" Def: 2d Wavelet convolutional layer (discrete) """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        Note: This is the 'Full Spectrum' version that applies weights to all decomposition levels.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet       
        self.mode = mode
        self.size = size
        
        self.dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        self.idwt = IDWT(mode=self.mode, wave=self.wavelet)

        # Weight Initialization using Xavier-like normalization for stability
        self.scale = (1 / (in_channels * out_channels))**0.5
        
        # Calculate mode sizes by running a dummy pass
        dummy_data = torch.zeros(1, in_channels, *size)
        yl, yh = self.dwt(dummy_data)
        
        # Low-pass (Approximate) weights
        self.weight_l = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, yl.shape[-2], yl.shape[-1]))
        
        # High-pass (Detail) weights for all levels
        # Each level has 3 subbands: Horizontal, Vertical, Diagonal
        self.weights_h = nn.ParameterList()
        for j in range(self.level):
            # Shape for each level's details: (3, in, out, H_j, W_j)
            h_j, w_j = yh[j].shape[-2], yh[j].shape[-1]
            self.weights_h.append(nn.Parameter(self.scale * torch.randn(3, in_channels, out_channels, h_j, w_j)))

    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, _, h, w = x.shape
        x_ft, x_coeff = self.dwt(x)

        # 1. Transform Low-pass subband
        out_ft = self.mul2d(x_ft, self.weight_l)
        
        # 2. Transform all High-pass subbands (Detail levels)
        out_h = []
        for j in range(self.level):
            # x_coeff[j] shape: (B, C, 3, H_j, W_j)
            # We treat the subband index as another dimension to multiply
            # We can use einsum to handle the 3 subbands at once
            # weights_h[j] shape: (3, I, O, H_j, W_j)
            # input shape: (B, I, 3, H_j, W_j)
            # We want output: (B, O, 3, H_j, W_j)
            
            # Using loop for clarity as in original
            h_sub = self.mul2d(x_coeff[j][:, :, 0].clone(), self.weights_h[j][0])
            v_sub = self.mul2d(x_coeff[j][:, :, 1].clone(), self.weights_h[j][1])
            d_sub = self.mul2d(x_coeff[j][:, :, 2].clone(), self.weights_h[j][2])
            
            out_level = torch.stack([h_sub, v_sub, d_sub], dim=2)
            out_h.append(out_level)
        
        # Reconstruct
        x_idwt = self.idwt((out_ft, out_h))
        
        # Ensure size match
        if x_idwt.shape[-2] != h or x_idwt.shape[-1] != w:
            x_idwt = x_idwt[:, :, :h, :w]
            
        return x_idwt

    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2):
        super(WaveConv2dCwt, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2        
        self.size = size
        
        self.dwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        self.idwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2)

        self.scale = (1 / (in_channels * out_channels))**0.5
        
        # Dummy pass for sizes
        dummy_data = torch.zeros(1, in_channels, *size)
        yl, yh = self.dwt(dummy_data)

        # Low-pass weights
        self.weight_l = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, yl.shape[-2], yl.shape[-1]))
        
        # High-pass weights for all levels (6 directional subbands)
        self.weights_h_real = nn.ParameterList()
        self.weights_h_imag = nn.ParameterList()
        
        for j in range(self.level):
            h_j, w_j = yh[j].shape[-3], yh[j].shape[-2]
            self.weights_h_real.append(nn.Parameter(self.scale * torch.randn(6, in_channels, out_channels, h_j, w_j)))
            self.weights_h_imag.append(nn.Parameter(self.scale * torch.randn(6, in_channels, out_channels, h_j, w_j)))

    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        _, _, h, w = x.shape
        yl, yh = self.dwt(x)
        
        # 1. Low-pass
        out_l = self.mul2d(yl, self.weight_l)
        
        # 2. High-pass (Complex detail)
        out_h = []
        for j in range(self.level):
            level_coeffs = []
            for s in range(6):
                re = self.mul2d(yh[j][:, :, s, :, :, 0].clone(), self.weights_h_real[j][s])
                im = self.mul2d(yh[j][:, :, s, :, :, 1].clone(), self.weights_h_imag[j][s])
                level_coeffs.append(torch.stack([re, im], dim=-1))
            out_level = torch.stack(level_coeffs, dim=2)
            out_h.append(out_level)
            
        x_idwt = self.idwt((out_l, out_h))
        
        if x_idwt.shape[-2] != h or x_idwt.shape[-1] != w:
            x_idwt = x_idwt[:, :, :h, :w]
            
        return x_idwt

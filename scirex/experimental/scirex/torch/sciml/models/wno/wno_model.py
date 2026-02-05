import torch
import torch.nn as nn
import torch.nn.functional as F
from scirex.torch.sciml.models.wno.layer.wno_layer import WaveConv2d

class WNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, size, level=3, n_layers=4, padding=2):
        super(WNO2d, self).__init__()
        self.width = width
        self.size = size
        self.level = level
        self.n_layers = n_layers
        self.padding = padding
        self.in_channels = in_channels

        # Lifting Layer
        self.fc0 = nn.Linear(self.in_channels, self.width)

        # Wavelet Layers
        # Note: Size for WaveConv must account for padding
        padded_size = [s + padding for s in size]
        self.conv_layers = nn.ModuleList([
            WaveConv2d(width, width, level, padded_size, wavelet='db6') 
            for _ in range(n_layers)
        ])

        # Linear projection layers (Skip connections)
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(n_layers)
        ])

        # Projection Layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        # x input shape: (B, nx, ny, 1) - just forcing term
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # (B, nx, ny, 3)
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (B, C, nx, ny)
        
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x) + self.w_layers[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
            
        x = x.permute(0, 2, 3, 1) # (B, nx, ny, C)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
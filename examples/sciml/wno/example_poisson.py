
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from scirex.torch.sciml.models.wno.wno_model import WaveConv2d
from scirex.torch.sciml.losses import LpLoss

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------------------------------------
# 1. Utilities: Normalizer and Data
# ------------------------------------------------------------------------------

class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        # Compute mean and std globally across batch and spatial dimensions
        self.mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        self.std = torch.std(x, dim=(0, 1, 2), keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

def generate_poisson_2d_data(n_samples=100, nx=64, ny=64):
    """
    Generate synthetic data for 2D Poisson equation -Delta u = f.
    Returns:
        input_f: (n_samples, nx, ny, 1) - forcing term
        output_u: (n_samples, nx, ny, 1) - solution
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    f_list = []
    u_list = []
    
    for _ in range(n_samples):
        # Random coefficients for a few modes
        m_max, n_max = 4, 4
        f_sample = np.zeros((nx, ny))
        u_sample = np.zeros((nx, ny))
        
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                a_mn = np.random.uniform(-1, 1)
                
                # f_mn = sin(m*pi*x) * sin(n*pi*y)
                mode = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
                f_sample += a_mn * mode
                
                # u_mn = a_mn / ((m^2 + n^2) * pi^2) * mode
                u_sample += a_mn * mode / ((m**2 + n**2) * np.pi**2)
                
        f_list.append(f_sample)
        u_list.append(u_sample)
        
    input_f = np.array(f_list)[:, :, :, np.newaxis].astype(np.float32)
    output_u = np.array(u_list)[:, :, :, np.newaxis].astype(np.float32)
    
    return torch.from_numpy(input_f), torch.from_numpy(output_u)

# ------------------------------------------------------------------------------
# 2. 2D WNO Model
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 3. Main Script
# ------------------------------------------------------------------------------

def main():
    # Parameters
    ntrain = 1000
    ntest = 100
    nx, ny = 64, 64
    batch_size = 20
    epochs = 100
    learning_rate = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check outputs dir
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_2d")
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    print("Generating 2D Poisson data...")
    input_f, output_u = generate_poisson_2d_data(ntrain + ntest, nx, ny)
    
    # Normalization (Train only statistics)
    x_normalizer = UnitGaussianNormalizer(input_f[:ntrain])
    y_normalizer = UnitGaussianNormalizer(output_u[:ntrain])
    
    input_f_norm = x_normalizer.encode(input_f)
    output_u_norm = y_normalizer.encode(output_u)
    
    # Move to device and loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f_norm[:ntrain], output_u_norm[:ntrain]),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f_norm[ntrain:], output_u_norm[ntrain:]),
        batch_size=batch_size, shuffle=False
    )
    
    # Model (in_channels=3: f + x + y)
    model = WNO2d(in_channels=3, out_channels=1, width=64, size=[nx, ny], level=3, n_layers=4, padding=2).to(device)
    
    # Optimizer with Weight Decay (as in original)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Loss: MSELoss (as requested)
    criterion = nn.MSELoss() 
    
    # Training Loop
    train_losses = []
    test_losses = []
    
    # Normalize statistics to device for inference un-normalize
    y_normalizer.to(device)
    
    print("Starting training with MSELoss and Weight Decay...")
    t0 = default_timer()
    
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            # Loss is calculated in normalized space usually, or un-normalized
            # Following instruction: use MSE loss ONLY for training
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        scheduler.step()
        ep_loss /= len(train_loader)
        train_losses.append(ep_loss)
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += criterion(out, y).item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}/{epochs}, Train MSE: {ep_loss:.5f}, Test MSE: {test_loss:.5f}")
            
    t1 = default_timer()
    print(f"Training completed in {t1-t0:.2f} seconds")
    
    # 4. Evaluation Metrics Calculation
    print("\nEvaluating final metrics...")
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    y_normalizer.to(device)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            y_pred = y_normalizer.decode(out)
            y_true = y_normalizer.decode(y)
            
            all_y_true.append(y_true.cpu())
            all_y_pred.append(y_pred.cpu())
            
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    # Calculate metrics on un-normalized data
    l2_diff = np.linalg.norm(all_y_true - all_y_pred)
    l2_true = np.linalg.norm(all_y_true)
    rel_l2_error = l2_diff / l2_true
    
    ss_res = np.sum((all_y_true - all_y_pred)**2)
    ss_tot = np.sum((all_y_true - np.mean(all_y_true))**2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f"Final Metrics:")
    print(f"  Relative L2 Error: {rel_l2_error:.6f}")
    print(f"  R2 Score:          {r2_score:.6f}")

    # 5. Save plots
    plt.figure()
    plt.plot(train_losses, label='Train MSE')
    plt.plot(test_losses, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Training History (Final Test MSE: {test_losses[-1]:.4f})')
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    sample_idx = 0
    # Shape is (100, 64, 64, 1), take the first sample's only channel
    y_true_sample = all_y_true[sample_idx, :, :, 0]
    y_pred_sample = all_y_pred[sample_idx, :, :, 0]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axs[0].imshow(y_true_sample, cmap='jet')
    axs[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(y_pred_sample, cmap='jet')
    axs[1].set_title('WNO Prediction')
    plt.colorbar(im1, ax=axs[1])
    
    im2 = axs[2].imshow(np.abs(y_true_sample - y_pred_sample), cmap='jet')
    axs[2].set_title(f'Absolute Error (RelL2: {rel_l2_error:.4f})')
    plt.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_results.png'))
    plt.close()

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

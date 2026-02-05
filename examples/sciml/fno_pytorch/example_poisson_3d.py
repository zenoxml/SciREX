"""
Example: 3D Poisson Equation with FNO

This script demonstrates the full workflow for the 3D Poisson equation:
1.  **Data Generation**: Generates synthetic pairs of Source Term f(x,y,z) and Solution u(x,y,z).
2.  **Model Training**: Trains a 3D FNO to map f -> u.
3.  **Visualization**: Shows slices of the 3D field to verify performance.

Problem:
    -∇²u = f  in [0, 1]³
    u = 0     on boundary (Dirichlet)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from config.poisson import Poisson3DConfig
from scirex.torch.sciml.models.fno import FNO
from scirex.torch.sciml.utils.unit_gaussian_normalizer import UnitGaussianNormalizer
from data_generation.poisson import generate_poisson_3d_data

def main():
    """Main function for 3D Poisson FNO Example."""
    
    # Load configuration
    cfg = Poisson3DConfig()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output setup
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_3d")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # Data Generation (with Caching)
    # ========================================
    print("\n" + "="*50)
    print("Data Generation")
    print("="*50)
    
    data_file = os.path.join(output_dir, "poisson_3d_data.pt")
    
    if os.path.exists(data_file):
        print(f"Loading cached data from {data_file}")
        data_dict = torch.load(data_file, map_location=device)
        input_data = data_dict["input"]
        output_data = data_dict["output"]
    else:
        print("Generating 3D Poisson data...")
        total_samples = cfg.data.n_train + cfg.data.n_test
        input_data, output_data = generate_poisson_3d_data(
            n_samples=total_samples,
            nx=cfg.data.nx,
            ny=cfg.data.ny,
            nz=cfg.data.nz,
            n_sources=cfg.data.n_source_features,
            device=device,
            include_mesh=cfg.data.include_mesh
        )
        print(f"Saving data to {data_file}")
        torch.save({"input": input_data, "output": output_data}, data_file)
        
    ntrain = cfg.data.n_train
    ntest = cfg.data.n_test
    print(f"Data shape: Input {input_data.shape}, Output {output_data.shape}")
    
    # ========================================
    # Normalization
    # ========================================
    # Move to device before normalization to avoid mismatch
    input_data = input_data.to(device)
    output_data = output_data.to(device)
    
    x_normalizer = UnitGaussianNormalizer(input_data[:ntrain]).to(device)
    y_normalizer = UnitGaussianNormalizer(output_data[:ntrain]).to(device)
    
    input_data = x_normalizer.encode(input_data)
    output_data = y_normalizer.encode(output_data)
    
    # ========================================
    # DataLoaders
    # ========================================
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_data[:ntrain], output_data[:ntrain]),
        batch_size=cfg.optimization.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_data[ntrain:], output_data[ntrain:]),
        batch_size=cfg.optimization.batch_size, shuffle=False
    )
    
    # ========================================
    # Model Setup
    # ========================================
    print("\n" + "="*50)
    print("Model Setup: FNO3D")
    print("="*50)
    
    # Map activation
    activation_map = {"gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}
    non_linearity = activation_map.get(cfg.model.activation, F.gelu)
    
    # 3D Positional Embedding
    from scirex.torch.sciml.builders.embeddings import GridEmbeddingND
    pos_embed = GridEmbeddingND(
        in_channels=cfg.model.in_channels,
        dim=3,
        grid_boundaries=list(cfg.data.domain)
    )
    
    model = FNO(
        n_modes=cfg.model.n_modes,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        non_linearity=non_linearity,
        use_channel_mlp=cfg.model.use_channel_mlp,
        positional_embedding=pos_embed
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimization.learning_rate)
    criterion = nn.MSELoss()
    
    # ========================================
    # Training Loop
    # ========================================
    print(f"Starting training for {cfg.optimization.n_epochs} epochs...")
    t1 = default_timer()
    train_losses = []
    test_losses = []
    
    for epoch in range(1, cfg.optimization.n_epochs + 1):
        model.train()
        train_mse = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()
        
        train_mse /= len(train_loader)
        train_losses.append(train_mse)
        
        # Evaluate every epoch for the loss plot
        model.eval()
        test_mse = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_mse += criterion(out, y).item()
        test_mse /= len(test_loader)
        test_losses.append(test_mse)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")
        elif epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train MSE: {train_mse:.6f}")
            
    t2 = default_timer()
    print(f"Training finished in {t2-t1:.2f}s")
    
    # ========================================
    # Loss Plot
    # ========================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train MSE')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Training and Test Loss (MSE)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved to {output_dir}/loss_curve.png")
    
    # ========================================
    # Evaluation: Relative L2 Error
    # ========================================
    model.eval()
    test_rel_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            
            y_dec = y_normalizer.decode(y)
            out_dec = y_normalizer.decode(out)
            
            # Compute Relative L2 norm per sample
            # dims: (Batch, Channel, X, Y, Z) -> sum over 1,2,3,4
            diff_norm = torch.norm(out_dec - y_dec, p=2, dim=[1, 2, 3, 4])
            y_norm = torch.norm(y_dec, p=2, dim=[1, 2, 3, 4])
            
            rel_l2 = diff_norm / y_norm
            test_rel_l2 += rel_l2.sum().item()
            
    test_rel_l2 /= len(test_loader.dataset)
    print(f"\nFinal Test Relative L2 Error: {test_rel_l2:.6e}")
    
    # ========================================
    # Visualization
    # ========================================
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        pred = model(x)
        
        # Decode on device then move to CPU
        y_dec = y_normalizer.decode(y).cpu()
        pred_dec = y_normalizer.decode(pred).cpu()
        x_dec = x_normalizer.decode(x).cpu() # Input source f
        
        # Extract first sample
        # x is (N, 4, nx, ny, nz) -> channel 0 is force f
        f_vol = x_dec[0, 0].numpy()
        u_true = y_dec[0, 0].numpy()
        u_pred = pred_dec[0, 0].numpy()
        
        # ========================================
        # Orthogonal Slices Comparison
        # ========================================
        nx_val, ny_val, nz_val = u_true.shape
        mid_x, mid_y, mid_z = nx_val // 2, ny_val // 2, nz_val // 2
        error_vol = np.abs(u_true - u_pred)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # True Slices
        im_t0 = axes[0, 0].imshow(u_true[:, :, mid_z], cmap='RdBu_r')
        axes[0, 0].set_title(f"True (XY, z={mid_z})")
        im_t1 = axes[0, 1].imshow(u_true[:, mid_y, :], cmap='RdBu_r')
        axes[0, 1].set_title(f"True (XZ, y={mid_y})")
        im_t2 = axes[0, 2].imshow(u_true[mid_x, :, :], cmap='RdBu_r')
        axes[0, 2].set_title(f"True (YZ, x={mid_x})")
        
        # Prediction Slices
        im_p0 = axes[1, 0].imshow(u_pred[:, :, mid_z], cmap='RdBu_r')
        axes[1, 0].set_title(f"Pred (XY, z={mid_z})")
        im_p1 = axes[1, 1].imshow(u_pred[:, mid_y, :], cmap='RdBu_r')
        axes[1, 1].set_title(f"Pred (XZ, y={mid_y})")
        im_p2 = axes[1, 2].imshow(u_pred[mid_x, :, :], cmap='RdBu_r')
        axes[1, 2].set_title(f"Pred (YZ, x={mid_x})")
        
        # Absolute Error Slices
        im_e0 = axes[2, 0].imshow(error_vol[:, :, mid_z], cmap='inferno')
        axes[2, 0].set_title("Abs Error (XY)")
        im_e1 = axes[2, 1].imshow(error_vol[:, mid_y, :], cmap='inferno')
        axes[2, 1].set_title("Abs Error (XZ)")
        im_e2 = axes[2, 2].imshow(error_vol[mid_x, :, :], cmap='inferno')
        axes[2, 2].set_title("Abs Error (YZ)")
        
        # Colorbars
        for j in range(3):
            plt.colorbar(axes[0, j].images[0], ax=axes[0, j])
            plt.colorbar(axes[1, j].images[0], ax=axes[1, j])
            plt.colorbar(axes[2, j].images[0], ax=axes[2, j])
            
        plt.suptitle("3D Poisson: Orthogonal Slices (True vs Pred vs Error)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "results_orthogonal_slices.png"))
        plt.close()

        # ========================================
        # Centerline Profiles
        # ========================================
        plt.figure(figsize=(10, 6))
        z_coords = np.linspace(0, 1, nz_val)
        plt.plot(z_coords, u_true[mid_x, mid_y, :], 'k-', linewidth=2, label='True')
        plt.plot(z_coords, u_pred[mid_x, mid_y, :], 'r--', linewidth=2, label='Pred')
        plt.title(f"Centerline Profile (x={mid_x/nx_val:.2f}, y={mid_y/ny_val:.2f})")
        plt.xlabel("z")
        plt.ylabel("u(x,y,z)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "results_centerline.png"))
        plt.close()
        
        print(f"Orthogonal slices and centerline profiles saved to {output_dir}")
        
        # ========================================
        # 3D Scatter Plot (High Intensity Regions - Absolute Magnitude)
        # ========================================
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 6))
        
        # Thresholds for visualization (mean + 0.5 std dev) of ABSOLUTE value
        # This ensures both positive and negative peaks are shown
        f_val_abs = np.abs(f_vol)
        u_val_abs = np.abs(u_pred)
        
        f_thresh = f_val_abs.mean() + 0.5 * f_val_abs.std()
        u_thresh = u_val_abs.mean() + 0.5 * u_val_abs.std()
        
        # Plot Source
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = np.where(f_val_abs > f_thresh)
        vals = f_vol[x, y, z]
        img1 = ax1.scatter(x, y, z, c=vals, cmap='RdBu_r', alpha=0.3, s=2)
        ax1.set_title("Source Term (3D Scatter |val| > Mean)")
        fig.colorbar(img1, ax=ax1, shrink=0.5)
        
        # Plot Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        x, y, z = np.where(u_val_abs > u_thresh)
        vals = u_pred[x, y, z]
        img2 = ax2.scatter(x, y, z, c=vals, cmap='RdBu_r', alpha=0.3, s=2)
        ax2.set_title("Prediction (3D Scatter |val| > Mean)")
        fig.colorbar(img2, ax=ax2, shrink=0.5)
        
        plt.suptitle("3D Volumetric Visualization (High Intensity)")
        plt.savefig(os.path.join(output_dir, "results_3d_scatter.png"))
        print(f"3D Visualization saved to {output_dir}/results_3d_scatter.png")

if __name__ == "__main__":
    main()

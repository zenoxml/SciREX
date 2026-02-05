
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

from scirex.torch.sciml.utils.unit_gaussian_normalizer import UnitGaussianNormalizer
from scirex.torch.sciml.models.wno.wno_model import WNO2d
from scirex.torch.sciml.losses import LpLoss

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------------------
# TXT DATA LOADER
# ------------------------------------------------------------
def load_poisson_txt_dataset(data_dir, nx, ny):
    """
    Loads Poisson 2D dataset saved as TXT files:
    x y f u

    Returns:
        input_f  : (N, nx, ny, 1)
        output_u : (N, nx, ny, 1)
    """
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ])

    f_list, u_list = [], []

    for file in files:
        data = np.loadtxt(file, skiprows=1)  # skip header

        f = data[:, 2].reshape(nx, ny)
        u = data[:, 3].reshape(nx, ny)

        f_list.append(f)
        u_list.append(u)

    input_f = np.array(f_list)[..., None].astype(np.float32)
    output_u = np.array(u_list)[..., None].astype(np.float32)

    return torch.from_numpy(input_f), torch.from_numpy(output_u)

def main():
    # Parameters
    nx, ny = 64, 64
    batch_size = 20
    epochs = 100
    learning_rate = 1e-3
    ntrain = 400
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check outputs dir
    data_path = "src/support_files/poisson2d"
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_2d")
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------- Load dataset ----------------
    print(f"Loading Poisson 2D data from: {data_path}")
    input_f, output_u = load_poisson_txt_dataset(data_path, nx, ny)

    print(f"Dataset shape: {input_f.shape}")
    
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

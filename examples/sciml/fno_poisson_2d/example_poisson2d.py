import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from scirex.torch.sciml.utils.unit_gaussian_normalizer import UnitGaussianNormalizer
from scirex.torch.sciml.models.fno.simple_fno import SimpleFNO
from scirex.torch.sciml.losses import LpLoss

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------------------------
# TXT DATA LOADER
# ------------------------------------------------------------------
def load_poisson_txt_dataset(data_dir, nx, ny):
    """
    Loads Poisson 2D dataset saved as TXT files:
    x y f u

    Returns:
        input_f  : (N, 1, nx, ny)
        output_u : (N, 1, nx, ny)
    """
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ])

    f_list, u_list = [], []

    for file in files:
        data = np.loadtxt(file, skiprows=1)

        f = data[:, 2].reshape(nx, ny)
        u = data[:, 3].reshape(nx, ny)

        f_list.append(f)
        u_list.append(u)

    # NOTE: FNO expects (B, C, X, Y)
    input_f = np.array(f_list)[:, None, :, :].astype(np.float32)
    output_u = np.array(u_list)[:, None, :, :].astype(np.float32)

    return torch.from_numpy(input_f), torch.from_numpy(output_u)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    nx, ny = 64, 64
    batch_size = 20
    epochs = 100
    learning_rate = 1e-3
    ntrain = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "src/support_files/poisson2d"
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_2d_fno")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    print(f"Loading Poisson 2D data from: {data_path}")
    input_f, output_u = load_poisson_txt_dataset(data_path, nx, ny)

    print(f"Input shape:  {input_f.shape}")
    print(f"Output shape: {output_u.shape}")

    # ------------------------------------------------------------
    # Normalization (train statistics only)
    # ------------------------------------------------------------
    x_normalizer = UnitGaussianNormalizer(input_f[:ntrain])
    y_normalizer = UnitGaussianNormalizer(output_u[:ntrain])

    input_f = x_normalizer.encode(input_f)
    output_u = y_normalizer.encode(output_u)

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f[:ntrain], output_u[:ntrain]),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f[ntrain:], output_u[ntrain:]),
        batch_size=batch_size,
        shuffle=False
    )

    # ------------------------------------------------------------
    # Model: SimpleFNO
    # ------------------------------------------------------------
    model = SimpleFNO(
        n_modes=(16, 16),
        in_channels=1,        # f only (grid embedding is internal)
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        use_channel_mlp=True,
        channel_mlp_dropout=0.0,
        norm="batch_norm",
        fno_skip="linear"
    ).to(device)

    # ------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    criterion = nn.MSELoss()

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    train_losses, test_losses = [] , []

    y_normalizer.to(device)

    print("Starting FNO training...")
    t0 = default_timer()

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= len(train_loader)
        train_losses.append(ep_loss)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += criterion(out, y).item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if (ep + 1) % 5 == 0:
            print(
                f"Epoch {ep+1:3d}/{epochs}, "
                f"Train MSE: {ep_loss:.5e}, "
                f"Test MSE: {test_loss:.5e}"
            )

    t1 = default_timer()
    print(f"Training completed in {t1 - t0:.2f} seconds")

    # ------------------------------------------------------------
    # Evaluation (un-normalized)
    # ------------------------------------------------------------
    print("\nEvaluating final metrics...")
    model.eval()

    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            y_pred = y_normalizer.decode(out)
            y_true = y_normalizer.decode(y)

            all_y_true.append(y_true.cpu())
            all_y_pred.append(y_pred.cpu())

    all_y_true = torch.cat(all_y_true).numpy()
    all_y_pred = torch.cat(all_y_pred).numpy()

    l2_diff = np.linalg.norm(all_y_true - all_y_pred)
    l2_true = np.linalg.norm(all_y_true)
    rel_l2_error = l2_diff / l2_true

    ss_res = np.sum((all_y_true - all_y_pred) ** 2)
    ss_tot = np.sum((all_y_true - np.mean(all_y_true)) ** 2)
    r2_score = 1.0 - ss_res / ss_tot

    print("Final Metrics:")
    print(f"  Relative L2 Error: {rel_l2_error:.6f}")
    print(f"  R2 Score:          {r2_score:.6f}")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses, label="Test MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("FNO Training History")
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close()

    # Sample visualization
    idx = 0
    y_true = all_y_true[idx, 0]
    y_pred = all_y_pred[idx, 0]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(y_true, cmap="jet")
    axs[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(y_pred, cmap="jet")
    axs[1].set_title("FNO Prediction")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(np.abs(y_true - y_pred), cmap="jet")
    axs[2].set_title(f"Absolute Error (RelL2={rel_l2_error:.4f})")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_results.png"))
    plt.close()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

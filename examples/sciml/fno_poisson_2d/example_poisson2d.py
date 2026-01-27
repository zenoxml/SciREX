import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from dataclasses import asdict

# Add project root to sys.path to access config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from config.poisson import PoissonConfig
from scirex.torch.sciml.models.fno import FNO
from scirex.torch.sciml.utils.unit_gaussian_normalizer import UnitGaussianNormalizer
from data_generation.poisson import generate_poisson_data

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def get_activation(name):
    activations = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh, "silu": F.silu}
    return activations.get(name.lower(), F.gelu)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    cfg = PoissonConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {cfg.name}")

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_2d_fno_v2")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Data Generation
    # ------------------------------------------------------------
    input_f, output_u = generate_poisson_data(
        n_samples=cfg.data.n_train + cfg.data.n_test, 
        nx=cfg.data.nx, 
        ny=cfg.data.ny,
        include_mesh=cfg.data.include_mesh
    )
    
    ntrain, ntest = cfg.data.n_train, cfg.data.n_test
    print(f"Split: {ntrain} train, {ntest} test")

    # ------------------------------------------------------------
    # Normalization
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
        batch_size=cfg.optimization.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_f[ntrain:], output_u[ntrain:]),
        batch_size=cfg.optimization.batch_size, shuffle=False
    )

    # ------------------------------------------------------------
    # Model: FNO
    # ------------------------------------------------------------
    model_params = {
        "n_modes": cfg.model.n_modes,
        "in_channels": cfg.model.in_channels,
        "out_channels": cfg.model.out_channels,
        "hidden_channels": cfg.model.hidden_channels,
        "n_layers": cfg.model.n_layers,
        "lifting_channel_ratio": cfg.model.lifting_channel_ratio,
        "projection_channel_ratio": cfg.model.projection_channel_ratio,
        "non_linearity": get_activation(cfg.model.activation),
        "norm": cfg.model.norm,
        "use_channel_mlp": cfg.model.use_channel_mlp,
        "channel_mlp_expansion": cfg.model.channel_mlp_expansion,
        "fno_skip": cfg.model.skip_connection,
        "channel_mlp_skip": cfg.model.skip_connection,
        "domain_padding": cfg.model.domain_padding,
        "factorization": cfg.model.factorization,
        "rank": cfg.model.rank,
        "implementation": cfg.model.implementation,
        "separable": cfg.model.separable,
        "preactivation": cfg.model.preactivation,
        "complex_data": cfg.model.complex_data,
        "positional_embedding": None if cfg.data.include_mesh else "grid"
    }
    
    print("Initializing FNO model...")
    model = FNO(**model_params).to(device)

    # ------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.optimization.scheduler.step_size,
        gamma=cfg.optimization.scheduler.gamma
    )

    criterion = nn.MSELoss()

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    train_losses, test_losses = [] , []

    y_normalizer.to(device)

    print("Starting Training...")
    t0 = default_timer()

    for ep in range(cfg.optimization.n_epochs):
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

        if (ep + 1) % cfg.optimization.eval_interval == 0:
            print(
                f"Epoch {ep+1:3d}/{cfg.optimization.n_epochs}, "
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

    print(f"Final Relative L2 Error: {rel_l2_error:.6f}")
    print(f"Final R2 Score:          {r2_score:.6f}")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses, label="Test MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(f"FNO Training History - {cfg.name}")
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close()

    # Sample visualization
    idx = 0
    if all_y_true.shape[0] > 0:
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
        axs[2].set_title(f"Abs Error (RelL2={rel_l2_error:.4f})")
        plt.colorbar(im2, ax=axs[2])

        plt.suptitle(f"Poisson 2D - {cfg.name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "results.png"))
        plt.close()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

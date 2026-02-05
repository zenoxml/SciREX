import torch
import numpy as np
import os

# ============================================================
#  Poisson 2D data generation
# ============================================================

def generate_poisson_2d_data(n_samples=1000, nx=64, ny=64):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    f_list, u_list = [], []

    for _ in range(n_samples):
        m_max, n_max = 4, 4
        f_sample = np.zeros((nx, ny))
        u_sample = np.zeros((nx, ny))

        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                a_mn = np.random.uniform(-1, 1)
                mode = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)

                f_sample += a_mn * mode
                u_sample += a_mn * mode / ((m**2 + n**2) * np.pi**2)

        f_list.append(f_sample)
        u_list.append(u_sample)

    f = np.array(f_list)[..., None].astype(np.float32)
    u = np.array(u_list)[..., None].astype(np.float32)

    return torch.from_numpy(f), torch.from_numpy(u)


# ============================================================
def save_sample_txt(X, Y, f, u, filepath):
    """
    Saves a single sample as:
    x  y  f  u
    """
    data = np.column_stack([
        X.reshape(-1),
        Y.reshape(-1),
        f.reshape(-1),
        u.reshape(-1)
    ])

    header = "x y f u"
    np.savetxt(
        filepath,
        data,
        header=header,
        comments="",
        fmt="%.8e"
    )


# ============================================================
#  MAIN: Generate dataset
# ============================================================

def main():
    # -------- USER SETTINGS --------
    output_dir = "src/support_files/poisson2d"
    n_samples = 500
    nx = 64
    ny = 64
    # --------------------------------

    os.makedirs(output_dir, exist_ok=True)

    # --- Generate batch ---
    f_batch, u_batch = generate_poisson_2d_data(
        n_samples=n_samples,
        nx=nx,
        ny=ny
    )

    # Convert to NumPy (once)
    f_batch = f_batch.numpy()  # (N, nx, ny, 1)
    u_batch = u_batch.numpy()

    # Create grid once
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    print(f"Saving {n_samples} samples to {output_dir}")

    for i in range(n_samples):
        f = f_batch[i, :, :, 0]
        u = u_batch[i, :, :, 0]

        filepath = os.path.join(output_dir, f"sample_{i:05d}.txt")
        save_sample_txt(X, Y, f, u, filepath)

        if i % 50 == 0:
            print(f"Saved {i}/{n_samples}")

    print("✅ Dataset generation complete.")


if __name__ == "__main__":
    main()
import sys
import os
import glob
import torch
import numpy as np
import math
import scipy.interpolate
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Any

# Adjust paths to import from scirex and local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))

# Import Trainer and local components
sys.path.append(os.path.dirname(__file__))
from trainer import Trainer
from scirex.torch.sciml.models.fno.simple_fno import SimpleFNO
from scirex.utils import count_model_params

from scirex.torch.sciml.losses import LpLoss

# --- Configuration ---

class SDFModelConfig:
    model_arch: str = "simple_fno"
    n_modes: List[int] = [32, 32]
    hidden_channels: int = 64
    n_layers: int = 8 
    in_channels: int = 3 # x, y, sdf
    out_channels: int = 1 # A

class SDFOptConfig:
    n_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_step: int = 20
    scheduler_gamma: float = 0.5
    training_loss: str = "l2"
    mixed_precision: bool = False # Added explicitly for trainer
    eval_interval: int = 1

class SDFDataConfig:
    data_folder: str = "support_files/test_file" # Updated to match recent logs/usage
    train_test_split: float = 0.8
    resolution: int = 64 

class ExperimentConfig:
    model: SDFModelConfig = SDFModelConfig()
    opt: SDFOptConfig = SDFOptConfig()
    data: SDFDataConfig = SDFDataConfig()
    verbose: bool = True
    
# --- Dataset ---

class SDFTextDataset(Dataset):
    def __init__(self, root_dir, resolution=None):
        """
        Reads txt files with format: x, y, sdf, A
        """
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.txt")))
        if not self.files:
            print(f"Warning: No .txt files found in {root_dir}")
        
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load text file
        # Format: x y sdf A
        try:
            data = np.loadtxt(self.files[idx])
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            raise e
        
        # Check shape
        # Assuming data represents a grid. 
        # We need to figure out spatial dims (rx, ry).
        # If resolution is provided, reshape.
        # Otherwise try to infer square root.
        
        N = data.shape[0]
        if self.resolution:
            rx, ry = self.resolution, self.resolution
        else:
            root = int(math.sqrt(N))
            if root * root == N:
                rx, ry = root, root
            else:
                # If not square, just take it as 1D sequence or raise specific logic
                # For FNO 2D, we need a grid.
                # Let's assume 1D for safety if not square, but user said "fourth_mesh" implying 2D/3D mesh.
                # We will force 2D reshape guess.
                rx = root
                ry = N // root
        
        # Reshape to (rx, ry, channels)
        # Columns: 0:x, 1:y, 2:sdf, 3:A
        try:
            feature_x = data[:, 0].reshape(rx, ry)
            feature_y = data[:, 1].reshape(rx, ry)
            feature_sdf = data[:, 2].reshape(rx, ry)
            target_A = data[:, 3].reshape(rx, ry)
        except ValueError:
            # Fallback for shape mismatch: Interpolate onto a regular grid
            
            # Extract unstructured points
            points = data[:, 0:2] # x, y
            values_sdf = data[:, 2]
            values_A = data[:, 3]
            
            # Create target grid
            # Determine bounds
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            
            grid_x, grid_y = np.mgrid[
                min_x:max_x:complex(0, rx), 
                min_y:max_y:complex(0, ry)
            ]
            
            # Interpolate
            # Use 'nearest' to avoid NaNs outside hull if not perfect rect, or 'linear'
            feature_sdf = scipy.interpolate.griddata(points, values_sdf, (grid_x, grid_y), method='nearest')
            target_A = scipy.interpolate.griddata(points, values_A, (grid_x, grid_y), method='nearest')
            
            feature_x = grid_x
            feature_y = grid_y
        
        x_in = np.stack([feature_x, feature_y, feature_sdf], axis=0) # (3, rx, ry)
        y_out = target_A[np.newaxis, ...] # (1, rx, ry)
        
        return {
            'x': torch.from_numpy(x_in).float(),
            'y': torch.from_numpy(y_out).float()
        }


# --- Main ---

def main():
    # 1. Config
    # config = make_config_from_cli(ExperimentConfig)
    cfg = ExperimentConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Data
    print(f"Loading data from {cfg.data.data_folder}...")
    dataset = SDFTextDataset(cfg.data.data_folder, resolution=cfg.data.resolution)
    
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return

    # Use full dataset for training and testing (overfitting/single batch mode)
    # n_total = len(dataset)
    # n_train = int(n_total * cfg.data.train_test_split)
    # n_test = n_total - n_train
    
    # train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    # "take one sample as one batch" -> batch_size=1
    # "dont go with train test split" -> Use full dataset
    
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    test_loaders = {"test": test_loader}

    # 3. Model
    print(f"Initializing SimpleFNO with modes={cfg.model.n_modes}, hidden={cfg.model.hidden_channels}, layers={cfg.model.n_layers}")
    model = SimpleFNO(
        n_modes=tuple(cfg.model.n_modes),
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        positional_embedding=None # We provide x,y in input channels
    )
    model = model.to(device)
    print(f"Model params: {count_model_params(model)}")

    # 4. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.opt.learning_rate, 
        weight_decay=cfg.opt.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.opt.scheduler_step, gamma=cfg.opt.scheduler_gamma
    )
    
    # 5. Loss
    if cfg.opt.training_loss == "l2":
        training_loss = LpLoss(d=2, p=2)
    else:
        training_loss = torch.nn.MSELoss()
        
    eval_losses = {"l2": LpLoss(d=2, p=2)}

    # 6. Trainer
    trainer = Trainer(
        model=model,
        n_epochs=cfg.opt.n_epochs,
        device=device,
        verbose=cfg.verbose,
        mixed_precision=cfg.opt.mixed_precision,
    )
    
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir="checkpoints/sdf_experiment"
    )
    print("Training complete.")

if __name__ == "__main__":
    main()

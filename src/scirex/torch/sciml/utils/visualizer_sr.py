import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class Visualizer:
    @staticmethod
    def plot_results(pde_type: str, losses: Dict[str, List[float]], 
                    viz_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    test_metrics: Dict[str, float]):
        viz_x, viz_y, viz_pred = viz_data
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'FNO Results: {pde_type.upper()} Equation', 
                    fontsize=14, fontweight='bold')
        
        # Training and test loss
        axes[0, 0].plot(losses['train_losses'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(losses['test_losses'], label='Test Loss', linewidth=2, linestyle='--')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Training and Test Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Initial condition
        axes[0, 1].plot(viz_x[0, :, 0].numpy())
        axes[0, 1].set_xlabel('Spatial Position')
        axes[0, 1].set_ylabel('u(x, t=0)')
        axes[0, 1].set_title('Initial Condition')
        axes[0, 1].grid(True, alpha=0.3)
        
        # True vs predicted solution
        axes[1, 0].plot(viz_y[0].numpy(), label='True', linewidth=2)
        axes[1, 0].plot(viz_pred[0].numpy(), '--', label='FNO Prediction', linewidth=2)
        axes[1, 0].set_xlabel('Spatial Position')
        axes[1, 0].set_ylabel('u(x, t=T)')
        axes[1, 0].set_title('Solution at Final Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error
        error = np.abs(viz_y[0].numpy() - viz_pred[0].numpy())
        axes[1, 1].plot(error)
        axes[1, 1].set_xlabel('Spatial Position')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title(f'Prediction Error (Mean: {error.mean():.6f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'fno_{pde_type}_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def print_summary(pde_type: str, losses: Dict[str, List[float]], 
                     test_metrics: Dict[str, float], viz_error: float, 
                     viz_y: torch.Tensor):
        print(f"\nSummary for {pde_type.upper()}:")
        if len(losses['train_losses']) > 0:
            print(f"  Final Train Loss: {losses['train_losses'][-1]:.6f}")
            print(f"  Final Test Loss: {losses['test_losses'][-1]:.6f}")
            print(f"  Test MSE: {test_metrics['mse']:.6f}")
            print(f"  Test MAE: {test_metrics['mae']:.6f}")
            print(f"  Visualization Error: {viz_error:.6f}")
            rel_error = viz_error / (np.abs(viz_y[0].numpy()).mean() + 1e-8)
            print(f"  Relative Error: {rel_error:.4%}")
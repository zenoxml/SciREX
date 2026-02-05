
import sys
import torch
import torch.nn as nn
from timeit import default_timer
from typing import Dict, Any, Union
from pathlib import Path

class Trainer:
    """
    A simplified Trainer class for single-machine training.
    Removes distributed components, complex autoregression, and data processors.
    """

    def __init__(
        self,
        model: nn.Module,
        n_epochs: int,
        device: str = "cpu",
        verbose: bool = True,
        mixed_precision: bool = False,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        self.verbose = verbose
        self.mixed_precision = mixed_precision
        
        # Simple mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    def train(
        self,
        train_loader,
        test_loaders: Dict[str, Any],
        optimizer,
        scheduler,
        training_loss,
        eval_losses: Dict[str, Any],
        save_dir: str = "checkpoints",
        save_every: int = 10,
    ):
        """
        Main training loop.
        """
        self.model = self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Create checkpoint directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_test_loss = float("inf")

        print(f"Training on {len(train_loader.dataset)} samples")

        for epoch in range(self.n_epochs):
            t1 = default_timer()
            
            # --- Training Step ---
            train_err, avg_loss = self.train_one_epoch(train_loader, training_loss)
            
            epoch_time = default_timer() - t1
            
            # --- Validation Step ---
            # Evaluate on all test loaders (usually just 'test')
            eval_metrics = {}
            for name, loader in test_loaders.items():
                metrics = self.evaluate(loader, eval_losses, prefix=name)
                eval_metrics.update(metrics)
            
            # --- Logging ---
            if self.verbose:
                msg = f"[{epoch+1}/{self.n_epochs}] time={epoch_time:.2f}s | Train Loss: {avg_loss:.4f}"
                for k, v in eval_metrics.items():
                    msg += f" | {k}: {v:.4f}"
                print(msg)
                sys.stdout.flush()



            # --- Scheduler ---
            # If scheduler expects metrics (ReduceLROnPlateau), pass validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Assume first eval metric is the main one to track
                first_metric = list(eval_metrics.values())[0] if eval_metrics else avg_loss
                scheduler.step(first_metric)
            else:
                scheduler.step()

            # --- Checkpointing ---
            # Save best model
            current_eval_loss = list(eval_metrics.values())[0] if eval_metrics else avg_loss
            if current_eval_loss < best_test_loss:
                best_test_loss = current_eval_loss
                torch.save(self.model.state_dict(), save_path / "best_model.pt")
                if self.verbose:
                    print(f"  New best model saved! Loss: {best_test_loss:.4f}")
            
            # Save periodic
            if (epoch + 1) % save_every == 0:
                 torch.save(self.model.state_dict(), save_path / f"model_epoch_{epoch+1}.pt")


    def train_one_epoch(self, loader, criterion):
        self.model.train()
        total_loss = 0
        n_batches = len(loader)
        
        for batch in loader:
            # Move data to device
            # Expect batch to be a dict {'x': ..., 'y': ...}
            if isinstance(batch, dict):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
            else:
                # Fallback if list/tuple
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    loss = criterion(pred, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / n_batches
        return total_loss, avg_loss

    def evaluate(self, loader, metrics, prefix="val"):
        self.model.eval()
        results = {name: 0.0 for name in metrics.keys()}
        n_batches = len(loader)
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch['x'].to(self.device)
                    y = batch['y'].to(self.device)
                else:
                     x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x)
                else:
                    pred = self.model(x)
                
                for name, metric_fn in metrics.items():
                    results[name] += metric_fn(pred, y).item()
        
        # Average over batches
        return {f"{prefix}_{k}": v / n_batches for k, v in results.items()}

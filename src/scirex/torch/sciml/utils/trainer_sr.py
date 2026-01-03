"""
Training module for FNO super-resolution models.

This module provides a Trainer class that handles model training, validation,
and learning rate scheduling for Fourier Neural Operator models.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from config_sr import TrainingConfig
from ..models import FNO1d


class Trainer:
    """
    Trainer class for FNO super-resolution models.
    
    Handles training loops, evaluation, gradient clipping, and learning rate scheduling.
    """
    
    def __init__(self, model: FNO1d, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            model: FNO1d model instance to train
            config: TrainingConfig object containing hyperparameters
        """
        self.model = model
        self.config = config
        
        # Initialize AdamW optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler that reduces LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.lr_factor, 
            patience=config.lr_patience, 
        )
        
        # Track loss history for monitoring training progress
        self.train_losses = []
        self.test_losses = []
    
    def train_epoch(self, train_x: torch.Tensor, train_y: torch.Tensor) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_x: Input training data tensor
            train_y: Target training data tensor
            
        Returns:
            Average training loss for the epoch
            
        Raises:
            RuntimeError: If NaN or Inf values are detected in loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        num_samples = train_x.shape[0]
        
        # Shuffle training data indices for better generalization
        shuffled_indices = torch.randperm(num_samples)
        
        # Iterate through batches
        for batch_start in range(0, num_samples, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, num_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]
            
            # Get batch data
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            
            # Check for numerical instability
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Numerical instability detected: NaN or Inf loss value")
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.grad_clip_norm
            )
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def evaluate(self, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            test_x: Input test data tensor
            test_y: Target test data tensor
            
        Returns:
            Mean squared error loss on test data
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_x)
            loss = F.mse_loss(predictions, test_y).item()
        return loss
    
    def train(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor,
        test_x: torch.Tensor, 
        test_y: torch.Tensor
    ) -> Dict[str, List[float]]:
        """
        Full training loop over multiple epochs.
        
        Args:
            train_x: Input training data tensor
            train_y: Target training data tensor
            test_x: Input test data tensor
            test_y: Target test data tensor
            
        Returns:
            Dictionary containing:
                - 'train_losses': List of training losses per epoch
                - 'test_losses': List of test losses per epoch
        """
        for epoch in range(self.config.epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_x, train_y)
            
            # Evaluate on test set
            test_loss = self.evaluate(test_x, test_y)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            # Adjust learning rate based on test loss plateau
            self.scheduler.step(test_loss)
            
            # Log progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'Epoch {epoch + 1:3d}/{self.config.epochs} | '
                    f'Train Loss: {train_loss:.6f} | '
                    f'Test Loss: {test_loss:.6f} | '
                    f'Learning Rate: {current_lr:.6e}'
                )
        
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
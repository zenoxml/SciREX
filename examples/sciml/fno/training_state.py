import os
import torch
from pathlib import Path

def save_training_state(save_dir, save_name, model, optimizer=None, scheduler=None, regularizer=None, epoch=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save best or regular
    path = save_dir / f"{save_name}_state_dict.pt"
    torch.save(state, path)


def load_training_state(save_dir, save_name, model, optimizer=None, regularizer=None, scheduler=None):
    save_dir = Path(save_dir)
    path = save_dir / f"{save_name}_state_dict.pt"
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    
    return model, optimizer, scheduler, regularizer, epoch

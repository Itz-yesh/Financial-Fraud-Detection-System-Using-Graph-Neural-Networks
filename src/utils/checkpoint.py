"""
Model Checkpointing Utilities
Save and restore model checkpoints during training
"""

import torch
import os
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str = "experiments/checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_metric = None
        self.best_checkpoint_path = None
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       filename: Optional[str] = None) -> str:
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Configuration dictionary
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def save_best_checkpoint(self,
                            model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            epoch: int,
                            metrics: Dict[str, float],
                            config: Dict[str, Any],
                            metric_name: str = 'val_roc_auc',
                            mode: str = 'max') -> Optional[str]:
        """
        Save checkpoint if it's the best so far
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Configuration dictionary
            metric_name: Metric to track for best model
            mode: 'max' or 'min' for metric optimization
            
        Returns:
            Path to saved checkpoint if it's the best, None otherwise
        """
        current_metric = metrics.get(metric_name)
        
        if current_metric is None:
            return None
        
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif mode == 'max' and current_metric > self.best_metric:
            is_best = True
        elif mode == 'min' and current_metric < self.best_metric:
            is_best = True
        
        if is_best:
            self.best_metric = current_metric
            filename = "best_model.pt"
            checkpoint_path = self.save_checkpoint(
                model, optimizer, epoch, metrics, config, filename
            )
            self.best_checkpoint_path = checkpoint_path
            return checkpoint_path
        
        return None
    
    def load_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_path: Optional[str] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to checkpoint file (uses best if None)
            device: Device to load model to
            
        Returns:
            Dictionary containing checkpoint information
        """
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
            
            # Fallback to looking for best_model.pt if we haven't trained in this session
            if checkpoint_path is None:
                default_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                if os.path.exists(default_path):
                    checkpoint_path = default_path
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints
        
        Returns:
            List of checkpoint filenames
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.endswith('.pt')]
        return sorted(checkpoints)
    
    def delete_old_checkpoints(self, keep_last_n: int = 5):
        """
        Delete old checkpoints, keeping only the last N
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Don't delete best model
        checkpoints = [c for c in checkpoints if c != "best_model.pt"]
        
        if len(checkpoints) > keep_last_n:
            to_delete = checkpoints[:-keep_last_n]
            for checkpoint in to_delete:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
                os.remove(checkpoint_path)


def save_model(model: torch.nn.Module, 
               filepath: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               **kwargs):
    """
    Convenience function to save model
    
    Args:
        model: PyTorch model
        filepath: Path to save model
        optimizer: Optimizer (optional)
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)


def load_model(model: torch.nn.Module,
               filepath: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: str = 'cpu') -> Dict[str, Any]:
    """
    Convenience function to load model
    
    Args:
        model: PyTorch model
        filepath: Path to checkpoint
        optimizer: Optimizer (optional)
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

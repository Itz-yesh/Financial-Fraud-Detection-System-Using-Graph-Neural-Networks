"""
Logging Utilities
Structured logging for experiments with TensorBoard integration
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Handles logging for experiments with file and TensorBoard output"""
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = "results/logs",
                 use_tensorboard: bool = True):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            tb_dir = os.path.join(log_dir, "tensorboard", f"{experiment_name}_{timestamp}")
            self.writer = SummaryWriter(tb_dir)
            self.info(f"TensorBoard logging to: {tb_dir}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to file and TensorBoard
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        # Log to file
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} - {prefix} {metric_str}")
        
        # Log to TensorBoard
        if self.writer:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.info("Hyperparameters:")
        for key, value in hparams.items():
            self.info(f"  {key}: {value}")
        
        if self.writer:
            # Convert all values to strings for TensorBoard
            hparams_str = {k: str(v) for k, v in hparams.items()}
            self.writer.add_hparams(hparams_str, {})
    
    def log_model_architecture(self, model: Any):
        """
        Log model architecture
        
        Args:
            model: PyTorch model
        """
        self.info("Model Architecture:")
        self.info(str(model))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
    
    def close(self):
        """Close logger and TensorBoard writer"""
        if self.writer:
            self.writer.close()
        
        # Remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def get_logger(experiment_name: str, 
               log_dir: str = "results/logs",
               use_tensorboard: bool = True) -> ExperimentLogger:
    """
    Convenience function to get experiment logger
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        use_tensorboard: Whether to use TensorBoard
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, log_dir, use_tensorboard)

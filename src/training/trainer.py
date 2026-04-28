"""
Trainer Module
Complete training loop with weighted loss, early stopping, and checkpointing
Implements training strategies from Tong et al. (2023)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from typing import Dict, Optional, Tuple
import time

from ..utils.logger import ExperimentLogger
from ..utils.checkpoint import CheckpointManager
from .loss_functions import get_loss_function


class Trainer:
    """Handles model training with advanced features"""
    
    def __init__(self,
                 model: nn.Module,
                 data: Data,
                 config: Dict,
                 logger: Optional[ExperimentLogger] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Initialize trainer
        
        Args:
            model: GNN model to train
            data: Graph data
            config: Training configuration
            logger: Experiment logger
            checkpoint_manager: Checkpoint manager
        """
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        
        # Setup device
        self.device = self._setup_device(config.get('device', 'auto'))
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(config)
        
        # Setup loss function
        self.criterion = self._setup_loss_function(config)
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 15)

        # Mini-batch threshold: use NeighborLoader if graph exceeds this size
        self._minibatch_threshold = 50_000  # nodes
        self._use_minibatch = self.data.num_nodes > self._minibatch_threshold
        if self._use_minibatch:
            print(f"[Trainer] Graph has {self.data.num_nodes:,} nodes — "
                  f"switching to mini-batch NeighborLoader training.")
            self.train_loader, self.val_loader = self._setup_loaders(config)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup compute device"""
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_config)
    
    def _setup_optimizer(self, config: Dict) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_loss_function(self, config: Dict) -> nn.Module:
        """Setup loss function with class imbalance handling"""
        imbalance_config = config.get('class_imbalance', {})
        method = imbalance_config.get('method', 'weighted_loss')
        
        if method == 'weighted_loss':
            pos_weight = imbalance_config.get('pos_weight', 10.0)
            return get_loss_function('weighted_bce', pos_weight=pos_weight)
        elif method == 'focal_loss':
            gamma = imbalance_config.get('focal_loss_gamma', 2.0)
            return get_loss_function('focal', gamma=gamma)
        else:
            return get_loss_function('balanced_ce')
    
    def _setup_loaders(self, config: Dict):
        """Create NeighborLoader instances for scalable mini-batch training."""
        num_neighbors = config.get('num_neighbors', [20, 10, 10])
        batch_size = config.get('batch_size', 512)

        train_loader = NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.data.train_mask,
            shuffle=True,
        )
        val_loader = NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.data.val_mask,
            shuffle=False,
        )
        return train_loader, val_loader

    def _setup_scheduler(self, config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_name = config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.get('scheduler_factor', 0.5),
                patience=config.get('scheduler_patience', 10),
                verbose=True
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('scheduler_step_size', 30),
                gamma=config.get('scheduler_factor', 0.5)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (full-graph or mini-batch depending on graph size)."""
        self.model.train()

        if self._use_minibatch:
            return self._train_epoch_minibatch()
        else:
            return self._train_epoch_fullgraph()

    def _train_epoch_fullgraph(self) -> Dict[str, float]:
        """Full-graph forward pass (only safe for small graphs)."""
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out, self.data.y, self.data.train_mask)

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config['gradient_clip']
            )
        self.optimizer.step()

        pred = out[self.data.train_mask].argmax(dim=1)
        correct = (pred == self.data.y[self.data.train_mask]).sum()
        acc = int(correct) / int(self.data.train_mask.sum())
        return {'loss': loss.item(), 'accuracy': acc}

    def _train_epoch_minibatch(self) -> Dict[str, float]:
        """Mini-batch training using NeighborLoader."""
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)

            # Only compute loss on the "seed" nodes (first batch_size nodes)
            batch_size = batch.batch_size
            loss = self.criterion(out[:batch_size], batch.y[:batch_size],
                                  torch.ones(batch_size, dtype=torch.bool, device=self.device))

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clip']
                )
            self.optimizer.step()

            pred = out[:batch_size].argmax(dim=1)
            total_correct += int((pred == batch.y[:batch_size]).sum())
            total_loss += float(loss) * batch_size
            total_nodes += batch_size

        return {
            'loss': total_loss / max(total_nodes, 1),
            'accuracy': total_correct / max(total_nodes, 1)
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model (full-graph or mini-batch depending on graph size)."""
        self.model.eval()

        if self._use_minibatch:
            return self._validate_minibatch()
        else:
            return self._validate_fullgraph()

    @torch.no_grad()
    def _validate_fullgraph(self) -> Dict[str, float]:
        """Full-graph validation."""
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out, self.data.y, self.data.val_mask)
        pred = out[self.data.val_mask].argmax(dim=1)
        correct = (pred == self.data.y[self.data.val_mask]).sum()
        acc = int(correct) / int(self.data.val_mask.sum())
        return {'loss': loss.item(), 'accuracy': acc}

    @torch.no_grad()
    def _validate_minibatch(self) -> Dict[str, float]:
        """Mini-batch validation using NeighborLoader."""
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            batch_size = batch.batch_size
            loss = self.criterion(out[:batch_size], batch.y[:batch_size],
                                  torch.ones(batch_size, dtype=torch.bool, device=self.device))
            pred = out[:batch_size].argmax(dim=1)
            total_correct += int((pred == batch.y[:batch_size]).sum())
            total_loss += float(loss) * batch_size
            total_nodes += batch_size

        return {
            'loss': total_loss / max(total_nodes, 1),
            'accuracy': total_correct / max(total_nodes, 1)
        }
    
    def train(self, num_epochs: int) -> Dict:
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        if self.logger:
            self.logger.info(f"Starting training on device: {self.device}")
            self.logger.info(f"Training for {num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Log metrics
            if self.logger:
                self.logger.log_metrics(train_metrics, epoch, prefix='train')
                self.logger.log_metrics(val_metrics, epoch, prefix='val')
            
            # Print progress
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:03d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Checkpointing
            if self.checkpoint_manager and epoch % self.config.get('save_every_n_epochs', 5) == 0:
                metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config
                )
            
            # Save best model
            if self.checkpoint_manager and val_metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = val_metrics['accuracy']
                metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                self.checkpoint_manager.save_best_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config,
                    metric_name='val_accuracy', mode='max'
                )
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                print(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        if self.logger:
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Best validation accuracy: {self.best_val_metric:.4f}")
        
        return self.history


def train_model(model: nn.Module,
               data: Data,
               config: Dict,
               logger: Optional[ExperimentLogger] = None,
               checkpoint_manager: Optional[CheckpointManager] = None) -> Dict:
    """
    Convenience function to train a model
    
    Args:
        model: GNN model
        data: Graph data
        config: Training configuration
        logger: Experiment logger
        checkpoint_manager: Checkpoint manager
        
    Returns:
        Training history
    """
    trainer = Trainer(model, data, config, logger, checkpoint_manager)
    num_epochs = config.get('num_epochs', 100)
    return trainer.train(num_epochs)

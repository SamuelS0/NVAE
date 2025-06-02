import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Any
from tqdm import tqdm
import core.CRMNIST.utils_crmnist as utils_crmnist
from core.utils import process_batch, get_model_name, _calculate_metrics

class CRMNISTTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args,
        patience: int = 5
    ):
        self.model = model.to(device)  # Move model to device immediately
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.patience = patience
        self.dataset = args.dataset
        # Early stopping setup
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0  # Track best accuracy for early stopping
        self.best_model_state = None
        self.patience_counter = 0
        self.epochs_trained = 0
        self.best_epoch = 0
        self.best_batch_metrics = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        
        # Create output directories with simpler structure
        self.models_dir = os.path.join(args.out, 'comparison_models', args.setting)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create reconstructions directory
        self.reconstructions_dir = os.path.join(args.out, 'reconstructions')
        os.makedirs(self.reconstructions_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs: int) -> torch.nn.Module:
        """Train the model with early stopping and model checkpointing."""
        # Select a diverse sample batch for reconstruction visualization
        # sample_batch = utils_crmnist.select_diverse_sample_batch(val_loader, self.args)
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')
            
            # Generate and visualize reconstructions after each epoch
            # Skip for DANN models which don't do reconstructions
            # if not isinstance(self.model, DANN):
            #     utils_crmnist.visualize_reconstructions(self.model, epoch+1, sample_batch, self.args, self.reconstructions_dir)
            
            # Early stopping check
            if self._check_early_stopping(val_loss, epoch, num_epochs, val_metrics):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.epochs_trained = epoch + 1
        
    def _train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        train_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        num_batches = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Training")
        
        for batch_idx, batch in train_pbar:
            self.optimizer.zero_grad()
            
            # Move data to device
            #x, y, c, r = x.to(self.device), y.to(self.device), c.to(self.device), r.to(self.device)
            x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)
            # Forward pass and loss calculation
            loss = self.model.loss_function(y, x, r)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics
            if batch_idx % 10 == 0:  # Calculate every 10 batches to save computation
                batch_metrics = _calculate_metrics(self.model, y, x, r)
                for k, v in batch_metrics.items():
                    train_metrics_sum[k] += v
                num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix(loss=loss.item())
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / num_batches for k, v in train_metrics_sum.items()}
        
        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        val_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                       desc=f"Validating")
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)
                
                loss = self.model.loss_function(y, x, r)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = _calculate_metrics(self.model, y, x, r)
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v
                
                # Update progress bar
                val_pbar.set_postfix(loss=loss.item())
        
        # Calculate averages
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        
        return val_loss, val_metrics
    
    
    
    
    def _check_early_stopping(self, val_loss: float, epoch: int, num_epochs: int, batch_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met based on validation accuracy."""
        current_val_accuracy = batch_metrics['y_accuracy']
        
        # Early stopping based on validation accuracy
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.best_epoch = epoch
            self.patience_counter = 0
            self.best_batch_metrics = batch_metrics
            
            # Save best model with consistent naming
            model_type = getattr(self.model, 'name', self.model.__class__.__name__.lower())
            model_name = get_model_name(self.args, model_type)
            best_model_path = os.path.join(self.models_dir, f'{model_name}.pt')
            
            
            torch.save({
                'state_dict': self.best_model_state,
                'training_metrics': self.best_batch_metrics,
                'epoch': self.best_epoch,
                'model_type': model_type
            }, best_model_path)
            print(f"  New best model saved! (Validation Accuracy: {self.best_val_accuracy:.4f}, Loss: {val_loss:.4f})")
            print(f"  Best model batch metrics: {self.best_batch_metrics}")
            return False
        else:
            self.patience_counter += 1
            print(f"  No improvement in validation accuracy. Patience: {self.patience_counter}/{self.patience}")
            
            # Use min(10, num_epochs // 2) as minimum epochs requirement
            min_required_epochs = min(10, num_epochs // 2)
            if self.patience_counter >= self.patience and epoch >= min_required_epochs:
                return True
            return False

    def save_final_model(self, epoch: int):
        """Save the final model state."""
        # Get model type from model name or class name
        model_type = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        model_name = get_model_name(self.args, model_type)
        final_model_path = os.path.join(self.models_dir, f'{model_name}_final.pt')
        
        # Get model parameters from training metrics
        #model_params = self.args.model_params if hasattr(self.args, 'model_params') else None
        
        torch.save({
            #'params': self.model_params,
            'state_dict': self.model.state_dict(),
            'training_metrics': self.best_batch_metrics,
            'epoch': epoch,
            'model_type': model_type
        }, final_model_path)
        print(f"Final model saved to {final_model_path}") 
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Any
from tqdm import tqdm
import core.CRMNIST.utils_crmnist as utils_crmnist
from core.utils import process_batch, get_model_name, _calculate_metrics, visualize_latent_spaces

class CRMNISTTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args,
        patience: int = 5,
        scheduler=None
    ):
        self.model = model.to(device)  # Move model to device immediately
        self.optimizer = optimizer
        self.scheduler = scheduler  # LR scheduler (optional)
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

        # Training history tracking
        self.epoch_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_recon_history = []
        self.val_recon_history = []
        self.lr_history = []  # Track learning rate

        # Create output directories with simpler structure
        self.models_dir = os.path.join(args.out, 'comparison_models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Create reconstructions directory
        self.reconstructions_dir = os.path.join(args.out, 'reconstructions')
        os.makedirs(self.reconstructions_dir, exist_ok=True)

        # Create latent visualization directory
        self.latent_viz_dir = os.path.join(args.out, 'latent_epoch_viz')
        os.makedirs(self.latent_viz_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs: int) -> torch.nn.Module:
        """Train the model with early stopping and model checkpointing."""
        # Select a diverse sample batch for reconstruction visualization
        # sample_batch = utils_crmnist.select_diverse_sample_batch(val_loader, self.args)
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)

            # Step the LR scheduler at the end of each epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"  LR decay: {current_lr:.6f} → {new_lr:.6f}")

            # Store training history
            self.epoch_history.append(epoch + 1)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_metrics.get('y_accuracy', 0))
            self.val_acc_history.append(val_metrics.get('y_accuracy', 0))
            self.train_recon_history.append(train_metrics.get('recon_mse', 0))
            self.val_recon_history.append(val_metrics.get('recon_mse', 0))
            self.lr_history.append(current_lr)

            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')

            # Visualize latent spaces after each epoch
            # DISABLED: Only generate final visualization after all epochs complete (saves time)
            # self.visualize_latent_epoch(val_loader, epoch)

            # Generate and visualize reconstructions after each epoch
            # Skip for DANN models which don't do reconstructions
            # if not isinstance(self.model, DANN):
            #     utils_crmnist.visualize_reconstructions(self.model, epoch+1, sample_batch, self.args, self.reconstructions_dir)
            
            # Early stopping check
            if self._check_early_stopping(val_loss, epoch, num_epochs, val_metrics):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.epochs_trained = epoch + 1

        # Save training history and plot curves
        print("\n📊 Saving training history and generating plots...")
        self.save_training_history()
        self.plot_training_curves()

    def _train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        train_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"Training")

        for batch_idx, batch in train_pbar:
            self.optimizer.zero_grad()

            # Move data to device
            #x, y, c, r = x.to(self.device), y.to(self.device), c.to(self.device), r.to(self.device)
            x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)
            # Forward pass and loss calculation
            loss_result = self.model.loss_function(y, x, r)

            # Handle both old (scalar) and new (tuple) return formats
            if isinstance(loss_result, tuple):
                loss, loss_components = loss_result
            else:
                loss = loss_result
                loss_components = None

            # Backward pass
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Calculate metrics every batch for consistency
            batch_metrics = _calculate_metrics(self.model, y, x, r, 'train')
            for k, v in batch_metrics.items():
                train_metrics_sum[k] += v

            # Update progress bar with detailed loss components
            if loss_components:
                train_pbar.set_postfix(
                    loss=f"{loss_components['total']:.2f}",
                    recon=f"{loss_components['recon']:.1f}",
                    y_ce=f"{loss_components['y_ce']:.1f}",
                    a_ce=f"{loss_components['a_ce']:.1f}",
                    l1_zy=f"{loss_components['l1_zy']:.3f}",
                    l1_za=f"{loss_components['l1_za']:.3f}",
                    l1_zay=f"{loss_components['l1_zay']:.3f}"
                )
            else:
                train_pbar.set_postfix(loss=loss.item())

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}

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

                loss_result = self.model.loss_function(y, x, r)

                # Handle both old (scalar) and new (tuple) return formats
                if isinstance(loss_result, tuple):
                    loss, loss_components = loss_result
                else:
                    loss = loss_result
                    loss_components = None

                val_loss += loss.item()

                # Calculate metrics
                batch_metrics = _calculate_metrics(self.model, y, x, r)
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v

                # Update progress bar with detailed loss components
                if loss_components:
                    val_pbar.set_postfix(
                        loss=f"{loss_components['total']:.2f}",
                        recon=f"{loss_components['recon']:.1f}",
                        y_ce=f"{loss_components['y_ce']:.1f}",
                        a_ce=f"{loss_components['a_ce']:.1f}",
                        l1_zy=f"{loss_components['l1_zy']:.3f}",
                        l1_za=f"{loss_components['l1_za']:.3f}",
                        l1_zay=f"{loss_components['l1_zay']:.3f}"
                    )
                else:
                    val_pbar.set_postfix(loss=loss.item())
        
        # Calculate averages
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        
        return val_loss, val_metrics
    
    
    
    
    def _check_early_stopping(self, val_loss: float, epoch: int, num_epochs: int, batch_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met based on validation accuracy."""
        current_val_accuracy = batch_metrics.get('y_accuracy', 0)
        
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
            if self.patience_counter >= self.patience and epoch + 1 >= min_required_epochs:
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
    
    def visualize_latent_epoch(self, val_loader, epoch):
        """
        Visualize latent spaces for the current epoch using balanced sampling.
        
        The visualization uses balanced sampling to collect equal numbers of samples
        for each (digit × color × rotation) combination, ensuring good representation
        of all data variations. Works for both NVAE and DIVA models.
        """
        try:
            latent_path = os.path.join(self.latent_viz_dir, f'crmnist_latent_epoch_{epoch+1:03d}.png')
            visualize_latent_spaces(
                model=self.model,
                dataloader=val_loader,
                device=self.device,
                type='crmnist',  # Uses balanced sampling for color × rotation combinations
                save_path=latent_path,
                max_samples=500
            )
            print(f"  Latent visualization saved to {latent_path}")
        except Exception as e:
            print(f"  Warning: Could not generate latent visualization for epoch {epoch+1}: {e}")

    def save_training_history(self):
        """Save training history to JSON and CSV files."""
        import json
        import pandas as pd

        # Create model-specific subdirectory for training history
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        history_dir = os.path.join(self.args.out, f'{model_name}_training')
        os.makedirs(history_dir, exist_ok=True)

        # Prepare history dictionary
        history = {
            'epoch': self.epoch_history,
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'train_accuracy': self.train_acc_history,
            'val_accuracy': self.val_acc_history,
            'train_recon_mse': self.train_recon_history,
            'val_recon_mse': self.val_recon_history
        }

        # Save as JSON
        json_path = os.path.join(history_dir, 'training_history.json')
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"   ✅ Training history saved to {json_path}")

        # Save as CSV
        csv_path = os.path.join(history_dir, 'training_history.csv')
        df = pd.DataFrame(history)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Training history saved to {csv_path}")

    def plot_training_curves(self):
        """Plot and save training/validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.epoch_history, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(self.epoch_history, self.train_acc_history, 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.epoch_history, self.val_acc_history, 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Reconstruction MSE curves
        axes[1, 0].plot(self.epoch_history, self.train_recon_history, 'b-', label='Train Recon MSE', linewidth=2)
        axes[1, 0].plot(self.epoch_history, self.val_recon_history, 'r-', label='Val Recon MSE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Reconstruction MSE', fontsize=12)
        axes[1, 0].set_title('Reconstruction Quality', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # Combined loss/accuracy plot
        ax_twin = axes[1, 1].twinx()
        axes[1, 1].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        ax_twin.plot(self.epoch_history, self.val_acc_history, 'g-', label='Val Accuracy', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Validation Loss', fontsize=12, color='r')
        ax_twin.set_ylabel('Validation Accuracy', fontsize=12, color='g')
        axes[1, 1].set_title('Validation Loss vs Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='y', labelcolor='r')
        ax_twin.tick_params(axis='y', labelcolor='g')
        axes[1, 1].grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')

        plt.tight_layout()

        # Save the plot in model-specific subdirectory
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        history_dir = os.path.join(self.args.out, f'{model_name}_training')
        os.makedirs(history_dir, exist_ok=True)
        plot_path = os.path.join(history_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training curves saved to {plot_path}") 
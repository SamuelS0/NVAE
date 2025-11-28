import torch
import os
import numpy as np
from typing import Dict, Tuple, Any

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    # Create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import core.CRMNIST.utils_crmnist as utils_crmnist
    from core.utils import process_batch, get_model_name
except ImportError as e:
    raise ImportError(f"Required CRMNIST modules not found: {e}. Please ensure the core modules are properly installed.")


class DANNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args,
        patience: int = 5,
        lambda_schedule: bool = True,
        scheduler=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler  # LR scheduler (optional)
        self.device = device
        self.args = args
        self.patience = patience
        self.dataset = args.dataset
        self.lambda_schedule = lambda_schedule

        # Early stopping setup
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.epochs_trained = 0
        self.best_epoch = 0
        self.best_batch_metrics = {
            'loss_y_main': 0, 'loss_d_main': 0, 'loss_d_adversarial': 0,
            'loss_y_adversarial': 0, 'sparsity_loss_zdy': 0, 'sparsity_loss_zy': 0, 'sparsity_loss_zd': 0,
            'y_accuracy': 0, 'a_accuracy': 0
        }

        # Create output directories
        # Use getattr to support both CRMNIST (has 'setting') and WILD (doesn't have 'setting')
        setting = getattr(args, 'setting', 'standard')
        self.models_dir = os.path.join(args.out, 'comparison_models', setting)
        os.makedirs(self.models_dir, exist_ok=True)

        # Track training progress for lambda scheduling
        self.current_epoch = 0
        self.total_epochs = 0
        self.lambda_history = []

        # Training history tracking
        self.epoch_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_y_acc_history = []
        self.val_y_acc_history = []
        self.train_d_acc_history = []
        self.val_d_acc_history = []
        self.lr_history = []  # Track learning rate

    def train(self, train_loader, val_loader, num_epochs: int) -> torch.nn.Module:
        """Train the DANN model with gradient reversal lambda scheduling."""
        self.total_epochs = num_epochs
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update gradient reversal lambda if scheduling is enabled
            if self.lambda_schedule:
                lambda_val = self.model.update_lambda_schedule(epoch, num_epochs)
                self.lambda_history.append(lambda_val)
                print(f'Epoch {epoch+1}/{num_epochs}: Gradient Reversal λ = {lambda_val:.4f}')
                print(f'  Sparsity weights: zdy={self.model.sparsity_weight_zdy_current:.4f}, '
                      f'zy={self.model.sparsity_weight_zy_current:.4f}, zd={self.model.sparsity_weight_zd_current:.4f}')
            
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
            self.train_y_acc_history.append(train_metrics.get('y_accuracy', 0))
            self.val_y_acc_history.append(val_metrics.get('y_accuracy', 0))
            self.train_d_acc_history.append(train_metrics.get('a_accuracy', 0))
            self.val_d_acc_history.append(val_metrics.get('a_accuracy', 0))
            self.lr_history.append(current_lr)

            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}')
            self._print_metrics(train_metrics, prefix='Train')
            print(f'  Val Loss: {val_loss:.4f}')
            self._print_metrics(val_metrics, prefix='Val')
            
            # Early stopping check
            if self._check_early_stopping(val_loss, epoch, num_epochs, val_metrics):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.epochs_trained = epoch + 1

        # Save training history and plot curves
        print("\n📊 Saving training history and generating plots...")
        self.save_training_history()
        self.plot_training_curves()

        # Plot lambda schedule if enabled
        if self.lambda_schedule and len(self.lambda_history) > 1:
            self._plot_lambda_schedule()
    
    def _train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with DANN-specific loss computation."""
        self.model.train()
        train_loss = 0
        train_metrics_sum = {
            'loss_y_main': 0, 'loss_d_main': 0, 'loss_d_adversarial': 0,
            'loss_y_adversarial': 0, 'sparsity_loss_zdy': 0, 'sparsity_loss_zy': 0, 'sparsity_loss_zd': 0,
            'y_accuracy': 0, 'a_accuracy': 0
        }
        num_batches = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Training DANN")
        
        for batch_idx, batch in train_pbar:
            self.optimizer.zero_grad()
            
            # Move data to device
            x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)

            # Forward pass and detailed loss calculation
            loss, loss_dict = self.model.detailed_loss(x, y, r)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics every 10 batches to save computation
            if batch_idx % 10 == 0:
                batch_metrics = self._calculate_dann_metrics(y, x, r)
                
                # Add loss components to metrics
                for k, v in loss_dict.items():
                    if k in train_metrics_sum:
                        train_metrics_sum[k] += v
                
                # Add accuracy metrics
                for k, v in batch_metrics.items():
                    if k in train_metrics_sum:
                        train_metrics_sum[k] += v
                
                num_batches += 1
            
            # Update progress bar with main metrics
            train_pbar.set_postfix({
                'loss': loss.item(),
                'y_acc': batch_metrics.get('y_accuracy', 0),
                'a_acc': batch_metrics.get('a_accuracy', 0)
            })
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / max(num_batches, 1) for k, v in train_metrics_sum.items()}
        
        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate the DANN model."""
        self.model.eval()
        val_loss = 0
        val_metrics_sum = {
            'loss_y_main': 0, 'loss_d_main': 0, 'loss_d_adversarial': 0,
            'loss_y_adversarial': 0, 'sparsity_loss_zdy': 0, 'sparsity_loss_zy': 0, 'sparsity_loss_zd': 0,
            'y_accuracy': 0, 'a_accuracy': 0
        }
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                       desc=f"Validating DANN")
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)

                # Forward pass and detailed loss calculation
                loss, loss_dict = self.model.detailed_loss(x, y, r)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = self._calculate_dann_metrics(y, x, r)
                
                # Add loss components to metrics
                for k, v in loss_dict.items():
                    if k in val_metrics_sum:
                        val_metrics_sum[k] += v
                
                # Add accuracy metrics
                for k, v in batch_metrics.items():
                    if k in val_metrics_sum:
                        val_metrics_sum[k] += v
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'y_acc': batch_metrics.get('y_accuracy', 0),
                    'a_acc': batch_metrics.get('a_accuracy', 0)
                })
        
        # Calculate averages
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        
        return val_loss, val_metrics
    
    def _calculate_dann_metrics(self, y, x, r) -> Dict[str, float]:
        """Calculate DANN-specific metrics."""
        with torch.no_grad():
            # Convert one-hot to indices if necessary (do this BEFORE dann_forward)
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_true = torch.argmax(y, dim=1)
            else:
                y_true = y.long()

            if len(r.shape) > 1 and r.shape[1] > 1:
                r_true = torch.argmax(r, dim=1)
            else:
                r_true = r.long()

            # Pass y and d for conditional adversarial training
            outputs = self.model.dann_forward(x, y=y_true, d=r_true)
            
            # Main task accuracies
            y_pred = outputs['y_pred_main'].argmax(dim=1)
            y_accuracy = (y_pred == y_true).float().mean().item()
            
            a_pred = outputs['d_pred_main'].argmax(dim=1)
            a_accuracy = (a_pred == r_true).float().mean().item()
            
            # Adversarial task accuracies (these should be low for good disentanglement)
            d_pred_adversarial = outputs['d_pred_adversarial'].argmax(dim=1)
            d_adv_accuracy = (d_pred_adversarial == r_true).float().mean().item()
            
            y_pred_adversarial = outputs['y_pred_adversarial'].argmax(dim=1)
            y_adv_accuracy = (y_pred_adversarial == y_true).float().mean().item()
            
            return {
                'y_accuracy': y_accuracy,
                'a_accuracy': a_accuracy,
                'd_adv_accuracy': d_adv_accuracy,  # Should be low
                'y_adv_accuracy': y_adv_accuracy   # Should be low
            }
    
    def _print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Print metrics in an organized way."""
        if prefix:
            prefix = f"  {prefix} "
        else:
            prefix = "  "
            
        # Main accuracies
        if 'y_accuracy' in metrics:
            print(f"{prefix}Class Accuracy: {metrics['y_accuracy']:.4f}")
        if 'a_accuracy' in metrics:
            print(f"{prefix}Domain Accuracy: {metrics['a_accuracy']:.4f}")
        
        # Adversarial accuracies (should be low)
        if 'd_adv_accuracy' in metrics:
            print(f"{prefix}Domain Adv. Acc: {metrics['d_adv_accuracy']:.4f} (should be low)")
        if 'y_adv_accuracy' in metrics:
            print(f"{prefix}Class Adv. Acc: {metrics['y_adv_accuracy']:.4f} (should be low)")
        
        # Loss components
        loss_components = ['loss_y_main', 'loss_d_main', 'loss_d_adversarial',
                          'loss_y_adversarial', 'sparsity_loss_zdy', 'sparsity_loss_zy', 'sparsity_loss_zd']
        for component in loss_components:
            if component in metrics:
                print(f"{prefix}{component}: {metrics[component]:.4f}")

        # Show current sparsity weights if available
        if 'sparsity_weight_zdy' in metrics:
            print(f"{prefix}sparsity_weight_zdy: {metrics['sparsity_weight_zdy']:.4f}")
        if 'sparsity_weight_zy_zd' in metrics:
            print(f"{prefix}sparsity_weight_zy_zd: {metrics['sparsity_weight_zy_zd']:.4f}")
    
    def _check_early_stopping(self, val_loss: float, epoch: int, num_epochs: int, 
                             batch_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met based on validation accuracy."""
        current_val_accuracy = batch_metrics.get('y_accuracy', 0)
        
        # For DANN, we also consider disentanglement quality
        # Good disentanglement means low adversarial accuracies
        d_adv_acc = batch_metrics.get('d_adv_accuracy', 1.0)
        y_adv_acc = batch_metrics.get('y_adv_accuracy', 1.0)
        disentanglement_score = 1.0 - (d_adv_acc + y_adv_acc) / 2  # Higher is better
        
        # Combined score: balance main task performance and disentanglement
        combined_score = current_val_accuracy * 0.7 + disentanglement_score * 0.3
        
        # Early stopping based on combined score
        if combined_score > self.best_val_accuracy:
            self.best_val_accuracy = combined_score
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.best_epoch = epoch
            self.patience_counter = 0
            self.best_batch_metrics = batch_metrics.copy()
            
            # Save best model
            model_type = getattr(self.model, 'name', self.model.__class__.__name__.lower())
            model_name = get_model_name(self.args, model_type)
            best_model_path = os.path.join(self.models_dir, f'{model_name}.pt')
            
            torch.save({
                'state_dict': self.best_model_state,
                'training_metrics': self.best_batch_metrics,
                'epoch': self.best_epoch,
                'model_type': model_type,
                'lambda_history': self.lambda_history
            }, best_model_path)
            
            print(f"  New best model saved! (Combined Score: {combined_score:.4f}, "
                  f"Class Acc: {current_val_accuracy:.4f}, Disentanglement: {disentanglement_score:.4f})")
            return False
        else:
            self.patience_counter += 1
            print(f"  No improvement in combined score. Patience: {self.patience_counter}/{self.patience}")
            
            # Use min(10, num_epochs // 2) as minimum epochs requirement
            min_required_epochs = min(10, num_epochs // 2)
            if self.patience_counter >= self.patience and epoch >= min_required_epochs:
                return True
            return False
    
    def _plot_lambda_schedule(self):
        """Plot the gradient reversal lambda schedule."""
        if plt is None:
            print("Matplotlib not available. Saving lambda schedule data as text file instead.")
            # Save lambda history as text file
            schedule_path = os.path.join(self.models_dir, 'lambda_schedule.txt')
            with open(schedule_path, 'w') as f:
                f.write("Epoch\tLambda\n")
                for epoch, lambda_val in enumerate(self.lambda_history, 1):
                    f.write(f"{epoch}\t{lambda_val:.6f}\n")
            print(f"Lambda schedule data saved to {schedule_path}")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.lambda_history) + 1), self.lambda_history, 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Reversal λ')
            plt.title('Gradient Reversal Lambda Schedule')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.models_dir, 'lambda_schedule.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Lambda schedule plot saved to {plot_path}")
        except Exception as e:
            print(f"Failed to create lambda schedule plot: {e}")
            # Fallback to text file
            schedule_path = os.path.join(self.models_dir, 'lambda_schedule.txt')
            with open(schedule_path, 'w') as f:
                f.write("Epoch\tLambda\n")
                for epoch, lambda_val in enumerate(self.lambda_history, 1):
                    f.write(f"{epoch}\t{lambda_val:.6f}\n")
            print(f"Lambda schedule data saved to {schedule_path} instead")
    
    def save_final_model(self, epoch: int):
        """Save the final model state."""
        model_type = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        model_name = get_model_name(self.args, model_type)
        final_model_path = os.path.join(self.models_dir, f'{model_name}_final.pt')
        
        torch.save({
            'state_dict': self.model.state_dict(),
            'training_metrics': self.best_batch_metrics,
            'epoch': epoch,
            'model_type': model_type,
            'lambda_history': self.lambda_history
        }, final_model_path)
        print(f"Final DANN model saved to {final_model_path}")
    
    def load_model(self, model_path: str):
        """Load a saved DANN model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if 'lambda_history' in checkpoint:
            self.lambda_history = checkpoint['lambda_history']
        
        print(f"DANN model loaded from {model_path}")
        print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
        
        if 'training_metrics' in checkpoint:
            metrics = checkpoint['training_metrics']
            print("Training metrics:")
            self._print_metrics(metrics)
        
        return checkpoint
    
    def evaluate_disentanglement(self, test_loader) -> Dict[str, float]:
        """
        Evaluate the quality of disentanglement on test data.
        Lower adversarial accuracies indicate better disentanglement.
        """
        self.model.eval()
        
        total_samples = 0
        y_correct = 0
        a_correct = 0
        d_adv_correct = 0  # Should be low
        y_adv_correct = 0  # Should be low
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Disentanglement"):
                x, y, r = process_batch(batch, self.device, dataset_type=self.dataset)

                # Convert one-hot to indices if necessary (do this BEFORE dann_forward)
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y_true = torch.argmax(y, dim=1)
                else:
                    y_true = y.long()

                if len(r.shape) > 1 and r.shape[1] > 1:
                    r_true = torch.argmax(r, dim=1)
                else:
                    r_true = r.long()

                # Pass y and d for conditional adversarial evaluation
                outputs = self.model.dann_forward(x, y=y_true, d=r_true)
                
                # Main task predictions
                y_pred = outputs['y_pred_main'].argmax(dim=1)
                a_pred = outputs['d_pred_main'].argmax(dim=1)
                
                # Adversarial predictions
                d_pred_adversarial = outputs['d_pred_adversarial'].argmax(dim=1)
                y_pred_adversarial = outputs['y_pred_adversarial'].argmax(dim=1)
                
                # Accumulate correct predictions
                y_correct += (y_pred == y_true).sum().item()
                a_correct += (a_pred == r_true).sum().item()
                d_adv_correct += (d_pred_adversarial == r_true).sum().item()
                y_adv_correct += (y_pred_adversarial == y_true).sum().item()
                total_samples += x.size(0)
        
        # Calculate accuracies
        y_accuracy = y_correct / total_samples
        a_accuracy = a_correct / total_samples
        d_adv_accuracy = d_adv_correct / total_samples
        y_adv_accuracy = y_adv_correct / total_samples
        
        # Disentanglement score (higher is better)
        disentanglement_score = 1.0 - (d_adv_accuracy + y_adv_accuracy) / 2
        
        results = {
            'class_accuracy': y_accuracy,
            'domain_accuracy': a_accuracy,
            'domain_adversarial_accuracy': d_adv_accuracy,  # Should be low
            'class_adversarial_accuracy': y_adv_accuracy,   # Should be low
            'disentanglement_score': disentanglement_score  # Higher is better
        }
        
        print("\nDisentanglement Evaluation Results:")
        print(f"Class Accuracy: {y_accuracy:.4f}")
        print(f"Domain Accuracy: {a_accuracy:.4f}")
        print(f"Domain Adversarial Accuracy: {d_adv_accuracy:.4f} (lower is better)")
        print(f"Class Adversarial Accuracy: {y_adv_accuracy:.4f} (lower is better)")
        print(f"Disentanglement Score: {disentanglement_score:.4f} (higher is better)")

        return results

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
            'train_y_accuracy': self.train_y_acc_history,
            'val_y_accuracy': self.val_y_acc_history,
            'train_domain_accuracy': self.train_d_acc_history,
            'val_domain_accuracy': self.val_d_acc_history
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
        if plt is None:
            print("   ⚠️  Matplotlib not available, skipping plot generation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.epoch_history, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Class accuracy curves
        axes[0, 1].plot(self.epoch_history, self.train_y_acc_history, 'b-', label='Train Y Accuracy', linewidth=2)
        axes[0, 1].plot(self.epoch_history, self.val_y_acc_history, 'r-', label='Val Y Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Class Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Domain accuracy curves
        axes[1, 0].plot(self.epoch_history, self.train_d_acc_history, 'b-', label='Train Domain Acc', linewidth=2)
        axes[1, 0].plot(self.epoch_history, self.val_d_acc_history, 'r-', label='Val Domain Acc', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Domain Accuracy', fontsize=12)
        axes[1, 0].set_title('Domain Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # Combined loss/accuracy plot
        ax_twin = axes[1, 1].twinx()
        axes[1, 1].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        ax_twin.plot(self.epoch_history, self.val_y_acc_history, 'g-', label='Val Y Accuracy', linewidth=2)
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
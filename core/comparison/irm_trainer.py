from core.WILD.trainer import WILDTrainer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys
from core.utils import process_batch, visualize_latent_spaces
import torch.nn.functional as F


class IRMTrainer(WILDTrainer):
    def __init__(self, model, optimizer, device, args, patience=5, scheduler=None):
        super().__init__(model, optimizer, device, args, patience, scheduler=scheduler)
        self.dataset = args.dataset
        self.optimizer = optimizer

        # IRM-specific training history tracking
        self.train_class_loss_history = []
        self.val_class_loss_history = []
        self.train_penalty_history = []
        self.val_penalty_history = []

    def visualize_latent_epoch(self, val_loader, epoch):
        """
        Visualize latent spaces for the current epoch.

        IRM has a single unified feature space, so we use its own
        visualization method for an honest single-space view.
        """
        try:
            dataset_name = self.dataset  # Use dataset from args
            latent_path = os.path.join(
                self.latent_viz_dir,
                f'{dataset_name}_latent_epoch_{epoch+1:03d}.png'
            )

            # IRM has single unified feature space
            # Use model's own visualization method for honest single-space view
            self.model.visualize_latent_space(
                dataloader=val_loader,
                device=self.device,
                save_path=latent_path,
                max_samples=500,
                dataset_type=self.dataset  # Pass dataset type (crmnist or wild)
            )

            print(f"  Latent visualization saved to {latent_path}")
        except Exception as e:
            print(f"  Warning: Could not generate latent visualization for epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()

    def _train_epoch(self, train_loader, epoch, current_beta):
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_penalty = 0
       
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, batch in train_pbar:
            x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)
            
            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(d.shape) > 1 and d.shape[1] > 1:
                d = torch.argmax(d, dim=1)

            # Forward pass and loss computation
            irm_loss, class_loss, penalty = self.model.loss_function(x, y, d)
            # Convert penalty to tensor if needed (avoids warning if already a tensor)
            if not isinstance(penalty, torch.Tensor):
                penalty = torch.tensor(penalty, device=self.device)
            self.optimizer.zero_grad()
            irm_loss.backward()
            self.optimizer.step()

            # Get predictions for accuracy calculation
            logits, _ = self.model.forward(x, y, d)
            y_pred = torch.argmax(logits, dim=1)

            # Update loss totals
            total_loss += irm_loss.item()
            total_class_loss += class_loss.item()
            total_penalty += penalty.item()
            
            # Update accuracy counts
            batch_size = len(y)
            total_samples += batch_size
            correct_y += (y_pred == y).sum().item()

            train_pbar.set_postfix({
                'loss': irm_loss.item(),
                'class_loss': class_loss.item(),
                'penalty': penalty.item(),
                'y_acc': (correct_y / total_samples) * 100
            })

        # Calculate final metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_penalty = total_penalty / len(train_loader)
        y_accuracy = correct_y / total_samples  # Returns 0-1 fraction

        avg_train_metrics = {
            'y_accuracy': y_accuracy,
            'class_loss': avg_class_loss,
            'penalty': avg_penalty
        }

        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader, epoch, current_beta):
        self.model.eval()
        total_loss = 0
        total_class_loss = 0
        total_penalty = 0
        
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0

        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)

                # Convert one-hot encoded labels to class indices
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                if len(d.shape) > 1 and d.shape[1] > 1:
                    d = torch.argmax(d, dim=1)

                # Forward pass and loss computation
                irm_loss, class_loss, penalty = self.model.loss_function(x, y, d)
                # Convert penalty to tensor if needed (avoids warning if already a tensor)
                if not isinstance(penalty, torch.Tensor):
                    penalty = torch.tensor(penalty, device=self.device)

                # Get predictions for accuracy calculation
                logits, _ = self.model.forward(x, y, d)
                y_pred = torch.argmax(logits, dim=1)

                # Update loss totals
                total_loss += irm_loss.item()
                total_class_loss += class_loss.item()
                total_penalty += penalty.item()
                
                # Update accuracy counts
                batch_size = len(y)
                total_samples += batch_size
                correct_y += (y_pred == y).sum().item()

                val_pbar.set_postfix({
                    'loss': irm_loss.item(),
                    'class_loss': class_loss.item(),
                    'penalty': penalty.item(),
                    'y_acc': (correct_y / total_samples) * 100
                })

        # Calculate final metrics
        avg_val_loss = total_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        avg_penalty = total_penalty / len(val_loader)
        y_accuracy = correct_y / total_samples  # Returns 0-1 fraction

        avg_val_metrics = {
            'y_accuracy': y_accuracy,
            'class_loss': avg_class_loss,
            'penalty': avg_penalty
        }

        return avg_val_loss, avg_val_metrics

    def train(self, train_loader, val_loader, num_epochs: int):
        """Override parent train method to track IRM-specific metrics."""
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            if self.beta_annealing:
                trn_current_beta = self.get_current_beta(epoch)
            else:
                trn_current_beta = self.beta_scale

            print(f'Training beta: {trn_current_beta}')
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, current_beta=trn_current_beta)

            # Validation phase
            val_loss, val_metrics = self._validate(val_loader, epoch, current_beta=trn_current_beta)

            # Store training history (including parent class metrics)
            self.epoch_history.append(epoch + 1)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_metrics.get('y_accuracy', 0))
            self.val_acc_history.append(val_metrics.get('y_accuracy', 0))

            # Store IRM-specific metrics
            self.train_class_loss_history.append(train_metrics.get('class_loss', 0))
            self.val_class_loss_history.append(val_metrics.get('class_loss', 0))
            self.train_penalty_history.append(train_metrics.get('penalty', 0))
            self.val_penalty_history.append(val_metrics.get('penalty', 0))

            # Print epoch results
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')

            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')

            # Visualize latent spaces after each epoch
            self.visualize_latent_epoch(val_loader, epoch)

            # Early stopping check
            if self._check_early_stopping(val_loss, epoch, num_epochs, val_metrics):
                break
            self.save_final_model(epoch)

        self.epochs_trained = epoch + 1

        # Save training history and plot curves
        print("\n📊 Saving training history and generating plots...")
        self.save_training_history()
        self.plot_training_curves()

    def save_training_history(self):
        """Save training history to JSON and CSV files including IRM-specific metrics."""
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
            'train_class_loss': self.train_class_loss_history,
            'val_class_loss': self.val_class_loss_history,
            'train_penalty': self.train_penalty_history,
            'val_penalty': self.val_penalty_history
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
        """Plot and save training/validation curves including IRM-specific metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.epoch_history, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss (Total IRM Loss)', fontsize=14, fontweight='bold')
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

        # Class loss curves (IRM-specific)
        axes[1, 0].plot(self.epoch_history, self.train_class_loss_history, 'b-', label='Train Class Loss', linewidth=2)
        axes[1, 0].plot(self.epoch_history, self.val_class_loss_history, 'r-', label='Val Class Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Class Loss', fontsize=12)
        axes[1, 0].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # IRM penalty curves (IRM-specific)
        axes[1, 1].plot(self.epoch_history, self.train_penalty_history, 'b-', label='Train Penalty', linewidth=2)
        axes[1, 1].plot(self.epoch_history, self.val_penalty_history, 'r-', label='Val Penalty', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('IRM Penalty', fontsize=12)
        axes[1, 1].set_title('IRM Invariance Penalty', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot in model-specific subdirectory
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        history_dir = os.path.join(self.args.out, f'{model_name}_training')
        os.makedirs(history_dir, exist_ok=True)
        plot_path = os.path.join(history_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training curves saved to {plot_path}") 
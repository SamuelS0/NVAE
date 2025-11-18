from core.WILD.trainer import WILDTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from core.utils import process_batch, visualize_latent_spaces

class DANNTrainer(WILDTrainer):
    def __init__(self, model, optimizer, device, args, patience=5):
        super().__init__(model, optimizer, device, args, patience)

        # Override models_dir to avoid polluting main models/ directory
        # Use comparison_models/ for trainer checkpoints (separate from run_wild.py's model saves)
        setting = getattr(args, 'setting', 'standard')
        self.models_dir = os.path.join(args.out, 'comparison_models', setting)
        os.makedirs(self.models_dir, exist_ok=True)

        self.dataset = args.dataset
        self.optimizer = optimizer
        self.num_epochs = getattr(args, 'epochs', 100)  # Total epochs for lambda scheduling

        # DANN-specific training history tracking
        self.train_y_loss_history = []
        self.val_y_loss_history = []
        self.train_domain_loss_history = []
        self.val_domain_loss_history = []
        self.train_discriminator_acc_history = []
        self.val_discriminator_acc_history = []

    def visualize_latent_epoch(self, val_loader, epoch):
        """
        Visualize latent spaces for the current epoch.

        Overrides WILDTrainer.visualize_latent_epoch() to use correct dataset type
        and routing based on model architecture (AugmentedDANN vs basic DANN).
        """
        try:
            dataset_name = self.dataset  # Use dataset from args
            latent_path = os.path.join(
                self.latent_viz_dir,
                f'{dataset_name}_latent_epoch_{epoch+1:03d}.png'
            )

            # Detect model architecture and route to appropriate visualization
            if hasattr(self.model, 'extract_features'):
                # AugmentedDANN with partitioned latent spaces
                # Use unified visualize_latent_spaces() for multi-space visualization
                visualize_latent_spaces(
                    model=self.model,
                    dataloader=val_loader,
                    device=self.device,
                    type='dann_augmented',  # Specify augmented variant
                    save_path=latent_path,
                    max_samples=1000,
                    epoch=epoch+1,
                    total_epochs=self.args.epochs
                )
            else:
                # Basic DANN with single unified feature space
                # Use model's own visualization method for honest single-space view
                self.model.visualize_latent_space(
                    dataloader=val_loader,
                    device=self.device,
                    save_path=latent_path,
                    max_samples=1000,
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
        total_y_loss = 0
        total_domain_loss = 0

        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0
        correct_domain = 0

        # Calculate lambda for this epoch using DANN's adaptive scheduling
        # λ(p) = 2/(1+exp(-10p)) - 1, where p = epoch/total_epochs
        current_lambda = self.model.get_lambda(epoch, self.num_epochs)

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"Training (λ={current_lambda:.3f})")

        for batch_idx, batch in train_pbar:
            #x, y, c, d = x.to(self.device), y.to(self.device), c.to(self.device), d.to(self.device)
            x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)
            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(d.shape) > 1 and d.shape[1] > 1:
                d = torch.argmax(d, dim=1)

            # Pass scheduled lambda to model forward pass
            y_logits, domain_logits = self.model(x, y, d, λ=current_lambda)

            y_pred = torch.argmax(y_logits, dim=1)
            domain_pred = torch.argmax(domain_logits, dim=1)

            loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, d)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update loss totals
            total_loss += loss.item()
            total_y_loss += y_loss.item()
            total_domain_loss += domain_loss.item()
            
            # Update accuracy counts
            batch_size = len(y)
            total_samples += batch_size
            correct_y += (y_pred == y).sum().item()
            correct_domain += (domain_pred == d).sum().item()

            train_pbar.set_postfix({
                'loss': loss.item(),
                'y_loss': y_loss.item(),
                'domain_loss': domain_loss.item(),
                'y_acc': (correct_y / total_samples) * 100,
                'domain_acc': (correct_domain / total_samples) * 100
            })

        # Calculate final metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_y_loss = total_y_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        y_accuracy = correct_y / total_samples  # Returns 0-1 fraction
        domain_accuracy = correct_domain / total_samples  # Returns 0-1 fraction

        avg_train_metrics = {
            'y_accuracy': y_accuracy,
            'discriminator_accuracy': domain_accuracy,
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader, epoch, current_beta):
        self.model.eval()
        total_loss = 0
        total_y_loss = 0
        total_domain_loss = 0

        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0
        correct_domain = 0

        # Use current scheduled lambda for validation (consistent with training)
        current_lambda = self.model.get_lambda(epoch, self.num_epochs)

        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                       desc=f"Validating (λ={current_lambda:.3f})")

        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)

                # Convert one-hot encoded labels to class indices
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                if len(d.shape) > 1 and d.shape[1] > 1:
                    d = torch.argmax(d, dim=1)

                # Pass scheduled lambda to model forward pass
                y_logits, domain_logits = self.model(x, y, d, λ=current_lambda)

                y_pred = torch.argmax(y_logits, dim=1)
                domain_pred = torch.argmax(domain_logits, dim=1)

                loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, d)

                # Update loss totals
                total_loss += loss.item()
                total_y_loss += y_loss.item()
                total_domain_loss += domain_loss.item()
                
                # Update accuracy counts
                batch_size = len(y)
                total_samples += batch_size
                correct_y += (y_pred == y).sum().item()
                correct_domain += (domain_pred == d).sum().item()

                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'y_loss': y_loss.item(),
                    'domain_loss': domain_loss.item(),
                    'y_acc': (correct_y / total_samples) * 100,
                    'domain_acc': (correct_domain / total_samples) * 100
                })

        # Calculate final metrics
        avg_val_loss = total_loss / len(val_loader)
        avg_y_loss = total_y_loss / len(val_loader)
        avg_domain_loss = total_domain_loss / len(val_loader)
        y_accuracy = correct_y / total_samples  # Returns 0-1 fraction
        domain_accuracy = correct_domain / total_samples  # Returns 0-1 fraction

        avg_val_metrics = {
            'y_accuracy': y_accuracy,
            'discriminator_accuracy': domain_accuracy,
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_val_loss, avg_val_metrics

    def train(self, train_loader, val_loader, num_epochs: int) -> torch.nn.Module:
        """Train the DANN model with custom metric tracking."""
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, current_beta=1.0)

            # Validation phase
            val_loss, val_metrics = self._validate(val_loader, epoch, current_beta=1.0)

            # Store training history (including parent class metrics)
            self.epoch_history.append(epoch + 1)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_metrics.get('y_accuracy', 0))
            self.val_acc_history.append(val_metrics.get('y_accuracy', 0))

            # Store DANN-specific metrics
            self.train_y_loss_history.append(train_metrics.get('y_loss', 0))
            self.val_y_loss_history.append(val_metrics.get('y_loss', 0))
            self.train_domain_loss_history.append(train_metrics.get('domain_loss', 0))
            self.val_domain_loss_history.append(val_metrics.get('domain_loss', 0))
            self.train_discriminator_acc_history.append(train_metrics.get('discriminator_accuracy', 0))
            self.val_discriminator_acc_history.append(val_metrics.get('discriminator_accuracy', 0))

            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')

            # Try to generate latent visualization
            try:
                self.visualize_latent_epoch(val_loader, epoch)
                print(f'  Latent visualization saved to {self.latent_viz_dir}/crmnist_latent_epoch_{epoch+1:03d}.png')
            except Exception as e:
                print(f'  Warning: Could not generate latent visualization for epoch {epoch+1}: {e}')

            # Save model if it's the best so far
            y_accuracy = val_metrics.get('y_accuracy', 0)
            if y_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = y_accuracy
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
                self.best_batch_metrics = val_metrics

                # Save model checkpoint
                model_path = os.path.join(self.models_dir, f'epoch_{epoch}_final.pt')
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'training_metrics': val_metrics,
                    'val_accuracy': y_accuracy
                }, model_path)

                print(f'  New best model saved! (Validation Accuracy: {y_accuracy:.4f}, Loss: {val_loss:.4f})')
                print(f'  Best model batch metrics: {val_metrics}')

            self.epochs_trained = epoch + 1

        # Save final model
        final_model_path = os.path.join(self.models_dir, f'epoch_{num_epochs}_final.pt')
        torch.save({
            'epoch': num_epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_metrics': self.best_batch_metrics,
            'val_accuracy': self.best_val_accuracy
        }, final_model_path)
        print(f'Final model saved to {final_model_path}')

        # Save training history and generate plots
        self.save_training_history()
        self.plot_training_curves()

        return self.model

    def save_training_history(self):
        """Save training history to JSON and CSV files including DANN-specific metrics."""
        import json
        import pandas as pd

        # Create model-specific subdirectory for training history
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        history_dir = os.path.join(self.args.out, f'{model_name}_training')
        os.makedirs(history_dir, exist_ok=True)

        # Prepare history dictionary - ONLY include metrics that DANN actually tracks
        # (exclude VAE reconstruction metrics which DANN doesn't have)
        history = {
            'epoch': self.epoch_history,
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'train_accuracy': self.train_acc_history,
            'val_accuracy': self.val_acc_history,
            'train_y_loss': self.train_y_loss_history,
            'val_y_loss': self.val_y_loss_history,
            'train_domain_loss': self.train_domain_loss_history,
            'val_domain_loss': self.val_domain_loss_history,
            'train_discriminator_acc': self.train_discriminator_acc_history,
            'val_discriminator_acc': self.val_discriminator_acc_history
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
        """Plot and save training/validation curves including DANN-specific metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.epoch_history, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epoch_history, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss (Total DANN Loss)', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(self.epoch_history, self.train_acc_history, 'b-', label='Train Y Accuracy', linewidth=2)
        axes[0, 1].plot(self.epoch_history, self.val_acc_history, 'r-', label='Val Y Accuracy', linewidth=2)
        axes[0, 1].plot(self.epoch_history, self.train_discriminator_acc_history, 'g--', label='Train Domain Acc', linewidth=2)
        axes[0, 1].plot(self.epoch_history, self.val_discriminator_acc_history, 'orange', linestyle='--', label='Val Domain Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Label and Domain Classification Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Y loss curves (classification loss)
        axes[1, 0].plot(self.epoch_history, self.train_y_loss_history, 'b-', label='Train Y Loss', linewidth=2)
        axes[1, 0].plot(self.epoch_history, self.val_y_loss_history, 'r-', label='Val Y Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Classification Loss', fontsize=12)
        axes[1, 0].set_title('Label Classification Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # Domain loss curves (adversarial loss)
        axes[1, 1].plot(self.epoch_history, self.train_domain_loss_history, 'b-', label='Train Domain Loss', linewidth=2)
        axes[1, 1].plot(self.epoch_history, self.val_domain_loss_history, 'r-', label='Val Domain Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Domain Loss', fontsize=12)
        axes[1, 1].set_title('Adversarial Domain Loss', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        # Add overall title showing number of epochs
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.upper())
        final_epoch = self.epochs_trained
        fig.suptitle(f'{model_name} Training Progress (Epochs 1-{final_epoch})',
                    fontsize=16, fontweight='bold', y=0.998)

        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust to make room for suptitle

        # Save the plot in model-specific subdirectory
        model_name = getattr(self.model, 'name', self.model.__class__.__name__.lower())
        history_dir = os.path.join(self.args.out, f'{model_name}_training')
        os.makedirs(history_dir, exist_ok=True)
        plot_path = os.path.join(history_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training curves saved to {plot_path}")

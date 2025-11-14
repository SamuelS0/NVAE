import torch
from tqdm import tqdm
import os
import sys
# Add project root to Python path
from typing import Dict, Tuple
from core.WILD.utils_wild import (
    visualize_reconstructions,
    select_diverse_sample_batch
)
from core.utils import process_batch, get_model_name, _calculate_metrics, visualize_latent_spaces

class WILDTrainer:
    def __init__(self, model, optimizer, device, args, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.patience = patience
        
        # Early stopping setup
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
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

        # Create output directories
        self.models_dir = os.path.join(args.out, 'models')
                
        self.reconstructions_dir = os.path.join(args.out, 'reconstructions')
        self.max_epochs = args.epochs
        self.beta_annealing = args.beta_annealing
        self.beta_scale = args.beta_scale
        print(f'Beta annealing: {self.beta_annealing}')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reconstructions_dir, exist_ok=True)
        
        # Create latent visualization directory
        self.latent_viz_dir = os.path.join(args.out, 'latent_epoch_viz')
        os.makedirs(self.latent_viz_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs: int):

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

            # Store training history
            self.epoch_history.append(epoch + 1)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_metrics.get('y_accuracy', 0))
            self.val_acc_history.append(val_metrics.get('y_accuracy', 0))
            self.train_recon_history.append(train_metrics.get('recon_mse', 0))
            self.val_recon_history.append(val_metrics.get('recon_mse', 0))

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


    def _train_epoch(self, train_loader, epoch, current_beta) -> Tuple[float, Dict[str, float]]:
        train_loss = 0
        train_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1} [Train]")
        
       
        for batch_idx, batch in train_pbar:
            x, y, hospital_id = process_batch(batch, self.device, dataset_type='wild')
            self.optimizer.zero_grad()

            loss = self.model.loss_function(y, x, hospital_id, current_beta)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            # Calculate metrics for every batch to be consistent with validation
            batch_metrics = _calculate_metrics(self.model, y, x, hospital_id, 'train')
            for k, v in batch_metrics.items():
                train_metrics_sum[k] += v
            
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}
        

        trn_sample_batch = select_diverse_sample_batch(train_loader, data_type = 'train', samples_per_domain=10)
        #save_domain_samples_visualization(*val_sample_batch, epoch+1, domain_samples_dir)
        image_dir = os.path.join(self.reconstructions_dir, f'train_epoch_{epoch}.png')
        visualize_reconstructions(self.model, epoch+1, trn_sample_batch, image_dir, args=self.args)


        return avg_train_loss, avg_train_metrics

    def _validate(self, val_loader, epoch, current_beta) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        val_loss = 0
        val_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                       desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, hospital_id = process_batch(batch, self.device, dataset_type='wild')

                loss = self.model.loss_function(y, x, hospital_id, current_beta)
                val_loss += loss.item()
                
                batch_metrics = _calculate_metrics(self.model, y, x, hospital_id, 'val')
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v
                
                val_pbar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

        #val_sample_batch = select_diverse_sample_batch(val_loader, data_type = self.args.val_type, samples_per_domain=10)
        #save_domain_samples_visualization(*val_sample_batch, epoch+1, domain_samples_dir)
        #image_dir = os.path.join(self.reconstructions_dir, f'val_epoch_{epoch}.png')
        #visualize_reconstructions(self.model, epoch+1, val_sample_batch, image_dir, args=self.args)
        
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
            
            # Get model parameters from training metrics
            model_params = self.args.model_params if hasattr(self.args, 'model_params') else None
            
            torch.save({
                #'params': model_params,
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
        #model_name = get_model_name(self.args, model_type)
        model_name = f'epoch_{epoch}'
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

    
    def get_current_beta(self, epoch):
        """Calculate the current beta value based on the epoch number.
        Beta increases linearly from 0 to 2 over max_epochs."""
        if epoch + 1>= self.max_epochs:
            return self.beta_scale
        return self.beta_scale * (epoch / self.max_epochs)
    
    def visualize_latent_epoch(self, val_loader, epoch):
        """Visualize latent spaces for the current epoch."""
        try:
            latent_path = os.path.join(self.latent_viz_dir, f'wild_latent_epoch_{epoch+1:03d}.png')
            visualize_latent_spaces(
                model=self.model,
                dataloader=val_loader,
                device=self.device,
                type='wild',
                save_path=latent_path,
                max_samples=1000  # Reduce samples for faster visualization
            )
            print(f"  Latent visualization saved to {latent_path}")
        except Exception as e:
            print(f"  Warning: Could not generate latent visualization for epoch {epoch+1}: {e}")

    def save_training_history(self):
        """Save training history to JSON and CSV files."""
        import json
        import pandas as pd
        import matplotlib.pyplot as plt

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
        json_path = os.path.join(self.args.out, 'training_history.json')
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"   ✅ Training history saved to {json_path}")

        # Save as CSV
        csv_path = os.path.join(self.args.out, 'training_history.csv')
        df = pd.DataFrame(history)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Training history saved to {csv_path}")

    def plot_training_curves(self):
        """Plot and save training/validation curves."""
        import matplotlib.pyplot as plt

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

        # Save the plot
        plot_path = os.path.join(self.args.out, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training curves saved to {plot_path}")
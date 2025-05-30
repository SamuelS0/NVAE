import torch
from tqdm import tqdm
import os
from typing import Dict, Tuple
from utils_wild import (
    visualize_reconstructions,
    select_diverse_sample_batch,
    calculate_metrics
)

class WILDTrainer:
    def __init__(self, model, optimizer, device, args, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.patience = patience
        
        # Early stopping setup
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.epochs_trained = 0
        self.best_epoch = 0
        
        # Create output directories
        self.models_dir = os.path.join(args.out, 'models')
        self.reconstructions_dir = os.path.join(args.out, 'reconstructions')
        self.max_epochs = args.epochs
        self.beta_annealing = args.beta_annealing
        self.beta_scale = args.beta_scale
        print(f'Beta annealing: {self.beta_annealing}')
        os.makedirs(self.models_dir, exist_ok=True)
    
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
            val_loss, val_metrics = self._validate(val_loader, epoch, current_beta = 2)
            
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')
                
            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')
            # Early stopping check
            if self._check_early_stopping(val_metrics, epoch, num_epochs):
                break
            self.save_final_model(epoch)    
        self.epochs_trained = epoch + 1


    def _train_epoch(self, train_loader, epoch, current_beta) -> Tuple[float, Dict[str, float]]:
        train_loss = 0
        train_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        num_batches = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (x, y, metadata) in train_pbar:
            hospital_id = metadata[:, 0]
            self.optimizer.zero_grad()
            
            if self.args.cuda:
                x = x.to(self.device)
                y = y.to(self.device)
                hospital_id = hospital_id.to(self.device)
            
            loss, class_y_loss = self.model.loss_function(hospital_id, x, y, current_beta)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                batch_metrics = calculate_metrics(self.model, y, x, hospital_id, args=self.args)
                for k, v in batch_metrics.items():
                    train_metrics_sum[k] += v
                num_batches += 1
            
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / num_batches for k, v in train_metrics_sum.items()}
        

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
        current_beta = self.get_current_beta(epoch)
        with torch.no_grad():
            for batch_idx, (x, y, metadata) in val_pbar:
                hospital_id = metadata[:, 0]
                
                if self.args.cuda:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    hospital_id = hospital_id.to(self.device)
                
                loss, _ = self.model.loss_function(hospital_id, x, y, current_beta)
                val_loss += loss.item()
                
                batch_metrics = calculate_metrics(self.model, y, x, hospital_id, args=self.args)
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v
                
                val_pbar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

        val_sample_batch = select_diverse_sample_batch(val_loader, data_type = self.args.val_type, samples_per_domain=10)
        #save_domain_samples_visualization(*val_sample_batch, epoch+1, domain_samples_dir)
        image_dir = os.path.join(self.reconstructions_dir, f'val_epoch_{epoch}.png')
        visualize_reconstructions(self.model, epoch+1, val_sample_batch, image_dir, args=self.args)
        
        return val_loss, val_metrics

    
    

    def _check_early_stopping(self, val_metrics: Dict[str, float], epoch: int, num_epochs: int) -> bool:
        """Check if early stopping criteria are met based on classification accuracy."""
        val_acc = val_metrics['y_accuracy']
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model_state = self.model.state_dict().copy()
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # Save best model immediately when new best is found
            best_model_path = os.path.join(self.models_dir, 'model_best.pt')
            torch.save(self.best_model_state, best_model_path)
            print(f"  New best model saved! (Validation Accuracy: {self.best_val_acc:.4f})")
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
        final_model_path = os.path.join(self.models_dir, f'model_checkpoint_epoch_{epoch+1}.pt')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    def get_current_beta(self, epoch):
        """Calculate the current beta value based on the epoch number.
        Beta increases linearly from 0 to 2 over max_epochs."""
        if epoch + 1>= self.max_epochs:
            return self.beta_scale
        return self.beta_scale * (epoch / self.max_epochs)
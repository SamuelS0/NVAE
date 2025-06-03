from core.WILD.trainer import WILDTrainer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys
from core.utils import process_batch
import torch.nn.functional as F


class IRMTrainer(WILDTrainer):
    def __init__(self, model, optimizer, device, args, patience=5):
        super().__init__(model, optimizer, device, args, patience)
        self.dataset = args.dataset
        self.optimizer = optimizer
        
        # Training history
        self.train_losses = []
        self.train_penalties = []
        self.val_accuracies = []
        self.test_accuracies = []
        
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
            penalty = torch.tensor(penalty)
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
        y_accuracy = (correct_y / total_samples) * 100

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
        y_accuracy = (correct_y / total_samples) * 100

        avg_val_metrics = {
            'y_accuracy': y_accuracy,
            'class_loss': avg_class_loss,
            'penalty': avg_penalty
        }

        return avg_val_loss, avg_val_metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the model for multiple epochs"""
        print(f"Training IRM model for {num_epochs} epochs...")
        print(f"Penalty weight: {self.model.penalty_weight}")
        print(f"Penalty annealing iterations: {self.model.penalty_anneal_iters}")
        save_every=10
        visualize_every=20
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, self.model.penalty_weight)
            self.train_losses.append(train_loss)
            self.train_penalties.append(train_metrics['penalty'])
            
            # Evaluate
            val_loss, val_metrics = self._validate(val_loader, epoch, self.model.penalty_weight)
            test_acc, test_loss = self.evaluate(self.test_loader, "Test")
            
            self.val_accuracies.append(val_metrics['y_accuracy'])
            self.test_accuracies.append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Penalty: {train_metrics['penalty']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            # Save best model
            if val_metrics['y_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['y_accuracy']
                self.save_model(f'best_irm_model.pth')
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f'irm_model_epoch_{epoch+1}.pth')
            
            # Visualize latent space
            if (epoch + 1) % visualize_every == 0:
                save_path = os.path.join(self.save_dir, f'irm_latent_epoch_{epoch+1}.png')
                self.model.visualize_latent_space(self.val_loader, self.device, save_path)
        
        # Final evaluation and visualization
        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        final_test_acc, _ = self.evaluate(self.test_loader, "Final Test")
        
        # Save final model and create final visualization
        self.save_model('final_irm_model.pth')
        final_viz_path = os.path.join(self.save_dir, 'irm_final_latent_space.png')
        self.model.visualize_latent_space(self.test_loader, self.device, final_viz_path)
        
        # Plot training curves
        self.plot_training_curves()
        
        return best_val_acc, final_test_acc
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_penalties': self.train_penalties,
            'val_accuracies': self.val_accuracies,
            'test_accuracies': self.test_accuracies,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_penalties = checkpoint.get('train_penalties', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.test_accuracies = checkpoint.get('test_accuracies', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Training penalty
        ax2.plot(epochs, self.train_penalties, 'r-', label='IRM Penalty')
        ax2.set_title('IRM Penalty')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Penalty')
        ax2.legend()
        ax2.grid(True)
        
        # Validation accuracy
        ax3.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Test accuracy
        ax4.plot(epochs, self.test_accuracies, 'm-', label='Test Accuracy')
        ax4.set_title('Test Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'irm_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.show()
        plt.close() 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class IRMTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, 
                 lr=1e-3, weight_decay=1e-4, save_dir='./checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training history
        self.train_losses = []
        self.train_penalties = []
        self.val_accuracies = []
        self.test_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_penalty = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, y, c, r) in enumerate(pbar):
            x, y, r = x.to(self.device), y.to(self.device), r.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass and loss computation
            irm_loss, class_loss, penalty = self.model.loss_function(x, y, r)
            
            # Backward pass
            irm_loss.backward()
            self.optimizer.step()
            
            # Accumulate statistics
            batch_size = x.size(0)
            total_loss += class_loss.item() * batch_size
            total_penalty += penalty.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{class_loss.item():.4f}',
                'Penalty': f'{penalty.item():.4f}',
                'IRM_Loss': f'{irm_loss.item():.4f}'
            })
        
        avg_loss = total_loss / total_samples
        avg_penalty = total_penalty / total_samples
        
        return avg_loss, avg_penalty
    
    def evaluate(self, dataloader, split_name="Validation"):
        """Evaluate the model on a dataset"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y, c, r in dataloader:
                x, y, r = x.to(self.device), y.to(self.device), r.to(self.device)
                
                logits, _ = self.model.forward(x, y, r)
                loss = torch.nn.functional.cross_entropy(logits, y)
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        print(f"{split_name} - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        return accuracy, avg_loss
    
    def train(self, num_epochs, save_every=10, visualize_every=20):
        """Train the model for multiple epochs"""
        print(f"Training IRM model for {num_epochs} epochs...")
        print(f"Penalty weight: {self.model.penalty_weight}")
        print(f"Penalty annealing iterations: {self.model.penalty_anneal_iters}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_penalty = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_penalties.append(train_penalty)
            
            # Evaluate
            val_acc, val_loss = self.evaluate(self.val_loader, "Validation")
            test_acc, test_loss = self.evaluate(self.test_loader, "Test")
            
            self.val_accuracies.append(val_acc)
            self.test_accuracies.append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Penalty: {train_penalty:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
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
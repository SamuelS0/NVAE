import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class IRM(nn.Module):
    def __init__(self, spec_data, z_dim, penalty_weight=1e4, penalty_anneal_iters=0):
        super(IRM, self).__init__()
        self.num_y_classes = spec_data['num_y_classes']
        self.num_r_classes = spec_data['num_r_classes'] 
        self.z_dim = z_dim
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.step_count = 0
        self.name = 'irm'

        # Feature extractor - EXACT same architecture as the VAE encoder (qz)
        # This matches the encoder in VAE model.py lines 796-820
        self.feature_extractor = nn.Sequential(
            # Block 1 - matching the qz encoder exactly
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # Block 2
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            # Block 4
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            # Block 5 - Linear projection to z_dim
            nn.Linear(192 * 7 * 7, self.z_dim)
        )

        # Classifier head
        self.classifier = nn.Linear(self.z_dim, self.num_y_classes)

    def forward(self, x, y=None, r=None):
        """
        Forward pass for IRM
        Args:
            x: input images
            y: digit labels (for training)
            r: domain/rotation labels (for environment splitting)
        """
        x = x.float()
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def compute_irm_penalty(self, logits, y):
        """
        Compute the IRM penalty term
        The penalty encourages the optimal classifier to be the same across environments
        """
        # Create a dummy classifier weight of all ones
        dummy_w = torch.ones_like(logits).requires_grad_()
        
        # Compute loss with dummy classifier
        dummy_logits = logits * dummy_w
        dummy_loss = F.cross_entropy(dummy_logits, y)
        
        # Compute gradient of loss w.r.t. dummy classifier
        dummy_grads = torch.autograd.grad(dummy_loss, dummy_w, create_graph=True)[0]
        
        # IRM penalty is the squared norm of these gradients
        penalty = torch.sum(dummy_grads ** 2)
        
        return penalty

    def loss_function(self, x, y, r):
        """
        Compute IRM loss with environment-specific penalties
        Args:
            x: input images
            y: digit labels  
            r: domain/rotation labels (used to split into environments)
        """
        logits, features = self.forward(x, y, r)
        
        # Convert r to environment indices if it's one-hot encoded
        if len(r.shape) > 1:
            env_labels = torch.argmax(r, dim=1)
        else:
            env_labels = r
        
        # Get unique environments
        unique_envs = torch.unique(env_labels)
        
        total_loss = 0.0
        total_penalty = 0.0
        
        # Compute loss and penalty for each environment
        for env in unique_envs:
            env_mask = (env_labels == env)
            if env_mask.sum() == 0:
                continue
                
            env_logits = logits[env_mask]
            env_y = y[env_mask]
            
            # Classification loss for this environment
            env_loss = F.cross_entropy(env_logits, env_y)
            total_loss += env_loss
            
            # IRM penalty for this environment
            if self.step_count >= self.penalty_anneal_iters:
                env_penalty = self.compute_irm_penalty(env_logits, env_y)
                total_penalty += env_penalty
        
        # Current penalty weight (with annealing)
        current_penalty_weight = self.penalty_weight if self.step_count >= self.penalty_anneal_iters else 0.0
        
        # Total IRM loss
        irm_loss = total_loss + current_penalty_weight * total_penalty
        
        self.step_count += 1
        
        return irm_loss, total_loss, total_penalty

    def get_features(self, x):
        """Extract features from the feature extractor"""
        return self.feature_extractor(x)

    def visualize_latent_space(self, dataloader, device, save_path=None, max_samples=5000):
        """
        Visualize the latent space using t-SNE
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples
            device: torch device
            save_path: Optional path to save the visualization
            max_samples: Maximum number of samples to use for visualization
        """
        self.eval()
        features_list = []
        y_list = []
        c_list = []
        r_list = []
        
        sample_count = 0
        
        with torch.no_grad():
            for x, y, c, r in dataloader:
                if sample_count >= max_samples:
                    break
                    
                x = x.to(device)
                features = self.get_features(x)
                
                batch_size = x.size(0)
                remaining_samples = max_samples - sample_count
                samples_to_take = min(batch_size, remaining_samples)
                
                features_list.append(features[:samples_to_take].cpu().numpy())
                y_list.append(y[:samples_to_take].cpu().numpy())
                c_list.append(c[:samples_to_take].cpu().numpy())
                r_list.append(r[:samples_to_take].cpu().numpy())
                
                sample_count += samples_to_take
        
        features = np.concatenate(features_list, axis=0)
        y_labels = np.concatenate(y_list, axis=0)
        c_labels = np.concatenate(c_list, axis=0)
        r_labels = np.concatenate(r_list, axis=0)
        
        # Convert one-hot encoded labels to single dimension
        if len(c_labels.shape) > 1:
            c_labels = np.argmax(c_labels, axis=1)
        if len(r_labels.shape) > 1:
            r_labels = np.argmax(r_labels, axis=1)
        if len(y_labels.shape) > 1:
            y_labels = y_labels.reshape(-1)
        
        print(f"Visualizing {len(features)} samples")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        features_2d = tsne.fit_transform(features)
        
        # Create three subplots: one for task classes, one for colors, one for rotations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot task classes (digits)
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=y_labels, cmap='tab10', alpha=0.7)
        ax1.set_title('IRM: Task Classes (Digits) in Latent Space')
        ax1.legend(*scatter1.legend_elements(), title="Digits")
        
        # Plot colors
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=c_labels, cmap='Set1', alpha=0.7)
        ax2.set_title('IRM: Colors in Latent Space')
        ax2.legend(*scatter2.legend_elements(), title="Colors")
        
        # Plot rotations (domains)
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=r_labels, cmap='Set2', alpha=0.7)
        ax3.set_title('IRM: Domains (Rotations) in Latent Space')
        ax3.legend(*scatter3.legend_elements(), title="Rotations")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IRM latent space visualization saved to {save_path}")
        plt.show()
        plt.close()

    def predict(self, x):
        """Make predictions on input data"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            predictions = torch.softmax(logits, dim=1)
        return predictions

    def get_accuracy(self, dataloader, device):
        """Compute accuracy on a dataset"""
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y, c, r in dataloader:
                x, y = x.to(device), y.to(device)
                logits, _ = self.forward(x)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0 
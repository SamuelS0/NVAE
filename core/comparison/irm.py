import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class IRM(nn.Module):
    def __init__(self, z_dim, num_y_classes, num_r_classes, dataset, penalty_weight=10.0, penalty_anneal_iters=1000):
        super(IRM, self).__init__()
        self.num_y_classes = num_y_classes
        self.num_r_classes = num_r_classes
        self.z_dim = z_dim
        self.dataset = dataset
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.step_count = 0
        self.name = 'irm'

        # Feature extractor - EXACT same architecture as the VAE encoder (qz)
        # This matches the encoder in VAE model.py lines 796-820
        if self.dataset == 'crmnist':
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

        elif self.dataset == 'wild':
            self.feature_extractor = nn.Sequential(
                # Block 1: 96x96x3 -> 48x48x64
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # Block 2: 48x48x64 -> 24x24x128
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # Block 3: 24x24x128 -> 12x12x256
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # Block 4: 12x12x256 -> 6x6x512
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                nn.Flatten(),
                nn.Linear(512 * 6 * 6, self.z_dim)  # Project to z_dim
            )

        # 3-layer MLP with 32 hidden units
        self.classifier = nn.Sequential(
            nn.Linear(self.z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_y_classes)
        )

        # Initialize classifier weights
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        with torch.no_grad():
            self.classifier[0].bias.zero_()

    def get_features(self, x):
        """Extract features from the feature extractor"""
        return self.feature_extractor(x)

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
        num_envs = len(unique_envs)

        total_loss = 0.0
        total_penalty = 0.0
        env_count = 0

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
            env_count += 1

            # IRM penalty for this environment
            # Only compute penalty if gradients are enabled (training mode)
            # Validation runs with torch.no_grad() which breaks compute_irm_penalty()
            if self.step_count >= self.penalty_anneal_iters and torch.is_grad_enabled():
                env_penalty = self.compute_irm_penalty(env_logits, env_y)
                total_penalty += env_penalty

        # CRITICAL FIX: Average over environments instead of summing
        # This ensures the loss scale is independent of the number of environments
        # and matches the scale the penalty_weight was calibrated for
        if env_count > 0:
            avg_loss = total_loss / env_count
            avg_penalty = total_penalty / env_count if self.step_count >= self.penalty_anneal_iters else 0.0
        else:
            avg_loss = total_loss
            avg_penalty = total_penalty

        # Current penalty weight (with annealing)
        current_penalty_weight = self.penalty_weight if self.step_count >= self.penalty_anneal_iters else 0.0

        # Total IRM loss (now properly scaled)
        irm_loss = avg_loss + current_penalty_weight * avg_penalty

        self.step_count += 1

        return irm_loss, avg_loss, avg_penalty

    def visualize_latent_space(self, dataloader, device, save_path=None, max_samples=500, dataset_type="crmnist"):
        """
        Visualize the latent space using t-SNE
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples for CRMNIST or (x, y, metadata) for WILD
            device: torch device
            save_path: Optional path to save the visualization
            max_samples: Maximum number of samples to use for visualization
            dataset_type: Type of dataset ("crmnist" or "wild")
        """
        self.eval()
        features_list = []
        y_list = []
        c_list = []
        r_list = []

        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break

                # Handle different dataset formats
                if dataset_type == "wild":
                    x, y, metadata = batch
                    c = torch.zeros_like(y)  # Dummy color for WILD
                    r = metadata[:, 0]  # Hospital ID as domain
                else:
                    x, y, c, r = batch

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
        # IMPORTANT: Handle grayscale images (all-zeros one-hot) separately
        # to avoid confusion with blue (index 0)
        if len(c_labels.shape) > 1:
            # Detect grayscale: where one-hot sum is 0 (no color applied)
            is_grayscale = np.sum(c_labels, axis=1) == 0
            c_labels = np.argmax(c_labels, axis=1)
            # Assign grayscale a distinct label (7) to separate from blue (0)
            c_labels[is_grayscale] = 7  # 7 = grayscale (no color)
        if len(r_labels.shape) > 1:
            r_labels = np.argmax(r_labels, axis=1)
        if len(y_labels.shape) > 1:
            y_labels = y_labels.reshape(-1)
        
        print(f"Visualizing {len(features)} samples")

        # Apply t-SNE with proper convergence settings
        n_samples = features.shape[0]
        perplexity = min(30, max(5, n_samples // 100))

        # Standardize data before t-SNE (critical for avoiding crescent shapes)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        tsne = TSNE(
            n_components=2,
            random_state=42,
            n_iter=4000,
            perplexity=perplexity,
            learning_rate='auto',  # Let sklearn choose optimal learning rate
            init='pca',
            n_jobs=-1,
            n_iter_without_progress=500
        )
        features_2d = tsne.fit_transform(features_scaled)
        
        # Create three subplots: one for task classes, one for colors, one for rotations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Add overall title
        fig.suptitle('IRM Feature Space t-SNE',
                     fontsize=13, fontweight='bold', y=1.02)

        # Plot task classes (digits)
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=y_labels, cmap='tab10', vmin=0, vmax=9, alpha=0.4)
        ax1.set_title('Colored by Digits (Target Variable)',
                     fontsize=11)
        ax1.set_xlabel('t-SNE Component 1', fontsize=10)
        ax1.set_ylabel('t-SNE Component 2', fontsize=10)
        ax1.legend(*scatter1.legend_elements(), title="Digits", fontsize=9)

        # Plot colors (vmax=7 to include grayscale as label 7)
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=c_labels, cmap='Set1', vmin=0, vmax=7, alpha=0.4)
        ax2.set_title('Colored by Color (Spurious Variable)',
                     fontsize=11)
        ax2.set_xlabel('t-SNE Component 1', fontsize=10)
        ax2.set_ylabel('t-SNE Component 2', fontsize=10)
        ax2.legend(*scatter2.legend_elements(), title="Colors", fontsize=9)

        # Plot rotations (domains)
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=r_labels, cmap='Set2', vmin=0, vmax=5, alpha=0.4)
        ax3.set_title('Colored by Rotation (Domain Variable)',
                     fontsize=11)
        ax3.set_xlabel('t-SNE Component 1', fontsize=10)
        ax3.set_ylabel('t-SNE Component 2', fontsize=10)
        ax3.legend(*scatter3.legend_elements(), title="Rotations", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    def get_accuracy(self, dataloader, device, dataset_type="crmnist"):
        """Compute accuracy on a dataset
        Args:
            dataloader: DataLoader containing batches
            device: torch device
            dataset_type: Type of dataset ("crmnist" or "wild")
        """
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                # Handle different dataset formats
                if dataset_type == "wild":
                    x, y, metadata = batch
                else:
                    x, y, c, r = batch

                x, y = x.to(device), y.to(device)
                logits, _ = self.forward(x)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0 
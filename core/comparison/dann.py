import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        # multiply incoming gradient by -λ
        return grad_output.neg() * ctx.λ, None

def grad_reverse(x, λ=1.0):
    return GradReverse.apply(x, λ)

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_r_classes, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_r_classes),  # Output num_r_classes
        )

    def forward(self, features, λ):
        # Apply gradient reversal
        reversed_features = grad_reverse(features, λ)
        return self.net(reversed_features)


class DANN(nn.Module):
    def __init__(self, z_dim, num_y_classes, num_r_classes, dataset):
        super(DANN, self).__init__()
        self.num_y_classes = num_y_classes
        self.num_r_classes = num_r_classes
        self.z_dim = z_dim
        self.name = 'dann'
        self.dataset = dataset
        
        if self.dataset == 'crmnist':
            # Feature extractor matching our VAE encoder architecture
            self.feature_extractor = nn.Sequential(
                # Block 1
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
                nn.Linear(192 * 7 * 7, self.z_dim)  # Project to z_dim directly
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
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")

        self.domain_discriminator = DomainDiscriminator(self.z_dim, self.num_r_classes)
        
        # Changed: Classifier now matches VAE's qy architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_y_classes)
        )
        
        # Initialize classifier weights to match VAE's qy initialization
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        with torch.no_grad():
            self.classifier[0].bias.zero_()

    def forward(self, x, y, r, λ=1.0):
        # Ensure input is float type
        x = x.float()
        features = self.feature_extractor(x)
        
        # Apply gradient reversal with adaptive λ in domain discriminator
        domain_predictions = self.domain_discriminator(features, λ)
        
        # Get raw logits for classification
        y_logits = self.classifier(features)

        return y_logits, domain_predictions
    
    def loss_function(self, y_predictions, domain_predictions, y, r, λ=1.0):
        """
        Compute the DANN loss as a sum of classification and domain losses.
        Args:
            y_predictions: Raw logits for digit classification
            domain_predictions: Predictions for domain classification
            y: True digit labels
            r: True domain labels
            λ: DEPRECATED - Lambda is now handled by the GRL during backpropagation.
               This parameter is kept for backward compatibility but is not used.
        Returns:
            total_loss: Sum of classification and domain losses
            y_loss: Classification loss
            domain_loss: Domain classification loss

        Note: The adversarial aspect is handled by the Gradient Reversal Layer (GRL)
        which multiplies domain gradients by -λ during backpropagation. Therefore,
        we do NOT multiply domain_loss by λ here to avoid double application.
        """
        # Classification loss (digit prediction) - use raw logits
        y_loss = F.cross_entropy(y_predictions, y)

        # Domain loss (rotation prediction)
        domain_loss = F.cross_entropy(domain_predictions, r)

        # Total loss: minimize classification error + domain loss
        # The GRL handles the adversarial aspect via -λ gradient multiplication
        # FIXED: Removed λ multiplication to avoid double lambda application
        total_loss = y_loss + domain_loss

        return total_loss, y_loss, domain_loss

    def get_features(self, x):
        """Extract features from the feature extractor"""
        return self.feature_extractor(x)

    def visualize_latent_space(self, dataloader, device, save_path=None, max_samples=500, dataset_type="crmnist"):
        """
        Visualize the latent space using t-SNE with balanced sampling
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples for CRMNIST or (x, y, metadata) for WILD
            device: torch device
            save_path: Optional path to save the visualization
            max_samples: Maximum number of samples to use for visualization
            dataset_type: Type of dataset ("crmnist" or "wild")
        """
        # Import the reusable sampling function
        from core.utils import balanced_sample_for_visualization

        # Use the generic balanced sampling function
        features_dict, labels_dict, sampling_stats = balanced_sample_for_visualization(
            model=self,
            dataloader=dataloader,
            device=device,
            model_type="dann",
            max_samples=max_samples,
            target_samples_per_combination=50,
            dataset_type=dataset_type
        )
        
        # Extract the features (DANN uses the same features for all spaces)
        features = features_dict['features'].numpy()
        y_labels = labels_dict['y'].numpy()
        c_labels = labels_dict['c'].numpy()
        r_labels = labels_dict['r'].numpy()
        
        print(f"Final features shape: {features.shape}")
        
        # Convert one-hot encoded labels to single dimension (if needed)
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
        
        # Ensure all labels are 1D arrays
        if len(y_labels.shape) > 1:
            y_labels = y_labels.reshape(-1)
        
        # Verify dimensions match
        assert len(y_labels) == len(features), f"Label dimension mismatch: {len(y_labels)} vs {len(features)}"
        assert len(c_labels) == len(features), f"Color label dimension mismatch: {len(c_labels)} vs {len(features)}"
        assert len(r_labels) == len(features), f"Rotation label dimension mismatch: {len(r_labels)} vs {len(features)}"
        
        # Apply t-SNE with proper convergence settings
        n_samples = features.shape[0]
        perplexity = min(30, max(5, n_samples // 100))
        learning_rate = max(50, n_samples / 48)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            n_iter=4000,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init='pca',
            n_jobs=-1,
            n_iter_without_progress=500
        )
        features_2d = tsne.fit_transform(features)
        
        # Create three subplots: one for task classes, one for colors, one for rotations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Add overall title
        fig.suptitle('DANN Feature Space t-SNE',
                     fontsize=13, fontweight='bold', y=1.02)

        # Plot task classes
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=y_labels, cmap='tab10', vmin=0, vmax=9, alpha=0.4)
        ax1.set_title('Colored by Digits (Target Variable)',
                     fontsize=11)
        ax1.set_xlabel('t-SNE Component 1', fontsize=10)
        ax1.set_ylabel('t-SNE Component 2', fontsize=10)
        ax1.legend(*scatter1.legend_elements(), title="Digits", fontsize=9)

        # Plot colors (vmax=7 to include grayscale as label 7)
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=c_labels, cmap='tab10', vmin=0, vmax=7, alpha=0.4)
        ax2.set_title('Colored by Color (Spurious Variable)',
                     fontsize=11)
        ax2.set_xlabel('t-SNE Component 1', fontsize=10)
        ax2.set_ylabel('t-SNE Component 2', fontsize=10)
        ax2.legend(*scatter2.legend_elements(), title="Colors", fontsize=9)

        # Plot rotations
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=r_labels, cmap='tab10', vmin=0, vmax=5, alpha=0.4)
        ax3.set_title('Colored by Rotation (Domain Variable)',
                     fontsize=11)
        ax3.set_xlabel('t-SNE Component 1', fontsize=10)
        ax3.set_ylabel('t-SNE Component 2', fontsize=10)
        ax3.legend(*scatter3.legend_elements(), title="Rotations", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path)
            print(f"Latent space visualization saved to {save_path}")
        plt.close()

    @staticmethod
    def get_lambda(epoch, max_epochs):
        """
        Adaptive lambda scheduling as described in DANN paper.
        Lambda starts at 0 and gradually increases to 1.
        """
        p = epoch / max_epochs
        return 2. / (1. + np.exp(-10 * p)) - 1

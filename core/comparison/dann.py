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
        # reverse gradient here
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
        Compute the DANN loss as a weighted sum of classification and domain losses.
        Args:
            y_predictions: Raw logits for digit classification
            domain_predictions: Predictions for domain classification
            y: True digit labels
            r: True domain labels
            λ: Weight for domain loss (should be scheduled during training)
        Returns:
            total_loss: Weighted sum of classification and domain losses
            y_loss: Classification loss
            domain_loss: Domain classification loss
        """
        # Classification loss (digit prediction) - use raw logits
        y_loss = F.cross_entropy(y_predictions, y)
        
        # Domain loss (rotation prediction)
        domain_loss = F.cross_entropy(domain_predictions, r)
        
        # Total loss: minimize classification error + domain adversarial loss
        # The gradient reversal in domain discriminator handles the adversarial aspect
        total_loss = y_loss + λ * domain_loss
        
        return total_loss, y_loss, domain_loss

    def get_features(self, x):
        """Extract features from the feature extractor"""
        return self.feature_extractor(x)

    def visualize_latent_space(self, dataloader, device, save_path=None):
        """
        Visualize the latent space using t-SNE
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples where:
                x: input images
                y: digit labels (0-9)
                c: color labels (one-hot encoded)
                r: rotation/domain labels (one-hot encoded)
            device: torch device
            save_path: Optional path to save the visualization
        """
        self.eval()
        features_list = []
        y_list = []
        c_list = []
        r_list = []
        
        with torch.no_grad():
            for x, y, c, r in dataloader:
                x = x.to(device)
                features = self.get_features(x)
                features_list.append(features.cpu().numpy())
                y_list.append(y.cpu().numpy())
                c_list.append(c.cpu().numpy())
                r_list.append(r.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        y_labels = np.concatenate(y_list, axis=0)
        c_labels = np.concatenate(c_list, axis=0)
        r_labels = np.concatenate(r_list, axis=0)
        
        # Convert one-hot encoded labels to single dimension
        if len(c_labels.shape) > 1:
            c_labels = np.argmax(c_labels, axis=1)
        if len(r_labels.shape) > 1:
            r_labels = np.argmax(r_labels, axis=1)
        
        # Ensure all labels are 1D arrays
        if len(y_labels.shape) > 1:
            y_labels = y_labels.reshape(-1)
        
        # Verify dimensions match
        assert len(y_labels) == len(features), f"Label dimension mismatch: {len(y_labels)} vs {len(features)}"
        assert len(c_labels) == len(features), f"Color label dimension mismatch: {len(c_labels)} vs {len(features)}"
        assert len(r_labels) == len(features), f"Rotation label dimension mismatch: {len(r_labels)} vs {len(features)}"
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create three subplots: one for task classes, one for colors, one for rotations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot task classes
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=y_labels, cmap='tab10')
        ax1.set_title('Task Classes in Latent Space')
        ax1.legend(*scatter1.legend_elements(), title="Classes")
        
        # Plot colors
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=c_labels, cmap='tab10')
        ax2.set_title('Colors in Latent Space')
        ax2.legend(*scatter2.legend_elements(), title="Colors")
        
        # Plot rotations
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=r_labels, cmap='tab10')
        ax3.set_title('Rotations in Latent Space')
        ax3.legend(*scatter3.legend_elements(), title="Rotations")
        
        plt.tight_layout()
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

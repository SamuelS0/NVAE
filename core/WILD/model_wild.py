import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

#consider implementing a sparsity penalty



class NModule(nn.Module):

    def __init__(self):
        super().__init__()
    
    @classmethod
    def set_class_variables(           
            cls,
            class_map,
            zy_dim,
            zx_dim,
            zay_dim,
            za_dim,
            y_dim,
            a_dim,
            ):
        cls.class_map = class_map
        cls.zy_dim = zy_dim
        cls.zx_dim = zx_dim
        cls.zay_dim = zay_dim
        cls.za_dim = za_dim
        cls.y_dim = y_dim
        cls.a_dim = a_dim


class VAE(NModule):
    def __init__(
            self,
            class_map,
            zy_dim = 16,
            zx_dim= 16,
            zay_dim = 16,
            za_dim=16,
            y_dim=2,    # Changed to 2 for binary classification (tumor/normal)
            a_dim=5,    # 5 hospitals in Camelyon17
            in_channels = 3,  # RGB images
            out_channels=64,  # Increased channels for more complex features
            kernel=3,
            stride=1,
            padding=1,
            beta_1=1,
            beta_2=1,
            beta_3=1,
            beta_4=1,
            alpha_1=1,
            alpha_2=1,
            recon_weight=1,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            resolution='high',  # Add resolution parameter
            model='vae',  # Add DIVA mode parameter
            l1_lambda_zy=0.0,
            l1_lambda_zx=0.0,
            l1_lambda_zay=0.0,
            l1_lambda_za=0.0
            ):
        
        super().__init__()
        if model == 'vae':
            self.diva = False
        elif model == 'diva':
            self.diva = True
        print('data resolution', resolution)
        print('diva', self.diva)
        self.class_map = class_map
        self.device = device
        self.y_dim = y_dim
        self.a_dim = a_dim

        if self.diva:
            assert zay_dim % 3 == 0, "zay_dim must be divisible by 3"
            extra_dim = zay_dim // 3
            
            self.zy_dim = zy_dim + extra_dim
            self.zx_dim = zx_dim + extra_dim
            self.za_dim = za_dim + extra_dim
            self.zay_dim = 0
            self.z_y_combined_dim = self.zy_dim
            self.z_a_combined_dim = self.za_dim
            self.z_total_dim = self.zx_dim + self.zy_dim + self.za_dim
        else:
            self.zy_dim = zy_dim
            self.zx_dim = zx_dim
            self.zay_dim = zay_dim
            self.za_dim = za_dim
            self.z_y_combined_dim = zy_dim + zay_dim
            self.z_a_combined_dim = za_dim + zay_dim
            self.z_total_dim = zx_dim + zy_dim + zay_dim + za_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.beta_1, self.beta_2, self.beta_3, self.beta_4  = beta_1, beta_2, beta_3, beta_4
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.recon_weight = recon_weight

        # L1 sparsity penalty weights
        self.l1_lambda_zy = l1_lambda_zy
        self.l1_lambda_zx = l1_lambda_zx
        self.l1_lambda_zay = l1_lambda_zay
        self.l1_lambda_za = l1_lambda_za

        self.zy_index_range = [0, self.zy_dim]
        self.zx_index_range = [self.zy_dim, self.zy_dim + self.zx_dim]
        if self.diva:
            self.zay_index_range = None
            self.za_index_range = [self.zy_dim + self.zx_dim, self.z_total_dim]
        else:
            self.zay_index_range = [self.zy_dim + self.zx_dim, self.zy_dim + self.zx_dim + self.zay_dim]
            self.za_index_range = [self.zy_dim + self.zx_dim + self.zay_dim, self.z_total_dim]
        
        if resolution == 'high':  # 448x448
            self.qz = qz(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim,
                         in_channels, out_channels, kernel, stride, padding, self.diva).to(self.device)
            self.px = px(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim,
                         in_channels, out_channels, kernel, stride, padding, self.diva).to(self.device)
        else:  # 96x96
            self.qz = qz_new(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim,
                             in_channels, out_channels, kernel, stride, padding, self.diva).to(self.device)
            self.px = px_new(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim,
                             in_channels, out_channels, kernel, stride, padding, self.diva).to(self.device)

        self.qy = qy(self.y_dim, self.zy_dim, self.zay_dim, self.diva).to(self.device)
        self.qa = qa(self.a_dim, self.za_dim, self.zay_dim, self.diva).to(self.device)

        self.pzy = pzy(self.y_dim, self.zy_dim).to(self.device)
        if self.diva:
            self.pzay = None
        else:
            self.pzay = pzay(self.y_dim, self.a_dim, self.zay_dim).to(self.device)
        self.pza = pza(self.a_dim, self.za_dim).to(self.device)

    def forward(self, y, x, a):
        # Encode
        qz_loc, qz_scale = self.qz(x)

        # Encoder Reparameterization
        qz = dist.Normal(qz_loc, qz_scale)
        z = qz.rsample()
        
        # Print shapes for debugging during first forward pass
        if not hasattr(self, '_shape_checked'):
            print(f"Full latent vector z shape: {z.shape}")
            print(f"Index ranges - zy: {self.zy_index_range}, zx: {self.zx_index_range}, " 
                  f"zay: {self.zay_index_range}, za: {self.za_index_range}")

        zy = z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        zx = z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            zay = None
        else:
            zay = z[:, self.zay_index_range[0]:self.zay_index_range[1]]
        za = z[:, self.za_index_range[0]:self.za_index_range[1]]
        
        if not hasattr(self, '_shape_checked'):
            zay_shape = "None" if zay is None else zay.shape
            print(f"Split latent vectors - zy: {zy.shape}, zx: {zx.shape}, zay: {zay_shape}, za: {za.shape}")
            self._shape_checked = True

        if self.diva:
            assert z.shape[1] == self.zy_dim + self.zx_dim + self.za_dim
        else:
            assert z.shape[1] == self.zy_dim + self.zx_dim + self.zay_dim + self.za_dim

        # Decoder Reconstruction
        x_recon = self.px(zy, zx, zay, za)

        # Priors
        pzy_loc, pzy_scale = self.pzy(y)
        # Move standard normal prior to same device as model
        pzx_loc = torch.zeros_like(zx).to(self.device)  # Use zeros_like to match zx tensor
        pzx_scale = torch.ones_like(zx).to(self.device) # Use ones_like to match zx tensor
        pza_loc, pza_scale = self.pza(a)
        
        if self.diva:
            pzay_loc, pzay_scale = None, None
        else:
            pzay_loc, pzay_scale = self.pzay(y, a)

        # Priors Reparameterization
        pzy = dist.Normal(pzy_loc, pzy_scale)
        pzx = dist.Normal(pzx_loc, pzx_scale)
        pza = dist.Normal(pza_loc, pza_scale)
        if self.diva:
            pzay = None
        else:
            pzay = dist.Normal(pzay_loc, pzay_scale)

        # Auxiliary
        y_hat = self.qy(zy, zay)
        a_hat = self.qa(za, zay)

        return x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za #zs were sampled from q


    def loss_function(self, y, x, a, current_beta = 1):

        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = self.forward(y, x, a)
        log_prob_z = qz.log_prob(z)
        log_prob_zy = log_prob_z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        log_prob_zx = log_prob_z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            log_prob_zay = torch.zeros_like(log_prob_zy)  # Create zero tensor for DIVA mode
        else:
            log_prob_zay = log_prob_z[:, self.zay_index_range[0]:self.zay_index_range[1]]
        log_prob_za = log_prob_z[:, self.za_index_range[0]:self.za_index_range[1]]

        x_recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # Get current beta value for annealing
        #print('current_beta: ', current_beta)
        #current_beta = 2
        
        kl_zy = torch.sum(log_prob_zy - pzy.log_prob(zy))
        kl_zx = torch.sum(log_prob_zx - pzx.log_prob(zx))
        if self.diva:
            kl_zay = torch.zeros_like(kl_zy)
        else:
            kl_zay = torch.sum(log_prob_zay - pzay.log_prob(zay))
        kl_za = torch.sum(log_prob_za - pza.log_prob(za))

        # Handle y label - could be index or one-hot
        if len(y.shape) > 1 and y.shape[1] > 1:
            # One-hot encoded
            _, y_target = y.max(dim=1)
        else:
            # Index tensor
            y_target = y.long()
        
        # Handle a label - could be index or one-hot
        if len(a.shape) > 1 and a.shape[1] > 1:
            # One-hot encoded
            _, a_target = a.max(dim=1)
        else:
            # Index tensor
            a_target = a.long()
            
        y_cross_entropy = F.cross_entropy(y_hat, y_target, reduction='sum')
        a_cross_entropy = F.cross_entropy(a_hat, a_target, reduction='sum')

        # L1 sparsity penalties
        l1_penalty = 0.0
        if self.l1_lambda_zy > 0:
            l1_penalty += self.l1_lambda_zy * torch.mean(torch.abs(zy))
        if self.l1_lambda_zx > 0:
            l1_penalty += self.l1_lambda_zx * torch.mean(torch.abs(zx))
        if not self.diva and self.l1_lambda_zay > 0:
            l1_penalty += self.l1_lambda_zay * torch.mean(torch.abs(zay))
        if self.l1_lambda_za > 0:
            l1_penalty += self.l1_lambda_za * torch.mean(torch.abs(za))

        # Calculate positive loss (removing negative sign)
        # In VAEs, we want to minimize the reconstruction loss + KL divergence
        # Apply current_beta to all KL terms
        total_loss = self.recon_weight * x_recon_loss + \
                    current_beta * (self.beta_1 * kl_zy + self.beta_2 * kl_zx + \
                    self.beta_3 * kl_zay + self.beta_4 * kl_za) + \
                    self.alpha_1 * y_cross_entropy + self.alpha_2 * a_cross_entropy + \
                    l1_penalty

        return total_loss
    
    def classifier(self, x):
        with torch.no_grad():
            z_loc, z_scale = self.qz(x)
            zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
            zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
            if self.diva:
                zay = None  # Keep consistent with forward method
            else:
                zay = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
            za = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
            y_hat = self.qy(zy, zay)
            return y_hat

    def generate(self, y, a, num_samples=10, device=None, latent_space = 'full'):
        """
        Generate samples conditionally based on class label and hospital.
        
        Args:
            y: Class label (0: Normal, 1: Tumor)
            a: Hospital ID (0-4)
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            generated_images: Tensor of generated images
            y_labels: Corresponding class labels
        """
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device
            
            # Handle y input
            if isinstance(y, int):
                y = torch.tensor([y]).repeat(num_samples).to(device)
            elif isinstance(y, torch.Tensor) and len(y.shape) == 0:
                y = y.repeat(num_samples).to(device)
            else:
                y = y.to(device)
            
            # Handle a input (hospital ID)
            if isinstance(a, int):
                a = torch.tensor([a]).repeat(num_samples).to(device)
            elif isinstance(a, torch.Tensor) and len(a.shape) == 0:
                a = a.repeat(num_samples).to(device)
            else:
                a = a.to(device)
            
            batch_size = y.shape[0]
            
            # Convert labels to one-hot
            y_one_hot = F.one_hot(y, self.y_dim).float()
            a_one_hot = F.one_hot(a, self.a_dim).float()
            
            # Sample from all priors
            pzy_loc, pzy_scale = self.pzy(y)
            pzy = dist.Normal(pzy_loc, pzy_scale)
            zy = pzy.sample()
            
            # Content latent variable
            zx = torch.randn(batch_size, self.zx_dim).to(device)
            
            # Hospital-specific latent variable
            pza_loc, pza_scale = self.pza(a)
            pza = dist.Normal(pza_loc, pza_scale)
            za = pza.sample()
            
            if self.diva:
                zay = None
            else:
                # Class-hospital interaction latent variable
                pzay_loc, pzay_scale = self.pzay(y, a)
                pzay = dist.Normal(pzay_loc, pzay_scale)
                zay = pzay.sample()
            
            generated_images_full = self.px(zy, zx, zay, za)
        
        return generated_images_full, y

    def visualize_latent_spaces(self, dataloader, device, save_path=None, max_samples=5000):
        """
        Visualize the latent spaces using t-SNE with balanced sampling for WILD dataset
        Args:
            dataloader: DataLoader containing (x, y, metadata) tuples where:
                x: input images
                y: label (0=normal, 1=tumor)
                metadata: contains hospital ID in metadata[:, 0]
            device: torch device
            save_path: Optional path to save the visualization
            max_samples: Maximum number of samples to use for visualization
        """
        # Import the reusable sampling function
        from core.utils import balanced_sample_for_visualization
        import numpy as np
        
        # Use the generic balanced sampling function with WILD dataset type
        features_dict, labels_dict, sampling_stats = balanced_sample_for_visualization(
            model=self,
            dataloader=dataloader,
            device=device,
            model_type="nvae" if not self.diva else "diva",
            max_samples=max_samples,
            target_samples_per_combination=50,
            dataset_type="wild"
        )
        
        # Extract the latent spaces
        zy_features = features_dict['zy'].numpy()
        za_features = features_dict['za'].numpy()
        zx_features = features_dict['zx'].numpy()
        y_labels = labels_dict['y'].numpy()
        r_labels = labels_dict['r'].numpy()  # Hospital IDs
        
        if not self.diva:
            zay_features = features_dict['zay'].numpy()
        
        print(f"Final features shapes: zy={zy_features.shape}, za={za_features.shape}, zx={zx_features.shape}")
        if not self.diva:
            print(f"zay={zay_features.shape}")
        
        # Convert labels to single dimension (if needed)
        if len(y_labels.shape) > 1:
            y_labels = np.argmax(y_labels, axis=1)
        if len(r_labels.shape) > 1:
            r_labels = np.argmax(r_labels, axis=1)
        
        # Ensure all labels are 1D arrays
        y_labels = y_labels.reshape(-1)
        r_labels = r_labels.reshape(-1)
        
        # Verify dimensions match
        assert len(y_labels) == len(zy_features), f"Label dimension mismatch: {len(y_labels)} vs {len(zy_features)}"
        assert len(r_labels) == len(zy_features), f"Hospital label dimension mismatch: {len(r_labels)} vs {len(zy_features)}"
        
        # Apply t-SNE to each latent space
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        tsne = TSNE(n_components=2, random_state=42)
        
        # Determine number of subplots based on DIVA mode
        if self.diva:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns for DIVA
            spaces = ['zy', 'za', 'zx']
            features_list = [zy_features, za_features, zx_features]
        else:
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 2 rows, 4 columns for full VAE
            spaces = ['zy', 'za', 'zay', 'zx']
            features_list = [zy_features, za_features, zay_features, zx_features]
        
        # Create visualizations for each latent space
        for i, (space_name, features) in enumerate(zip(spaces, features_list)):
            features_2d = tsne.fit_transform(features)
            
            # Plot colored by labels (tumor/normal) in first row
            scatter1 = axes[0, i].scatter(features_2d[:, 0], features_2d[:, 1], c=y_labels, cmap='RdYlBu', alpha=0.6)
            axes[0, i].set_title(f'{space_name} - Colored by Label\n(Blue=Normal, Red=Tumor)')
            axes[0, i].legend(*scatter1.legend_elements(), title="Labels", loc='upper right')
            
            # Plot colored by hospital in second row
            scatter2 = axes[1, i].scatter(features_2d[:, 0], features_2d[:, 1], c=r_labels, cmap='tab10', alpha=0.6)
            axes[1, i].set_title(f'{space_name} - Colored by Hospital')
            axes[1, i].legend(*scatter2.legend_elements(), title="Hospitals", loc='upper right')
        
        # Add overall title
        model_type = "DIVA" if self.diva else "VAE"
        plt.suptitle(f'WILD {model_type} Latent Space Visualization (t-SNE)', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent space visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
        # Print sampling statistics
        print("\nSampling Statistics:")
        for combination, count in sampling_stats.items():
            if isinstance(combination, tuple) and len(combination) == 2:
                label, hospital = combination
                label_name = "Normal" if label == 0 else "Tumor"
                print(f"  {label_name} + Hospital {hospital}: {count} samples")

    def is_diva_mode(self):
        """Check if the model is in DIVA mode"""
        return self.diva

    def visualize_latent_spaces_diva(self, dataloader, device, save_path=None, max_samples=5000):
        """Convenience method for DIVA-specific visualization"""
        if not self.diva:
            print("Warning: Model is not in DIVA mode. Using regular VAE visualization.")
        return self.visualize_latent_spaces(dataloader, device, save_path, max_samples)


#Encoder module 
class qz(NModule):
    def __init__(self,
                 zy_dim,
                 zx_dim,
                 zay_dim,
                 za_dim,
                 z_total_dim, 
                 in_channels=3,
                 out_channels=64,
                 kernel=3,
                 stride=1,
                 padding=1,
                 diva=False):
        super().__init__()
        self.diva = diva
        
        # Modified encoder for 448x448x3 input
        self.encoder = nn.Sequential(
            # 448x448x3 -> 224x224x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 224x224x64 -> 112x112x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 112x112x128 -> 56x56x256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 56x56x256 -> 28x28x512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 28x28x512 -> 14x14x512
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 14x14x512 -> 7x7x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten()
        )
        
        # Adjust linear layer dimensions: 7x7x512 = 25088
        self.loc = nn.Linear(512*7*7, z_total_dim)
        self.scale = nn.Linear(512*7*7, z_total_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = F.relu(h)
        z_loc = self.loc(h)
        z_scale = F.softplus(self.scale(h)) + 1e-6
        
        # Add numerical stability checks
        z_loc = torch.clamp(z_loc, min=-10, max=10)  # Prevent extreme values
        z_scale = torch.clamp(z_scale, min=1e-6, max=10)  # Ensure positive and bounded
        
        return z_loc, z_scale

# New encoder module for 96x96x3 images
class qz_new(NModule):
    def __init__(self,
                 zy_dim,
                 zx_dim,
                 zay_dim,
                 za_dim,
                 z_total_dim, 
                 in_channels=3,
                 out_channels=64,
                 kernel=3,
                 stride=1,
                 padding=1,
                 diva=False):
        super().__init__()
        

        # Enhanced encoder architecture with proper downsampling
        self.encoder = nn.Sequential(
            # 96x96x3 -> 48x48x64
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # 48x48x64 -> 24x24x128
            nn.Conv2d(out_channels, out_channels*2, kernel, stride, padding),
            nn.BatchNorm2d(out_channels*2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # 24x24x128 -> 12x12x256
            nn.Conv2d(out_channels*2, out_channels*4, kernel, stride, padding),
            nn.BatchNorm2d(out_channels*4),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # 12x12x256 -> 6x6x512
            nn.Conv2d(out_channels*4, out_channels*8, kernel, stride, padding),
            nn.BatchNorm2d(out_channels*8),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size of flattened features: 6x6x512 = 18432
        self.flat_features = out_channels*8 * 6 * 6
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flat_features, z_total_dim)
        self.fc_var = nn.Linear(self.flat_features, z_total_dim)
        
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get mean and variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        # Add numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)  # Prevent extreme log variance
        std = torch.exp(0.5 * log_var) + 1e-6  # Ensure positive std
        
        # Clamp outputs for stability
        mu = torch.clamp(mu, min=-10, max=10)
        std = torch.clamp(std, min=1e-6, max=10)
        
        return mu, std

#Auxiliarry classifiers
class qy(NModule):
    def __init__(self, y_dim, zy_dim, zay_dim, diva=False):
        super().__init__()
        self.diva = diva
        
        # Fix dimension calculation based on DIVA mode
        if self.diva:
            self.zay_dim = 0
            self.z_combined_dim = zy_dim  # When diva=True, we only use zy_dim
        else:
            self.zay_dim = zay_dim
            self.z_combined_dim = zy_dim + zay_dim

        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, 64)  # Add intermediate layer
        self.fc3 = nn.Linear(64, y_dim)  # Output layer

        # Initialize all layers properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc3.bias.zero_()

    def forward(self, zy, zay):
        if self.diva:
            # When diva=True, we only use zy (zay should be None or zeros)
            if zay is not None:
                assert torch.all(zay == 0), "zay should be None or all zeros in DIVA mode"
            z_combined = zy
        else:
            # When diva=False, concatenate zy and zay
            assert zay is not None, "zay cannot be None in non-DIVA mode"
            z_combined = torch.cat((zy, zay), -1)
        
        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        # Return raw logits for numerical stability with cross_entropy loss
        return logits
    
    def get_probabilities(self, zy, zay):
        """Get probability distributions (for interpretation/inference)"""
        logits = self.forward(zy, zay)
        return torch.softmax(logits, dim=1)


class qa(NModule):
    def __init__(self, a_dim, za_dim, zay_dim, diva=False):
        super().__init__()
        self.diva = diva
        
        # Fix dimension calculation based on DIVA mode
        if self.diva:
            self.zay_dim = 0
            self.z_combined_dim = za_dim  # When diva=True, we only use za_dim
        else:
            self.zay_dim = zay_dim
            self.z_combined_dim = za_dim + zay_dim

        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, 64)  # Add intermediate layer like qy
        self.fc3 = nn.Linear(64, a_dim)  # Output layer

        # Initialize all layers properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc3.bias.zero_()

    def forward(self, za, zay):
        if self.diva:
            # When diva=True, we only use za (zay should be None or zeros)
            if zay is not None:
                assert torch.all(zay == 0), "zay should be None or all zeros in DIVA mode"
            z_combined = za
        else:
            # When diva=False, concatenate za and zay
            assert zay is not None, "zay cannot be None in non-DIVA mode"
            z_combined = torch.cat((za, zay), -1)

        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        # Return raw logits for numerical stability with cross_entropy loss
        return logits
    
    def get_probabilities(self, za, zay):
        """Get probability distributions (for interpretation/inference)"""
        logits = self.forward(za, zay)
        return torch.softmax(logits, dim=1)



# # individual encoder modules : as in DIVA  !! Include in paper the significance of new model and being able to use one nn instead of multiple !!
# class qzx(NModule):
#     def __init__(
#             self,
#             zx_dim=128, 
#             in_channels=1, 
#             out_channels=32, 
#             kernel=3, 
#             stride=1, 
#             padding=0):
        
#         super().__init__()

#         self.zx_dim = zx_dim
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel, stride, padding), 
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel, stride, padding),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels*2, kernel, stride, padding),
#             nn.ReLU(),
#             nn.Conv2d(out_channels*2, out_channels*4, kernel, stride, padding),
#             nn.ReLU(),
#             nn.Conv2d(out_channels*4, out_channels*8, kernel, stride, padding),
#             nn.ReLU(),
#             View((-1, out_channels*8))
#         )

# class qza(NModule):
#     def __init__(
#             self,
#             za_dim=32,
#             in_channels=1,
#             out_channels=32,
#             kernel=3,
#             stride=1,
#             padding=0):
#         super().__init__()
#         self.zx_za = za_dim
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel, stride, padding), 
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel, stride, padding),
#             nn.ReLU(),
#             nn.Conv2d(out_channels*2, out_channels*4, kernel, stride, padding),
#             nn.ReLU(),
#             nn.Conv2d(out_channels*4, out_channels*8, kernel, padding),
#             nn.ReLU(),
#             View((-1, out_channels*8)))
        
# class qzay(NModule):
#     def __init__(self, zay_dim=16):
#         super().__init__()

# class qzy(NModule):
#     def __init__(self, zy_dim=32):
#         super().__init__()
        

#Decoder modules.
class px(NModule): 
    def __init__(
            self,
            zy_dim,
            zx_dim,
            zay_dim,
            za_dim,
            z_total_dim,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=1,
            diva=False):
        super().__init__()
        self.diva = diva
        
        # Store parameters
        self.z_total_dim = zy_dim + zx_dim + zay_dim + za_dim
        
        # Adjust initial projection to match encoder's final dimensions
        self.fc1 = nn.Linear(z_total_dim, 512 * 7 * 7)  # Match encoder's final spatial dimensions
        
        # Modified decoder for 448x448 output
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ReLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ReLU(),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # 224x224 -> 448x448
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, zy, zx, zay, za):
        # Combine latent variables
        if self.diva:
            assert zay is None, "zay should be None in DIVA mode"
            z_combined = torch.cat((zy, zx, za), -1)
        else:
            z_combined = torch.cat((zy, zx, zay, za), -1)
        
        combined_features = z_combined
        
        # Decode
        h = self.fc1(combined_features)
        h = h.view(-1, 512, 7, 7)  # Changed to 7x7 to match encoder
        h = self.decoder(h)
        return h

# New decoder module for 96x96x3 output
class px_new(NModule):
    def __init__(
            self,
            zy_dim,
            zx_dim,
            zay_dim,
            za_dim,
            z_total_dim,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=1,
            diva=False):
        super().__init__()
        self.diva = diva
        # Calculate initial size for decoder
        self.initial_size = 6  # Starting from 6x6
        self.initial_channels = out_channels*8  # 512 channels
        
        # Project latent space to initial feature map
        self.fc = nn.Linear(z_total_dim, self.initial_channels * self.initial_size * self.initial_size)
        
        # Enhanced decoder architecture with proper upsampling
        self.decoder = nn.Sequential(
            # 6x6x512 -> 12x12x256
            nn.ConvTranspose2d(self.initial_channels, out_channels*4, 4, 2, 1),
            nn.BatchNorm2d(out_channels*4),
            nn.LeakyReLU(0.2),
            
            # 12x12x256 -> 24x24x128
            nn.ConvTranspose2d(out_channels*4, out_channels*2, 4, 2, 1),
            nn.BatchNorm2d(out_channels*2),
            nn.LeakyReLU(0.2),
            
            # 24x24x128 -> 48x48x64
            nn.ConvTranspose2d(out_channels*2, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            
            # 48x48x64 -> 96x96x3
            nn.ConvTranspose2d(out_channels, in_channels, 4, 2, 1),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        
    def forward(self, zy, zx, zay, za):
        # Combine latent variables
        if self.diva:
            # When diva=True, we only use za
            z_combined = torch.cat([zy, zx, za], dim=1)
        else:
            # When diva=False, concatenate za and zay
            z_combined = torch.cat([zy, zx, zay, za], dim=1)
        combined_features = z_combined
        # Project to initial feature map
        x = self.fc(z_combined)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        
        # Decode
        x = self.decoder(x)
        
        return x

# y is input, y is one hot encoded as vector of dimension y_dim.
class pzy(NModule):
    def __init__(self, y_dim, zy_dim):
        super().__init__()
        self.y_dim = y_dim
        self.zy_dim = zy_dim

        # Simplify architecture to avoid BatchNorm issues
        self.decoder = nn.Sequential(
            nn.Linear(y_dim, zy_dim),
            nn.ReLU()
        )

        self.loc = nn.Linear(zy_dim, zy_dim)
        self.scale = nn.Sequential(
            nn.Linear(zy_dim, zy_dim),
            nn.Softplus()
        )
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.decoder[0].weight)
        torch.nn.init.zeros_(self.decoder[0].bias)
        torch.nn.init.xavier_uniform_(self.loc.weight)
        torch.nn.init.zeros_(self.loc.bias)
        torch.nn.init.xavier_uniform_(self.scale[0].weight)
        torch.nn.init.zeros_(self.scale[0].bias)

    def forward(self, y):
        # Handle case where y is already one-hot encoded or needs to be converted
        if len(y.shape) == 1 or y.shape[1] == 1:
            # If y is a vector of indices
            valid_y_indices = torch.clamp(y.long(), 0, self.y_dim-1)
            y_one_hot = F.one_hot(valid_y_indices, self.y_dim).float()
        else:
            # If y is already one-hot encoded
            y_one_hot = y.float()
            
        hidden = self.decoder(y_one_hot)
        zy_loc = self.loc(hidden)
        zy_scale = self.scale(hidden) + 1e-6  # Add epsilon for numerical stability
        
        # Ensure scale is positive
        zy_scale = torch.clamp(zy_scale, min=1e-6)

        return zy_loc, zy_scale

#need to handle encoding (one hot?) of a and y?
class pzay(NModule):
    def __init__(self, y_dim, a_dim, zay_dim):
        super().__init__()
        self.y_dim = y_dim
        self.a_dim = a_dim
        self.zay_dim = zay_dim
        self.in_dim = y_dim + a_dim  # This should be 10 + 5 = 15

        # Simplify architecture to avoid BatchNorm issues
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, zay_dim),
            nn.ReLU()
        )

        self.loc = nn.Linear(zay_dim, zay_dim)
        self.scale = nn.Sequential(
            nn.Linear(zay_dim, zay_dim),
            nn.Softplus()
        )
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.decoder[0].weight)
        torch.nn.init.zeros_(self.decoder[0].bias)
        torch.nn.init.xavier_uniform_(self.loc.weight)
        torch.nn.init.zeros_(self.loc.bias)
        torch.nn.init.xavier_uniform_(self.scale[0].weight)
        torch.nn.init.zeros_(self.scale[0].bias)

    def forward(self, y, a):
        # Handle case where y and a are already one-hot encoded or need to be converted
        if len(y.shape) == 1 or y.shape[1] == 1:
            # If y is a vector of indices
            valid_y_indices = torch.clamp(y.long(), 0, self.y_dim-1)
            y_one_hot = F.one_hot(valid_y_indices, self.y_dim).float()
        else:
            # If y is already one-hot encoded
            y_one_hot = y.float()
            
        if len(a.shape) == 1 or a.shape[1] == 1:
            # If a is a vector of indices
            valid_a_indices = torch.clamp(a.long(), 0, self.a_dim-1)
            a_one_hot = F.one_hot(valid_a_indices, self.a_dim).float()
        else:
            # If a is already one-hot encoded
            a_one_hot = a.float()

        h = self.decoder(torch.cat((y_one_hot, a_one_hot), dim=1))
        zy_loc = self.loc(h)
        zy_scale = self.scale(h) + 1e-6  # Add epsilon for numerical stability
        
        # Ensure scale is positive
        zy_scale = torch.clamp(zy_scale, min=1e-6)

        return zy_loc, zy_scale


class pza(NModule):
    def __init__(self, a_dim, za_dim):
        super().__init__()
        self.a_dim = a_dim
        
        # Simplify the decoder to reduce chances of numerical instability
        self.decoder = nn.Sequential(
            nn.Linear(a_dim, za_dim),
            nn.ReLU()
        )
        
        self.loc = nn.Linear(za_dim, za_dim)
        self.scale = nn.Sequential(
            nn.Linear(za_dim, za_dim),
            nn.Softplus()
        )
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.decoder[0].weight)
        torch.nn.init.zeros_(self.decoder[0].bias)
        torch.nn.init.xavier_uniform_(self.loc.weight)
        torch.nn.init.zeros_(self.loc.bias)
        torch.nn.init.xavier_uniform_(self.scale[0].weight)
        torch.nn.init.zeros_(self.scale[0].bias)

    def forward(self, a):
        # Handle case where a is already one-hot encoded or needs to be converted
        if len(a.shape) == 1 or a.shape[1] == 1:
            # If a is a vector of indices
            valid_a_indices = torch.clamp(a.long(), 0, self.a_dim-1)
            a_one_hot = F.one_hot(valid_a_indices, self.a_dim).float()
        else:
            # If a is already one-hot encoded
            a_one_hot = a.float()
        
        # Forward pass
        hidden = self.decoder(a_one_hot)
        za_loc = self.loc(hidden)
        za_scale = self.scale(hidden) + 1e-6  # Add epsilon for numerical stability
        
        # Ensure scale is positive
        za_scale = torch.clamp(za_scale, min=1e-6)
        
        return za_loc, za_scale

'''class VAE_LowRes(NModule):
    def __init__(
            self,
            class_map,
            zy_dim = 16,
            zx_dim = 16,
            zay_dim = 16,
            za_dim = 16,
            y_dim = 2,    # Binary classification (tumor/normal)
            a_dim = 5,    # 5 hospitals
            in_channels = 3,  # RGB images
            out_channels = 64,
            kernel = 3,
            stride = 1,
            padding = 1,
            beta_1 = 1,
            beta_2 = 1,
            beta_3 = 1,
            beta_4 = 1,
            alpha_1 = 1,
            alpha_2 = 1, 
            recon_weight = 1,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ):
        super().__init__()

        self.class_map = class_map
        self.device = device
        self.y_dim = y_dim
        self.a_dim = a_dim

        # Latent dimensions
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.za_dim = za_dim
        self.zay_dim = zay_dim
        self.z_y_combined_dim = zy_dim + zay_dim
        self.z_a_combined_dim = za_dim + zay_dim
        self.z_total_dim = zx_dim + zy_dim + zay_dim + za_dim

        # Architecture parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        # Loss weights
        self.beta_1, self.beta_2, self.beta_3, self.beta_4 = beta_1, beta_2, beta_3, beta_4
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.recon_weight = recon_weight

        # Latent space indexing
        self.zy_index_range = [0, zy_dim]
        self.zx_index_range = [zy_dim, zy_dim + zx_dim]
        self.zay_index_range = [zy_dim + zx_dim, zy_dim + zx_dim + zay_dim]
        self.za_index_range = [zy_dim + zx_dim + zay_dim, self.z_total_dim]

        # Initialize network components
        # Use qz_new and px_new for 96x96 images
        self.qz = qz_new(zy_dim, zx_dim, zay_dim, za_dim, self.z_total_dim,
                      in_channels, out_channels, kernel, stride, padding).to(self.device)
        
        self.px = px_new(zy_dim, zx_dim, zay_dim, za_dim, self.z_total_dim,
                      in_channels, out_channels, kernel).to(self.device)

        # These components remain the same as they handle latent space operations
        self.qy = qy(y_dim, zy_dim, zay_dim).to(self.device)
        self.qa = qa(a_dim, za_dim, zay_dim).to(self.device)
        self.pzy = pzy(y_dim, zy_dim).to(self.device)
        self.pzay = pzay(y_dim, a_dim, zay_dim).to(self.device)
        self.pza = pza(a_dim, za_dim).to(self.device)

    # Forward, loss_function, classifier, and generate methods remain the same
    # as they operate on the latent space which is unchanged
    def forward(self, y, x, a):
        # Encode
        qz_loc, qz_scale = self.qz(x)

        # Encoder Reparameterization
        qz = dist.Normal(qz_loc, qz_scale)
        z = qz.rsample()
        
        # Print shapes for debugging during first forward pass
        if not hasattr(self, '_shape_checked'):
            print(f"Full latent vector z shape: {z.shape}")
            print(f"Index ranges - zy: {self.zy_index_range}, zx: {self.zx_index_range}, " 
                  f"zay: {self.zay_index_range}, za: {self.za_index_range}")

        zy = z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        zx = z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        zay = z[:, self.zay_index_range[0]:self.zay_index_range[1]]
        za = z[:, self.za_index_range[0]:self.za_index_range[1]]
        
        if not hasattr(self, '_shape_checked'):
            print(f"Split latent vectors - zy: {zy.shape}, zx: {zx.shape}, zay: {zay.shape}, za: {za.shape}")
            self._shape_checked = True

        # Decoder Reconstruction
        x_recon = self.px(zy, zx, zay, za)

        # Priors
        pzy_loc, pzy_scale = self.pzy(y)
        # Move standard normal prior to same device as model
        pzx_loc = torch.zeros_like(zx).to(self.device)  # Use zeros_like to match zx tensor
        pzx_scale = torch.ones_like(zx).to(self.device) # Use ones_like to match zx tensor
        pza_loc, pza_scale = self.pza(a)
        pzay_loc, pzay_scale = self.pzay(y, a)

        # Priors Reparameterization

        pzy = dist.Normal(pzy_loc, pzy_scale)
        pzx = dist.Normal(pzx_loc, pzx_scale)
        pza = dist.Normal(pza_loc, pza_scale)
        pzay = dist.Normal(pzay_loc, pzay_scale)

        

        # Auxiliary
        y_hat = self.qy(zy, zay)
        a_hat = self.qa(za, zay)

        return x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za #zs were sampled from q

    def loss_function(self, y, x, a, current_beta = 1):

        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = self.forward(y, x, a)
        log_prob_z = qz.log_prob(z)
        log_prob_zy = log_prob_z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        log_prob_zx = log_prob_z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            log_prob_zay = torch.zeros_like(log_prob_zy)  # Create zero tensor for DIVA mode
        else:
            log_prob_zay = log_prob_z[:, self.zay_index_range[0]:self.zay_index_range[1]]
        log_prob_za = log_prob_z[:, self.za_index_range[0]:self.za_index_range[1]]

        x_recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # Get current beta value for annealing
        #print('current_beta: ', current_beta)
        #current_beta = 2
        
        kl_zy = torch.sum(log_prob_zy - pzy.log_prob(zy))
        kl_zx = torch.sum(log_prob_zx - pzx.log_prob(zx))
        if self.diva:
            kl_zay = torch.zeros_like(kl_zy)
        else:
            kl_zay = torch.sum(log_prob_zay - pzay.log_prob(zay))
        kl_za = torch.sum(log_prob_za - pza.log_prob(za))

        # Handle y label - could be index or one-hot
        if len(y.shape) > 1 and y.shape[1] > 1:
            # One-hot encoded
            _, y_target = y.max(dim=1)
        else:
            # Index tensor
            y_target = y.long()
        
        # Handle a label - could be index or one-hot
        if len(a.shape) > 1 and a.shape[1] > 1:
            # One-hot encoded
            _, a_target = a.max(dim=1)
        else:
            # Index tensor
            a_target = a.long()
            
        y_cross_entropy = F.cross_entropy(y_hat, y_target, reduction='sum')
        a_cross_entropy = F.cross_entropy(a_hat, a_target, reduction='sum')

        # L1 sparsity penalties
        l1_penalty = 0.0
        if self.l1_lambda_zy > 0:
            l1_penalty += self.l1_lambda_zy * torch.mean(torch.abs(zy))
        if self.l1_lambda_zx > 0:
            l1_penalty += self.l1_lambda_zx * torch.mean(torch.abs(zx))
        if not self.diva and self.l1_lambda_zay > 0:
            l1_penalty += self.l1_lambda_zay * torch.mean(torch.abs(zay))
        if self.l1_lambda_za > 0:
            l1_penalty += self.l1_lambda_za * torch.mean(torch.abs(za))

        # Calculate positive loss (removing negative sign)
        # In VAEs, we want to minimize the reconstruction loss + KL divergence
        # Apply current_beta to all KL terms
        total_loss = self.recon_weight * x_recon_loss + \
                    current_beta * (self.beta_1 * kl_zy + self.beta_2 * kl_zx + \
                    self.beta_3 * kl_zay + self.beta_4 * kl_za) + \
                    self.alpha_1 * y_cross_entropy + self.alpha_2 * a_cross_entropy + \
                    l1_penalty

        return total_loss
    
    def classifier(self, x):
        with torch.no_grad():
            z_loc, z_scale = self.qz(x)
            zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
            zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
            if self.diva:
                zay = None  # Keep consistent with forward method
            else:
                zay = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
            za = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
            y_hat = self.qy(zy, zay)
            return y_hat

    def generate(self, y, a, num_samples=10, device=None, latent_space='full'):
        """
        Generate samples conditionally based on class label and hospital.
        
        Args:
            y: Class label (0: Normal, 1: Tumor)
            a: Hospital ID (0-4)
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            generated_images: Tensor of generated images
            y_labels: Corresponding class labels
        """
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device
            
            # Handle y input
            if isinstance(y, int):
                y = torch.tensor([y]).repeat(num_samples).to(device)
            elif isinstance(y, torch.Tensor) and len(y.shape) == 0:
                y = y.repeat(num_samples).to(device)
            else:
                y = y.to(device)
            
            # Handle a input (hospital ID)
            if isinstance(a, int):
                a = torch.tensor([a]).repeat(num_samples).to(device)
            elif isinstance(a, torch.Tensor) and len(a.shape) == 0:
                a = a.repeat(num_samples).to(device)
            else:
                a = a.to(device)
            
            batch_size = y.shape[0]
            
            # Convert labels to one-hot
            y_one_hot = F.one_hot(y, self.y_dim).float()
            a_one_hot = F.one_hot(a, self.a_dim).float()
            
            # Sample from all priors
            pzy_loc, pzy_scale = self.pzy(y)
            pzy = dist.Normal(pzy_loc, pzy_scale)
            zy = pzy.sample()
            
            # Content latent variable
            zx = torch.randn(batch_size, self.zx_dim).to(device)
            
            # Hospital-specific latent variable
            pza_loc, pza_scale = self.pza(a)
            pza = dist.Normal(pza_loc, pza_scale)
            za = pza.sample()
            
            if self.diva:
                zay = None
            else:
                # Class-hospital interaction latent variable
                pzay_loc, pzay_scale = self.pzay(y, a)
                pzay = dist.Normal(pzay_loc, pzay_scale)
                zay = pzay.sample()
            
            generated_images_full = self.px(zy, zx, zay, za)
        
        return generated_images_full, y'''
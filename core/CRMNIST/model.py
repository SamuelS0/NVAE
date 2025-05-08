import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from core.utils import kl_divergence
from torchsummary import summary
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

# make latent dims divisible by 3
class VAE(NModule):
    def __init__(
            self,
            class_map,
            zy_dim = 12,
            zx_dim= 12,
            zay_dim = 12,
            za_dim=12,
            y_dim=10,   # 10 for MNIST's 10 digit classes (0-9)
            a_dim=5,    # Updated to 5 for exactly 5 attribute classes (0-4)
            in_channels = 3,
            out_channels=32,
            kernel=3,
            stride=1,
            padding=0,
            beta_1=1,
            beta_2=1,
            beta_3=1,
            beta_4=1,
            alpha_1=1,
            alpha_2=2,
            diva=False
            ):
        
        super().__init__()

        self.class_map = class_map

        if diva:
            assert zay_dim % 3 == 0, "zay_dim must be divisible by 3"

            extra_dim = zay_dim // 3

            self.zy_dim = zy_dim + extra_dim
            self.zx_dim = zx_dim + extra_dim
            self.za_dim = za_dim + extra_dim
            self.zay_dim = None
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

        self.y_dim = y_dim
        self.a_dim = a_dim


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.beta_1, self.beta_2, self.beta_3, self.beta_4  = beta_1, beta_2, beta_3, beta_4
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.diva = diva

        self.zy_index_range = [0, self.zy_dim]
        self.zx_index_range = [self.zy_dim, self.zy_dim + self.zx_dim]
        if diva:
            self.zay_index_range = None
            self.za_index_range = [self.zy_dim + self.zx_dim, self.z_total_dim]
        else:
            self.zay_index_range = [self.zy_dim + self.zx_dim, self.zy_dim + self.zx_dim + self.zay_dim]
            self.za_index_range = [self.zy_dim + self.zx_dim + self.zay_dim, self.z_total_dim]
        
        self.qz = qz(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim,
                     in_channels, out_channels, kernel, stride, padding, diva)
        self.qy = qy(self.y_dim, self.zy_dim, self.zay_dim, diva)
        self.qa = qa(self.a_dim, self.za_dim, self.zay_dim, diva)

        self.pzy = pzy(self.y_dim, self.zy_dim)
        
        if diva:
            self.pzay = None
        else:
            self.pzay = pzay(self.y_dim, self.a_dim, self.zay_dim)

        self.pza = pza(self.a_dim, self.za_dim)
        self.px = px(self.zy_dim, self.zx_dim, self.zay_dim, self.za_dim, self.z_total_dim, in_channels, out_channels, kernel, diva=diva)

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
        device = next(self.parameters()).device  # Get the device of the model
        pzx_loc = torch.zeros(zx.size(0), self.zx_dim, device=device)
        pzx_scale = torch.ones(zx.size(0), self.zx_dim, device=device)
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



    def loss_function(self, y, x, a):
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

        # Calculate positive loss (removing negative sign)
        # In VAEs, we want to minimize the reconstruction loss + KL divergence
        total_loss = x_recon_loss + self.beta_1 * kl_zy + self.beta_2 * kl_zx \
                    + self.beta_3 * kl_zay + self.beta_4 * kl_za \
                    + self.alpha_1 * y_cross_entropy + self.alpha_2 * a_cross_entropy

        return total_loss
    
    def classifier(self, x):
        with torch.no_grad():
            z_loc, z_scale = self.qz(x)
            zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
            zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
            if self.diva:
                zay = torch.zeros_like(zy)
            else:
                zay = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
            za = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
            y_hat = self.qy(zy, zay)
            return y_hat

    def generate(self, y, a, num_samples=10, device=None):
        """
        Generate samples conditionally based on class label y and domain label a.
        
        Args:
            y: Class label (digit 0-9) to generate. Can be an integer, a tensor of integers,
               or None (which will generate one sample per class)
            num_samples: Number of samples to generate per class
            device: Device to generate samples on
                
        Returns:
            generated_images: Tensor of generated images
            y_labels: Corresponding class labels
        """
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device

            if isinstance(y, int):
                y = torch.tensor([y]).repeat(num_samples).to(device)
            elif isinstance(y, torch.Tensor) and len(y.shape) == 0:
                y = y.repeat(num_samples).to(device)
            else:
                y = y.to(device)
        
            batch_size = y.shape[0]
            
            # Sample from prior distributions
            # For zy, use the learned prior conditioned on y
            y_one_hot = F.one_hot(y, self.y_dim).float()
            pzy_loc, pzy_scale = self.pzy(y)
            pzy = dist.Normal(pzy_loc, pzy_scale)
            zy = pzy.sample()
            
            # For zx, sample from standard normal
            zx = torch.randn(batch_size, self.zx_dim).to(device)
            
            # sample for za and zay
            # can change this to control specific attributes


            if isinstance(a, int):
                a = torch.tensor([a]).repeat(num_samples).to(device)
            elif isinstance(a, torch.Tensor) and len(a.shape) == 0:
                a = a.repeat(num_samples).to(device)
            else:
                a = a.to(device)
            
            # Sample from prior distributions
            # For zy, use the learned prior conditioned on a
            a_one_hot = F.one_hot(a, self.a_dim).float()
            pza_loc, pza_scale = self.pza(a)
            pza = dist.Normal(pza_loc, pza_scale)
            za = pza.sample()

            a = torch.randint(0, self.a_dim, (batch_size,)).to(device)
            a_one_hot = F.one_hot(a, self.a_dim).float()

            if self.diva:
                zay = None
            else:
                pza_loc, pza_scale = self.pza(a)
                pza = dist.Normal(pza_loc, pza_scale)
                za = pza.sample()
            
            if self.diva:
                pzay_loc, pzay_scale = None, None
            else:
                pzay_loc, pzay_scale = self.pzay(y, a)
                pzay = dist.Normal(pzay_loc, pzay_scale)
                zay = pzay.sample()
            
            # Generate images using the decoder with conditional y, a
            generated_images = self.px(zy, zx, zay, za)
            
            return generated_images, y

    def visualize_latent_spaces(self, dataloader, device, save_path=None):
        """
        Visualize all latent spaces using t-SNE
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples where:
                x: input images
                y: digit labels (0-9)
                c: color labels (one-hot encoded, 5 dimensions)
                r: rotation/domain labels (one-hot encoded, 5 dimensions)
            device: torch device
            save_path: Optional path to save the visualization
        """
        self.eval()
        zy_list, za_list, zay_list, zx_list = [], [], [], []
        y_list, c_list, r_list = [], [], []
        
        with torch.no_grad():
            for x, y, c, r in dataloader:
                x = x.to(device)
  
                # Get latent representations
                z_loc, _ = self.qz(x)
                zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
                zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
                if self.diva:
                    zay = None
                else:
                    zay = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
                za = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
                
                # Store latent vectors and labels
                zy_list.append(zy.cpu().numpy())
                za_list.append(za.cpu().numpy())
                if not self.diva:
                    zay_list.append(zay.cpu().numpy())
                zx_list.append(zx.cpu().numpy())
                y_list.append(y.cpu().numpy())
                c_list.append(c.cpu().numpy())
                r_list.append(r.cpu().numpy())
        
        # Convert to numpy arrays
        zy = np.concatenate(zy_list, axis=0)
        za = np.concatenate(za_list, axis=0)
        if not self.diva:
            zay = np.concatenate(zay_list, axis=0)
        zx = np.concatenate(zx_list, axis=0)
        y_labels = np.concatenate(y_list, axis=0)
        c_labels = np.concatenate(c_list, axis=0)
        r_labels = np.concatenate(r_list, axis=0)
        
        # Print raw rotation labels for debugging
        print("\nRaw rotation labels shape:", r_labels.shape)
        print("Sample of raw rotation labels:", r_labels[:5])
        print("Unique values in raw rotation labels:", np.unique(r_labels, axis=0))
        
        # Convert one-hot encoded labels to single dimension
        if len(c_labels.shape) > 1:
            c_labels = np.argmax(c_labels, axis=1)
        if len(r_labels.shape) > 1:
            # Ensure we're working with the correct axis
            print("\nBefore argmax - r_labels shape:", r_labels.shape)
            print("Sample of r_labels before argmax:", r_labels[:5])
            r_labels = np.argmax(r_labels, axis=1)
            print("\nAfter argmax - r_labels shape:", r_labels.shape)
            print("Sample of r_labels after argmax:", r_labels[:5])
        
        # Print distribution of labels

        print("\nUnique values in labels:")
        print("Unique digit labels:", np.unique(y_labels))
        print("Unique color labels:", np.unique(c_labels))
        print("Unique rotation labels:", np.unique(r_labels))
        
        # Ensure all labels are 1D arrays
        if len(y_labels.shape) > 1:
            y_labels = y_labels.reshape(-1)
        
        # Verify dimensions match
        assert len(y_labels) == len(zy), f"Label dimension mismatch: {len(y_labels)} vs {len(zy)}"
        assert len(c_labels) == len(za), f"Color label dimension mismatch: {len(c_labels)} vs {len(za)}"
        assert len(r_labels) == len(za), f"Rotation label dimension mismatch: {len(r_labels)} vs {len(za)}"
        
        # Apply t-SNE to each latent space
        tsne = TSNE(n_components=2, random_state=42)
        zy_2d = tsne.fit_transform(zy)
        za_2d = tsne.fit_transform(za)
        if not self.diva:
            zay_2d = tsne.fit_transform(zay)
        zx_2d = tsne.fit_transform(zx)
        
        # Create figure with 2 rows (digit labels and rotation labels) and 4 columns (zy, za, zay, zx)
        if self.diva:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 3 columns for DIVA mode
        else:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Define rotation angles for legend
        rotation_angles = ['0°', '15°', '30°', '45°', '60°']
        
        # Plot each latent space
        latent_spaces = [
            (zy_2d, 'Label-specific (zy)'),
            (za_2d, 'Domain-specific (za)'),
            (zay_2d if not self.diva else None, 'Domain-Label (zay)'),
            (zx_2d, 'Residual (zx)')
        ]
        
        for col, (space_2d, title) in enumerate(latent_spaces):
            if space_2d is None:  # Skip zay in DIVA mode
                continue
                
            # Top row: color by digit label
            scatter1 = axes[0, col].scatter(space_2d[:, 0], space_2d[:, 1], 
                                          c=y_labels, cmap='tab10', alpha=0.7)
            axes[0, col].set_title(f'{title}\nColored by Digit')
            axes[0, col].legend(*scatter1.legend_elements(), title="Digits")
            
            # Bottom row: color by rotation
            scatter2 = axes[1, col].scatter(space_2d[:, 0], space_2d[:, 1], 
                                          c=r_labels, cmap='tab10', 
                                          vmin=0, vmax=4,  # Set the range to match our rotation indices
                                          alpha=0.7)
            axes[1, col].set_title(f'{title}\nColored by Rotation')
            # Create custom legend for rotations
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.tab10(i/4),  # Normalize to [0,1] range
                                        label=angle, markersize=10)
                             for i, angle in enumerate(rotation_angles)]
            axes[1, col].legend(handles=legend_elements, title="Rotations")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Latent space visualization saved to {save_path}")
        plt.close()

    def visualize_disentanglement(self, dataloader, device, save_path=None):
        """
        Visualize disentanglement by showing how changing each latent space affects generation
        while keeping others fixed.
        """
        self.eval()
        
        # Get a batch of data
        x, y, c, r = next(iter(dataloader))  # Changed to unpack 4 values
        x = x.to(device)
        y = y.to(device)
        r = r.to(device)  # Using r (rotation) as domain label
        
        # Get base latent representations
        z_loc, _ = self.qz(x)
        zy_base = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
        zx_base = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            zay_base = None
        else:
            zay_base = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
        za_base = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
        
        # Create figure for visualization
        if self.diva:
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # 3 rows for DIVA mode
        else:
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        # For each latent space, show how changing it affects generation
        for i in range(5):  # Show 5 variations
            # 1. Vary zy (label-specific)
            zy_varied = zy_base.clone()
            zy_varied[0] = zy_varied[0] * (1 + 0.5 * (i - 2))  # Scale the first sample
            img_zy = self.px(zy_varied, zx_base, zay_base, za_base)[0].cpu()
            axes[0, i].imshow(img_zy.permute(1, 2, 0).detach().numpy())
            axes[0, i].set_title(f'zy variation {i-2}')
            axes[0, i].axis('off')
            
            # 2. Vary za (domain-specific)
            za_varied = za_base.clone()
            za_varied[0] = za_varied[0] * (1 + 0.5 * (i - 2))
            img_za = self.px(zy_base, zx_base, zay_base, za_varied)[0].cpu()
            axes[1, i].imshow(img_za.permute(1, 2, 0).detach().numpy())
            axes[1, i].set_title(f'za variation {i-2}')
            axes[1, i].axis('off')
            
            # 3. Vary zay (domain-label interaction)
            if not self.diva:
                zay_varied = zay_base.clone()
                zay_varied[0] = zay_varied[0] * (1 + 0.5 * (i - 2))
                img_zay = self.px(zy_base, zx_base, zay_varied, za_base)[0].cpu()
                axes[2, i].imshow(img_zay.permute(1, 2, 0).detach().numpy())
                axes[2, i].set_title(f'zay variation {i-2}')
                axes[2, i].axis('off')
            
            # 4. Vary zx (residual)
            zx_varied = zx_base.clone()
            zx_varied[0] = zx_varied[0] * (1 + 0.5 * (i - 2))
            img_zx = self.px(zy_base, zx_varied, zay_base, za_base)[0].cpu()
            if self.diva:
                axes[2, i].imshow(img_zx.permute(1, 2, 0).detach().numpy())
                axes[2, i].set_title(f'zx variation {i-2}')
                axes[2, i].axis('off')
            else:
                axes[3, i].imshow(img_zx.permute(1, 2, 0).detach().numpy())
                axes[3, i].set_title(f'zx variation {i-2}')
                axes[3, i].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Label-specific (zy)', size='large')
        axes[1, 0].set_ylabel('Domain-specific (za)', size='large')
        if not self.diva:
            axes[2, 0].set_ylabel('Domain-Label (zay)', size='large')
            axes[3, 0].set_ylabel('Residual (zx)', size='large')
        else:
            axes[2, 0].set_ylabel('Residual (zx)', size='large')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Disentanglement visualization saved to {save_path}")
        plt.close()

    def visualize_latent_correlations(self, dataloader, device, save_path=None):
        """
        Visualize correlations between different latent spaces to show disentanglement
        """
        self.eval()
        zy_list, za_list, zay_list, zx_list = [], [], [], []
        
        with torch.no_grad():
            for x, y, c, r in dataloader:  # Changed to unpack 4 values
                x = x.to(device)
                z_loc, _ = self.qz(x)
                zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
                zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
                if self.diva:
                    zay = None
                else:
                    zay = z_loc[:, self.zay_index_range[0]:self.zay_index_range[1]]
                za = z_loc[:, self.za_index_range[0]:self.za_index_range[1]]
                
                zy_list.append(zy.cpu().numpy())
                za_list.append(za.cpu().numpy())
                if not self.diva:
                    zay_list.append(zay.cpu().numpy())
                zx_list.append(zx.cpu().numpy())
        
        # Convert to numpy arrays
        zy = np.concatenate(zy_list, axis=0)
        za = np.concatenate(za_list, axis=0)
        if not self.diva:
            zay = np.concatenate(zay_list, axis=0)
        zx = np.concatenate(zx_list, axis=0)
        
        # Calculate correlations
        corr_zy_za = np.corrcoef(zy.T, za.T)
        corr_zy_zx = np.corrcoef(zy.T, zx.T)
        corr_za_zx = np.corrcoef(za.T, zx.T)
        if not self.diva:
            corr_zy_zay = np.corrcoef(zy.T, zay.T)
            corr_za_zay = np.corrcoef(za.T, zay.T)
            corr_zx_zay = np.corrcoef(zx.T, zay.T)
        
        # Create correlation heatmaps
        if self.diva:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot correlations
        if self.diva:
            im1 = axes[0].imshow(corr_zy_za, cmap='RdBu', vmin=-1, vmax=1)
            axes[0].set_title('zy vs za correlation')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(corr_zy_zx, cmap='RdBu', vmin=-1, vmax=1)
            axes[1].set_title('zy vs zx correlation')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(corr_za_zx, cmap='RdBu', vmin=-1, vmax=1)
            axes[2].set_title('za vs zx correlation')
            plt.colorbar(im3, ax=axes[2])
        else:
            im1 = axes[0, 0].imshow(corr_zy_za, cmap='RdBu', vmin=-1, vmax=1)
            axes[0, 0].set_title('zy vs za correlation')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(corr_zy_zx, cmap='RdBu', vmin=-1, vmax=1)
            axes[0, 1].set_title('zy vs zx correlation')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[1, 0].imshow(corr_za_zx, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, 0].set_title('za vs zx correlation')
            plt.colorbar(im3, ax=axes[1, 0])
            
            im4 = axes[1, 1].imshow(corr_zy_zay, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, 1].set_title('zy vs zay correlation')
            plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Latent correlation visualization saved to {save_path}")
        plt.close()



#Encoder module 
class qz(NModule):
    def __init__(self,
                 zy_dim,
                 zx_dim,
                 zay_dim,
                 za_dim,
                 z_total_dim, 
                 in_channels=3,
                 out_channels=32,
                 kernel=3,
                 stride=1,
                 padding=1,
                 diva=False):
        super().__init__()

        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.za_dim = za_dim 
        if diva:
            self.zay_dim = 0
            self.z_total_dim = zy_dim + zx_dim + za_dim
        else:
            self.zay_dim = zay_dim
            self.z_total_dim = zy_dim + zx_dim + zay_dim + za_dim

        # triple conv channels
        self.encoder = nn.Sequential(                                          #out dims... in dims is (N, 3, 28, 28)
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),     #(N, 32, 28, 28) 
            nn.ReLU(),
            nn.MaxPool2d(2),                                                   #(N, 32, 14, 14)
            nn.Conv2d(out_channels, out_channels*2, kernel_size = 3, stride = 1, padding = 1),  #(N, 64, 14, 14) 
            nn.ReLU(),
            nn.MaxPool2d(2),                                                   #(N, 64, 7, 7)
            nn.Flatten()
            )
        self.loc = nn.Linear(64*7*7, z_total_dim)                          
        self.scale = nn.Linear(64*7*7, z_total_dim)                       



    def forward(self, x):
        h = self.encoder(x)
        h = F.relu(h)
        z_loc = self.loc(h)
        z_scale = F.softplus(self.scale(h)) + 1e-7
        
        # h = self.encoder(x)
        # z_loc = self.loc(h)
        # z_scale = F.softplus(self.scale(h)) + 1e-7
        

        return z_loc, z_scale
        
#Auxiliarry classifiers
class qy(NModule):
    def __init__(self, y_dim, zy_dim, zay_dim, diva=False):
        super().__init__()
        self.diva = diva

        if self.diva:
            self.zay_dim = None
            self.z_combined_dim = zy_dim  # When diva=True, we only use zy_dim
        else:
            self.zay_dim = zay_dim
            self.z_combined_dim = zy_dim + zay_dim

        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, y_dim)
        self.fc3 = nn.Linear(y_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()

    def forward(self, zy, zay):
        if self.diva:
            # When diva=True, we only use zy
            z_combined = zy
        else:
            # When diva=False, concatenate zy and zay
            z_combined = torch.cat((zy, zay), -1)

        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        y_hat = torch.softmax(logits, dim=1)
        return y_hat


class qa(NModule):
    def __init__(self, a_dim, za_dim, zay_dim, diva=False):
        super().__init__()
        self.diva = diva
        if self.diva:
            self.zay_dim = 0
            self.z_combined_dim = za_dim  # When diva=True, we only use za_dim
        else:
            self.zay_dim = zay_dim
            self.z_combined_dim = za_dim + zay_dim

        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, a_dim)
        self.fc3 = nn.Linear(a_dim, a_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()

    def forward(self, za, zay):
        if self.diva:
            # When diva=True, we only use za
            z_combined = za
        else:
            # When diva=False, concatenate za and zay
            z_combined = torch.cat((za, zay), -1)

        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        a_hat = torch.softmax(logits, dim=1)
        return a_hat



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
#             nn.Conv2d(out_channels, out_channels*2, kernel, stride, padding),
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
            padding=0,
            diva=False):
        super().__init__()

        self.diva = diva
        print(f"diva: {diva}")
        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.za_dim = za_dim
        print(f"zy_dim: {zy_dim}")
        if self.diva:
            self.zay_dim = None
            self.z_total_dim = self.zy_dim + self.zx_dim + self.za_dim
        else:
            self.zay_dim = zay_dim
            self.z_total_dim = self.zy_dim + self.zx_dim + self.zay_dim + self.za_dim

        self.out_channels = out_channels

        self.fc1 = nn.Linear(z_total_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Add sigmoid to ensure output values are in [0,1] range
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
        h = h.view(-1, 64, 7, 7)
        h = self.decoder(h)
        return h

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
        self.in_dim = y_dim + a_dim
        

        self.zay_dim = zay_dim

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
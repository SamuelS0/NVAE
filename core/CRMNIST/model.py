import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torchsummary import summary
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

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
            zdy_dim,
            zd_dim,
            y_dim,
            a_dim,
            ):
        cls.class_map = class_map
        cls.zy_dim = zy_dim
        cls.zx_dim = zx_dim
        cls.zdy_dim = zdy_dim
        cls.zd_dim = zd_dim
        cls.y_dim = y_dim
        cls.a_dim = a_dim

# make latent dims divisible by 3
class VAE(NModule):
    def __init__(
            self,
            class_map,
            zy_dim = 8,
            zx_dim= 8,
            zdy_dim = 8,
            zd_dim=8,
            y_dim=10,
            a_dim=6,
            in_channels=3,
            out_channels=32,
            kernel=3,
            stride=1,
            padding=0,
            beta_zy=1,
            beta_zx=1,
            beta_zdy=1,
            beta_zd=1,
            alpha_y=1,
            alpha_d=2,
            diva=False,
            l1_lambda_zy=0.0,
            l1_lambda_zx=0.0,
            l1_lambda_zdy=0.0,
            l1_lambda_zd=0.0,
            separate_encoders=False
            ):
        
        super().__init__()

        self.name = 'nvae' if not diva else 'diva'

        self.class_map = class_map

        if diva:
            # Redistribute zdy dimensions to zy, zx, zd to maintain total of 32
            # With zdy_dim=8: extra_dim=2, remainder=2, so zy=11, zx=10, zd=11 (total=32)
            extra_dim = zdy_dim // 3  # 8 // 3 = 2
            remainder = zdy_dim % 3   # 8 % 3 = 2

            self.zy_dim = zy_dim + extra_dim + (1 if remainder > 0 else 0)  # 8 + 2 + 1 = 11
            self.zx_dim = zx_dim + extra_dim                                 # 8 + 2 = 10
            self.zd_dim = zd_dim + extra_dim + (1 if remainder > 1 else 0)  # 8 + 2 + 1 = 11
            self.zdy_dim = None
            self.z_y_combined_dim = self.zy_dim
            self.z_a_combined_dim = self.zd_dim
            self.z_total_dim = self.zx_dim + self.zy_dim + self.zd_dim

        else:
            self.zy_dim = zy_dim
            self.zx_dim = zx_dim
            self.zdy_dim = zdy_dim
            self.zd_dim = zd_dim
            self.z_y_combined_dim = zy_dim + zdy_dim
            self.z_a_combined_dim = zd_dim + zdy_dim
            self.z_total_dim = zx_dim + zy_dim + zdy_dim + zd_dim

        self.y_dim = y_dim
        self.a_dim = a_dim


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.beta_zy, self.beta_zx, self.beta_zdy, self.beta_zd = beta_zy, beta_zx, beta_zdy, beta_zd
        self.alpha_y, self.alpha_d = alpha_y, alpha_d
        self.diva = diva

        # L1 sparsity penalty weights
        self.l1_lambda_zy = l1_lambda_zy
        self.l1_lambda_zx = l1_lambda_zx
        self.l1_lambda_zdy = l1_lambda_zdy
        self.l1_lambda_zd = l1_lambda_zd

        self.zy_index_range = [0, self.zy_dim]
        self.zx_index_range = [self.zy_dim, self.zy_dim + self.zx_dim]
        if diva:
            self.zdy_index_range = None
            self.zd_index_range = [self.zy_dim + self.zx_dim, self.z_total_dim]
        else:
            self.zdy_index_range = [self.zy_dim + self.zx_dim, self.zy_dim + self.zx_dim + self.zdy_dim]
            self.zd_index_range = [self.zy_dim + self.zx_dim + self.zdy_dim, self.z_total_dim]

        # Store separate_encoders flag for checkpoint compatibility
        self.separate_encoders = separate_encoders

        # Choose encoder: separate (for true independence) or shared (original)
        if separate_encoders:
            # Use separate encoders for each latent space - achieves I(Z_i; Z_j) = 0
            self.qz = SeparateEncoders(self.zy_dim, self.zx_dim, self.zdy_dim, self.zd_dim,
                                       in_channels, diva)
        else:
            # Original shared encoder (backward compatible)
            self.qz = qz(self.zy_dim, self.zx_dim, self.zdy_dim, self.zd_dim, self.z_total_dim,
                         in_channels, out_channels, kernel, stride, padding, diva)
        self.qy = qy(self.y_dim, self.zy_dim, self.zdy_dim, diva)
        self.qd = qd(self.a_dim, self.zd_dim, self.zdy_dim, diva)

        self.pzy = pzy(self.y_dim, self.zy_dim)
        
        if diva:
            self.pzdy = None
        else:
            self.pzdy = pzdy(self.y_dim, self.a_dim, self.zdy_dim)

        self.pzd = pzd(self.a_dim, self.zd_dim)
        self.px = px(self.zy_dim, self.zx_dim, self.zdy_dim, self.zd_dim, self.z_total_dim, in_channels, out_channels, kernel, diva=diva)


    def forward(self, y, x, a):
        # Encode
        qz_loc, qz_scale = self.qz(x)

        # Encoder Reparameterization
        qz = dist.Normal(qz_loc, qz_scale)
        z = qz.rsample()
        
        zy = z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        zx = z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            zdy = None
        else:
            zdy = z[:, self.zdy_index_range[0]:self.zdy_index_range[1]]

        zd = z[:, self.zd_index_range[0]:self.zd_index_range[1]]
        
        if self.diva:
            assert z.shape[1] == self.zy_dim + self.zx_dim + self.zd_dim
        else:
            assert z.shape[1] == self.zy_dim + self.zx_dim + self.zdy_dim + self.zd_dim

        # Decoder Reconstruction
        x_recon = self.px(zy, zx, zdy, zd)

        # Priors
        pzy_loc, pzy_scale = self.pzy(y)
        device = next(self.parameters()).device  # Get the device of the model
        pzx_loc = torch.zeros(zx.size(0), self.zx_dim, device=device)
        pzx_scale = torch.ones(zx.size(0), self.zx_dim, device=device)
        pzd_loc, pzd_scale = self.pzd(a)
        
        if self.diva:
            pzdy_loc, pzdy_scale = None, None
        else:
            pzdy_loc, pzdy_scale = self.pzdy(y, a)

        # Priors Reparameterization
        pzy = dist.Normal(pzy_loc, pzy_scale)
        pzx = dist.Normal(pzx_loc, pzx_scale)
        pzd = dist.Normal(pzd_loc, pzd_scale)
        if self.diva:
            pzdy = None
        else:
            pzdy = dist.Normal(pzdy_loc, pzdy_scale)

        # Auxiliary
        y_hat = self.qy(zy, zdy)
        a_hat = self.qd(zd, zdy)

        return x_recon, z, qz, pzy, pzx, pzd, pzdy, y_hat, a_hat, zy, zx, zdy, zd #zs were sampled from q



    def loss_function(self, y, x, a):
        x_recon, z, qz, pzy, pzx, pzd, pzdy, y_hat, a_hat, zy, zx, zdy, zd = self.forward(y, x, a)

        log_prob_z = qz.log_prob(z)
        log_prob_zy = log_prob_z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        log_prob_zx = log_prob_z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        if self.diva:
            log_prob_zdy = torch.zeros_like(log_prob_zy)  # Create zero tensor for DIVA mode
        else:
            log_prob_zdy = log_prob_z[:, self.zdy_index_range[0]:self.zdy_index_range[1]]
        log_prob_zd = log_prob_z[:, self.zd_index_range[0]:self.zd_index_range[1]]

        x_recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_zy = torch.sum(log_prob_zy - pzy.log_prob(zy))
        kl_zx = torch.sum(log_prob_zx - pzx.log_prob(zx))
        if self.diva:
            kl_zdy = torch.zeros_like(kl_zy)
        else:
            kl_zdy = torch.sum(log_prob_zdy - pzdy.log_prob(zdy))
        kl_zd = torch.sum(log_prob_zd - pzd.log_prob(zd))

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

        # L1 sparsity penalties (compute individually for detailed tracking)
        l1_zy = self.l1_lambda_zy * torch.mean(torch.abs(zy)) if self.l1_lambda_zy > 0 else 0.0
        l1_zx = self.l1_lambda_zx * torch.mean(torch.abs(zx)) if self.l1_lambda_zx > 0 else 0.0
        l1_zdy = (self.l1_lambda_zdy * torch.mean(torch.abs(zdy)) if self.l1_lambda_zdy > 0 else 0.0) if not self.diva else 0.0
        l1_zd = self.l1_lambda_zd * torch.mean(torch.abs(zd)) if self.l1_lambda_zd > 0 else 0.0

        # Total L1 penalty
        l1_penalty = l1_zy + l1_zx + l1_zdy + l1_zd

        # Calculate positive loss (removing negative sign)
        # In VAEs, we want to minimize the reconstruction loss + KL divergence
        total_loss = x_recon_loss + self.beta_zy * kl_zy + self.beta_zx * kl_zx \
                    + self.beta_zdy * kl_zdy + self.beta_zd * kl_zd \
                    + self.alpha_y * y_cross_entropy + self.alpha_d * a_cross_entropy \
                    + l1_penalty

        # Return both total loss and components for detailed tracking
        loss_components = {
            'total': total_loss.item(),
            'recon': x_recon_loss.item(),
            'kl_zy': (self.beta_zy * kl_zy).item(),
            'kl_zx': (self.beta_zx * kl_zx).item(),
            'kl_zdy': (self.beta_zdy * kl_zdy).item(),
            'kl_zd': (self.beta_zd * kl_zd).item(),
            'y_ce': (self.alpha_y * y_cross_entropy).item(),
            'a_ce': (self.alpha_d * a_cross_entropy).item(),
            'l1': l1_penalty if isinstance(l1_penalty, float) else l1_penalty.item(),
            'l1_zy': l1_zy if isinstance(l1_zy, float) else l1_zy.item(),
            'l1_zx': l1_zx if isinstance(l1_zx, float) else l1_zx.item(),
            'l1_zdy': l1_zdy if isinstance(l1_zdy, float) else l1_zdy.item(),
            'l1_zd': l1_zd if isinstance(l1_zd, float) else l1_zd.item()
        }

        return total_loss, loss_components
    
    def classifier(self, x):
        with torch.no_grad():
            z_loc, z_scale = self.qz(x)
            zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
            zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
            if self.diva:
                zdy = None  # More efficient than creating zero tensors
            else:
                zdy = z_loc[:, self.zdy_index_range[0]:self.zdy_index_range[1]]
            zd = z_loc[:, self.zd_index_range[0]:self.zd_index_range[1]]
            # Use get_probabilities for inference (returns softmax probabilities)
            y_hat = self.qy.get_probabilities(zy, zdy)
            return y_hat

    def generate(self, y, a=None, num_samples=10, device=None):
        """
        Generate samples conditionally based on class label y and domain label a.
        
        Args:
            y: Class label (digit 0-9) to generate. Can be an integer, a tensor of integers,
               or None (which will generate one sample per class)
            a: Domain/attribute label. If None, will randomly sample one per generated image
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
            
            # Handle attribute labels
            if a is None:
                # Randomly sample attributes if not provided
                a = torch.randint(0, self.a_dim, (batch_size,)).to(device)
            elif isinstance(a, int):
                a = torch.tensor([a]).repeat(num_samples).to(device)
            elif isinstance(a, torch.Tensor) and len(a.shape) == 0:
                a = a.repeat(num_samples).to(device)
            else:
                a = a.to(device)
            
            # Sample from prior distributions
            # For zy, use the learned prior conditioned on a
            a_one_hot = F.one_hot(a, self.a_dim).float()
            pzd_loc, pzd_scale = self.pzd(a)
            pzd = dist.Normal(pzd_loc, pzd_scale)
            zd = pzd.sample()

            if self.diva:
                zdy = None
            else:
                pzdy_loc, pzdy_scale = self.pzdy(y, a)
                pzdy = dist.Normal(pzdy_loc, pzdy_scale)
                zdy = pzdy.sample()
            
            # Generate images using the decoder with conditional y, a
            generated_images = self.px(zy, zx, zdy, zd)
            
            return generated_images, y

    # def visualize_disentanglement(self, dataloader, device, save_path=None):
    #     """
    #     Visualize disentanglement by showing how changing each latent space affects generation
    #     while keeping others fixed.
    #     """
    #     self.eval()
        
    #     # Get a batch of data
    #     x, y, c, r = next(iter(dataloader))  # Changed to unpack 4 values
    #     x = x.to(device)
    #     y = y.to(device)
    #     r = r.to(device)  # Using r (rotation) as domain label
        
    #     # Get base latent representations
    #     z_loc, _ = self.qz(x)
    #     zy_base = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
    #     zx_base = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
    #     if self.diva:
    #         zay_base = None
    #     else:
    #         zay_base = z_loc[:, self.zdy_index_range[0]:self.zdy_index_range[1]]
    #     za_base = z_loc[:, self.zd_index_range[0]:self.zd_index_range[1]]
        
    #     # Create figure for visualization
    #     if self.diva:
    #         fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # 3 rows for DIVA mode
    #     else:
    #         fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
    #     # For each latent space, show how changing it affects generation
    #     for i in range(5):  # Show 5 variations
    #         # 1. Vary zy (label-specific)
    #         zy_varied = zy_base.clone()
    #         zy_varied[0] = zy_varied[0] * (1 + 0.5 * (i - 2))  # Scale the first sample
    #         img_zy = self.px(zy_varied, zx_base, zay_base, za_base)[0].cpu()
    #         axes[0, i].imshow(img_zy.permute(1, 2, 0).detach().numpy())
    #         axes[0, i].set_title(f'zy variation {i-2}')
    #         axes[0, i].axis('off')
            
    #         # 2. Vary zd (domain-specific)
    #         zd_varied = zd_base.clone()
    #         zd_varied[0] = zd_varied[0] * (1 + 0.5 * (i - 2))
    #         img_zd = self.px(zy_base, zx_base, zdy_base, zd_varied)[0].cpu()
    #         axes[1, i].imshow(img_zd.permute(1, 2, 0).detach().numpy())
    #         axes[1, i].set_title(f'zd variation {i-2}')
    #         axes[1, i].axis('off')
            
    #         # 3. Vary zdy (domain-label interaction)
    #         if not self.diva:
    #             zdy_varied = zdy_base.clone()
    #             zdy_varied[0] = zdy_varied[0] * (1 + 0.5 * (i - 2))
    #             img_zdy = self.px(zy_base, zx_base, zdy_varied, zd_base)[0].cpu()
    #             axes[2, i].imshow(img_zdy.permute(1, 2, 0).detach().numpy())
    #             axes[2, i].set_title(f'zdy variation {i-2}')
    #             axes[2, i].axis('off')
            
    #         # 4. Vary zx (residual)
    #         zx_varied = zx_base.clone()
    #         zx_varied[0] = zx_varied[0] * (1 + 0.5 * (i - 2))
    #         img_zx = self.px(zy_base, zx_varied, zdy_base, zd_base)[0].cpu()
    #         if self.diva:
    #             axes[2, i].imshow(img_zx.permute(1, 2, 0).detach().numpy())
    #             axes[2, i].set_title(f'zx variation {i-2}')
    #             axes[2, i].axis('off')
    #         else:
    #             axes[3, i].imshow(img_zx.permute(1, 2, 0).detach().numpy())
    #             axes[3, i].set_title(f'zx variation {i-2}')
    #             axes[3, i].axis('off')
        
    #     # Add row labels
    #     axes[0, 0].set_ylabel('Label-specific (zy)', size='large')
    #     axes[1, 0].set_ylabel('Domain-specific (zd)', size='large')
    #     if not self.diva:
    #         axes[2, 0].set_ylabel('Domain-Label (zdy)', size='large')
    #         axes[3, 0].set_ylabel('Residual (zx)', size='large')
    #     else:
    #         axes[2, 0].set_ylabel('Residual (zx)', size='large')
        
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path)
    #         print(f"Disentanglement visualization saved to {save_path}")
    #     plt.close()

    # def visualize_latent_correlations(self, dataloader, device, save_path=None):
    #     """
    #     Visualize correlations between different latent spaces to show disentanglement
    #     """
    #     self.eval()
    #     zy_list, za_list, zay_list, zx_list = [], [], [], []
        
    #     with torch.no_grad():
    #         for x, y, c, r in dataloader:  # Changed to unpack 4 values
    #             x = x.to(device)
    #             z_loc, _ = self.qz(x)
    #             zy = z_loc[:, self.zy_index_range[0]:self.zy_index_range[1]]
    #             zx = z_loc[:, self.zx_index_range[0]:self.zx_index_range[1]]
    #             if self.diva:
    #                 zdy = None
    #             else:
    #                 zdy = z_loc[:, self.zdy_index_range[0]:self.zdy_index_range[1]]
    #             zd = z_loc[:, self.zd_index_range[0]:self.zd_index_range[1]]
                
    #             zy_list.append(zy.cpu().numpy())
    #             zd_list.append(zd.cpu().numpy())
    #             if not self.diva:
    #                 zdy_list.append(zdy.cpu().numpy())
    #             zx_list.append(zx.cpu().numpy())

    #     # Convert to numpy arrays
    #     zy = np.concatenate(zy_list, axis=0)
    #     zd = np.concatenate(zd_list, axis=0)
    #     if not self.diva:
    #         zdy = np.concatenate(zdy_list, axis=0)
    #     zx = np.concatenate(zx_list, axis=0)

    #     # Calculate correlations
    #     corr_zy_zd = np.corrcoef(zy.T, zd.T)
    #     corr_zy_zx = np.corrcoef(zy.T, zx.T)
    #     corr_zd_zx = np.corrcoef(zd.T, zx.T)
    #     if not self.diva:
    #         corr_zy_zdy = np.corrcoef(zy.T, zdy.T)
    #         corr_zd_zdy = np.corrcoef(zd.T, zdy.T)
    #         corr_zx_zdy = np.corrcoef(zx.T, zdy.T)

    #     # Create correlation heatmaps
    #     if self.diva:
    #         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #     else:
    #         fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    #     # Plot correlations
    #     if self.diva:
    #         im1 = axes[0].imshow(corr_zy_zd, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[0].set_title('zy vs zd correlation')
    #         plt.colorbar(im1, ax=axes[0])

    #         im2 = axes[1].imshow(corr_zy_zx, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[1].set_title('zy vs zx correlation')
    #         plt.colorbar(im2, ax=axes[1])

    #         im3 = axes[2].imshow(corr_zd_zx, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[2].set_title('zd vs zx correlation')
    #         plt.colorbar(im3, ax=axes[2])
    #     else:
    #         im1 = axes[0, 0].imshow(corr_zy_zd, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[0, 0].set_title('zy vs zd correlation')
    #         plt.colorbar(im1, ax=axes[0, 0])
            
    #         im2 = axes[0, 1].imshow(corr_zy_zx, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[0, 1].set_title('zy vs zx correlation')
    #         plt.colorbar(im2, ax=axes[0, 1])
            
    #         im3 = axes[1, 0].imshow(corr_zd_zx, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[1, 0].set_title('zd vs zx correlation')
    #         plt.colorbar(im3, ax=axes[1, 0])

    #         im4 = axes[1, 1].imshow(corr_zy_zdy, cmap='RdBu', vmin=-1, vmax=1)
    #         axes[1, 1].set_title('zy vs zdy correlation')
    #         plt.colorbar(im4, ax=axes[1, 1])
        
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path)
    #         print(f"Latent correlation visualization saved to {save_path}")
    #     plt.close()



#Encoder module for a single latent space
class SingleLatentEncoder(nn.Module):
    """
    Single encoder for one latent space (used for separate encoders mode).
    Same architecture as qz but outputs only one latent dimension.
    """
    def __init__(self, z_dim, in_channels=3):
        super().__init__()
        self.z_dim = z_dim

        # Same architecture as qz encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=5, stride=1, padding=2),
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
            nn.Flatten()
        )

        # Output layers for this specific latent space
        self.loc = nn.Linear(192 * 7 * 7, z_dim)
        self.scale = nn.Sequential(
            nn.Linear(192 * 7 * 7, z_dim),
            nn.Softplus()
        )

    def forward(self, x):
        h = self.encoder(x)
        z_loc = self.loc(h)
        z_scale = self.scale(h) + 1e-7
        return z_loc, z_scale


class SeparateEncoders(nn.Module):
    """
    Manages 4 independent CNN encoders for each latent space (zy, zx, zdy, zd).
    Each encoder processes the input independently to achieve true statistical independence I(Z_i; Z_j) = 0.
    """
    def __init__(self, zy_dim, zx_dim, zdy_dim, zd_dim, in_channels=3, diva=False):
        super().__init__()
        self.diva = diva
        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.zdy_dim = zdy_dim if not diva else 0
        self.zd_dim = zd_dim

        # Create independent encoders for each latent space
        self.encoder_zy = SingleLatentEncoder(zy_dim, in_channels)
        self.encoder_zx = SingleLatentEncoder(zx_dim, in_channels)
        if not diva:
            self.encoder_zdy = SingleLatentEncoder(zdy_dim, in_channels)
        else:
            self.encoder_zdy = None
        self.encoder_zd = SingleLatentEncoder(zd_dim, in_channels)

    def forward(self, x):
        # Each encoder processes x independently - true independence!
        mu_zy, logvar_zy = self.encoder_zy(x)
        mu_zx, logvar_zx = self.encoder_zx(x)
        mu_zd, logvar_zd = self.encoder_zd(x)

        if not self.diva:
            mu_zdy, logvar_zdy = self.encoder_zdy(x)
            # Concatenate in the expected order: zy, zx, zdy, zd
            mu = torch.cat([mu_zy, mu_zx, mu_zdy, mu_zd], dim=1)
            logvar = torch.cat([logvar_zy, logvar_zx, logvar_zdy, logvar_zd], dim=1)
        else:
            # DIVA mode: no zdy
            mu = torch.cat([mu_zy, mu_zx, mu_zd], dim=1)
            logvar = torch.cat([logvar_zy, logvar_zx, logvar_zd], dim=1)

        return mu, logvar


#Encoder module (shared encoder - original)
class qz(NModule):
    def __init__(self,
                 zy_dim,
                 zx_dim,
                 zdy_dim,
                 zd_dim,
                 z_total_dim, 
                 in_channels=3,
                 out_channels=32,
                 kernel=5,
                 stride=1,
                 padding=2,
                 diva=False):
        super().__init__()

        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.zd_dim = zd_dim 
        if diva:
            self.zdy_dim = 0
            self.z_total_dim = zy_dim + zx_dim + zd_dim
        else:
            self.zdy_dim = zdy_dim
            self.z_total_dim = zy_dim + zx_dim + zdy_dim + zd_dim

        # Encoder architecture matching DIVA paper but with 3x channels
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),  # 32 -> 96
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # Block 2
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # 64 -> 192
            nn.BatchNorm2d(192),
            nn.ReLU(),
            # Block 4
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        
        # Block 5: Linear layers
        self.loc = nn.Linear(192 * 7 * 7, z_total_dim)  # 64 -> 192
        self.scale = nn.Sequential(
            nn.Linear(192 * 7 * 7, z_total_dim),  # 64 -> 192
            nn.Softplus()
        )

    def forward(self, x):
        h = self.encoder(x)
        z_loc = self.loc(h)
        z_scale = self.scale(h) + 1e-7  # Add small epsilon for numerical stability
        
        return z_loc, z_scale
        
#Auxiliarry classifiers
class qy(NModule):
    def __init__(self, y_dim, zy_dim, zdy_dim, diva=False):
        super().__init__()
        self.diva = diva

        if self.diva:
            self.zdy_dim = None
            self.z_combined_dim = zy_dim  # When diva=True, we only use zy_dim
        else:
            self.zdy_dim = zdy_dim
            self.z_combined_dim = zy_dim + zdy_dim

        # 3-layer MLP with 64 hidden units
        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc3.bias.zero_()

    def forward(self, zy, zdy):
        if self.diva:
            # When diva=True, we only use zy (zdy should be None)
            if zdy is not None:
                assert torch.all(zdy == 0), "zdy should be None or all zeros in DIVA mode"
            z_combined = zy
        else:
            # When diva=False, concatenate zy and zdy
            assert zdy is not None, "zdy cannot be None in non-DIVA mode"
            z_combined = torch.cat((zy, zdy), -1)

        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        # Return raw logits for numerical stability with cross_entropy loss
        return logits

    def get_probabilities(self, zy, zdy):
        """Get probability distributions (for interpretation/inference)"""
        logits = self.forward(zy, zdy)
        return torch.softmax(logits, dim=1)


class qd(NModule):
    def __init__(self, a_dim, zd_dim, zdy_dim, diva=False):
        super().__init__()
        self.diva = diva
        if self.diva:
            self.zdy_dim = 0
            self.z_combined_dim = zd_dim  # When diva=True, we only use zd_dim
        else:
            self.zdy_dim = zdy_dim
            self.z_combined_dim = zd_dim + zdy_dim

        # 3-layer MLP with 64 hidden units
        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, a_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc3.bias.zero_()

    def forward(self, zd, zdy):
        if self.diva:
            # When diva=True, we only use zd (zdy should be None)
            if zdy is not None:
                assert torch.all(zdy == 0), "zdy should be None or all zeros in DIVA mode"
            z_combined = zd
        else:
            # When diva=False, concatenate zd and zdy
            assert zdy is not None, "zdy cannot be None in non-DIVA mode"
            z_combined = torch.cat((zd, zdy), -1)

        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        # Return raw logits for numerical stability with cross_entropy loss
        return logits

    def get_probabilities(self, zd, zdy):
        """Get probability distributions (for interpretation/inference)"""
        logits = self.forward(zd, zdy)
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
#             zd_dim=32,
#             in_channels=1,
#             out_channels=32,
#             kernel=3,
#             stride=1,
#             padding=0):
#         super().__init__()
#         self.zx_za = zd_dim
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
#     def __init__(self, zdy_dim=16):
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
            zdy_dim,
            zd_dim,
            z_total_dim,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            diva=False):
        super().__init__()

        self.diva = diva
        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.zd_dim = zd_dim
        if self.diva:
            self.zdy_dim = None
            self.z_total_dim = self.zy_dim + self.zx_dim + self.zd_dim
        else:
            self.zdy_dim = zdy_dim
            self.z_total_dim = self.zy_dim + self.zx_dim + self.zdy_dim + self.zd_dim

        # Block 1: Initial linear layer
        self.fc = nn.Sequential(
            nn.Linear(z_total_dim, 64 * 7 * 7),  # Changed from 1024 to match reshape dimensions
            nn.BatchNorm1d(64 * 7 * 7),
            nn.ReLU()
        )

        # Block 2: First upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # Block 3: First transposed convolution
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Block 4: Second upsampling
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # Block 5: Second transposed convolution
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Block 6: Final convolution
        self.conv3 = nn.Conv2d(256, 3, kernel_size=1)
        
    def forward(self, zy, zx, zdy, zd):
        # Combine latent variables
        if self.diva:
            assert zdy is None, "zdy should be None in DIVA mode"
            z_combined = torch.cat((zy, zx, zd), -1)
        else:
            z_combined = torch.cat((zy, zx, zdy, zd), -1)
        
        # Initial linear layer
        h = self.fc(z_combined)
        h = h.view(-1, 64, 7, 7)  
        
        # Upsampling and convolutions
        h = self.upsample1(h)  # 7x7 -> 14x14
        h = self.conv1(h)
        h = self.upsample2(h)  # 14x14 -> 28x28
        h = self.conv2(h)
        h = self.conv3(h)
        
        return torch.sigmoid(h)  # Ensure output is in [0,1] range

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
class pzdy(NModule):
    def __init__(self, y_dim, a_dim, zdy_dim):
        super().__init__()
        self.y_dim = y_dim
        self.a_dim = a_dim
        self.in_dim = y_dim + a_dim
        

        self.zdy_dim = zdy_dim

        # Simplify architecture to avoid BatchNorm issues
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, zdy_dim),
            nn.ReLU()
        )

        self.loc = nn.Linear(zdy_dim, zdy_dim)
        self.scale = nn.Sequential(
            nn.Linear(zdy_dim, zdy_dim),
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


class pzd(NModule):
    def __init__(self, a_dim, zd_dim):
        super().__init__()
        self.a_dim = a_dim
        
        # Simplify the decoder to reduce chances of numerical instability
        self.decoder = nn.Sequential(
            nn.Linear(a_dim, zd_dim),
            nn.ReLU()
        )
        
        self.loc = nn.Linear(zd_dim, zd_dim)
        self.scale = nn.Sequential(
            nn.Linear(zd_dim, zd_dim),
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
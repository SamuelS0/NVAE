import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from core.utils import kl_divergence
from torchsummary import summary

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
            alpha_2=2
            ):
        
        super().__init__()

        self.class_map = class_map
        
        self.y_dim = y_dim
        self.a_dim = a_dim

        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.za_dim = za_dim
        self.zay_dim = zay_dim
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

        self.zy_index_range = [0, zy_dim]
        self.zx_index_range = [zy_dim, zy_dim + zx_dim]
        self.zay_index_range = [zy_dim + zx_dim, zy_dim + zx_dim + zay_dim]
        self.za_index_range = [zy_dim + zx_dim + zay_dim, self.z_total_dim]
        
        self.qz = qz(zy_dim, zx_dim, zay_dim, za_dim, self.z_total_dim,
                     in_channels, out_channels, kernel, stride, padding)
        self.qy = qy(y_dim, zy_dim, zay_dim)
        self.qa = qa(a_dim, za_dim, zay_dim)

        self.pzy = pzy(y_dim, zy_dim)
        self.pzay = pzay(y_dim, a_dim, zay_dim)
        self.pza = pza(a_dim, za_dim)
        self.px = px(zy_dim, zx_dim, zay_dim, za_dim, self.z_total_dim, in_channels, out_channels, kernel)

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
        pzx_loc, pzx_scale = torch.zeros(zx.size(0), self.zx_dim), torch.ones(zx.size(0), self.zx_dim)
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



    def loss_function(self, y, x, a):
        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = self.forward(y, x, a)

        log_prob_z = qz.log_prob(z)
        log_prob_zy = log_prob_z[:, self.zy_index_range[0]:self.zy_index_range[1]]
        log_prob_zx = log_prob_z[:, self.zx_index_range[0]:self.zx_index_range[1]]
        log_prob_zay = log_prob_z[:, self.zay_index_range[0]:self.zay_index_range[1]]
        log_prob_za = log_prob_z[:, self.za_index_range[0]:self.za_index_range[1]]

        x_recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_zy = torch.sum(log_prob_zy - pzy.log_prob(zy))
        kl_zx = torch.sum(log_prob_zx - pzx.log_prob(zx))
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
            
            pza_loc, pza_scale = self.pza(a)
            pza = dist.Normal(pza_loc, pza_scale)
            za = pza.sample()
            
            pzay_loc, pzay_scale = self.pzay(y, a)
            pzay = dist.Normal(pzay_loc, pzay_scale)
            zay = pzay.sample()
            
            # Generate images using the decoder with conditional y, a
            generated_images = self.px(zy, zx, zay, za)
            
            return generated_images, y



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
                 padding=1
                 ):
        super().__init__()

        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.zay_dim = zay_dim
        self.za_dim = za_dim
        self.z_total_dim = zy_dim + zx_dim + zay_dim + za_dim
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
    def __init__(self, y_dim, zy_dim, zay_dim):
        super().__init__()

        self.fc1 = nn.Linear(zy_dim + zay_dim, 64)
        self.fc2 = nn.Linear(64, y_dim)
        self.fc3 = nn.Linear(y_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()

    def forward(self, zy, zay):
        z_combined = torch.cat((zy, zay), -1)
        h = self.fc1(z_combined)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        logits = self.fc3(h)
        y_hat = torch.softmax(logits, dim=1)
        return y_hat


class qa(NModule):
    def __init__(self, a_dim, za_dim, zay_dim):
        super().__init__()

        self.z_combined_dim = za_dim + zay_dim

        self.fc1 = nn.Linear(self.z_combined_dim, 64)
        self.fc2 = nn.Linear(64, a_dim)
        self.fc3 = nn.Linear(a_dim, a_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        with torch.no_grad():
            self.fc1.bias.zero_()

    def forward(self, za, zay):
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
            padding=0):
        super().__init__()
        
        # Store parameters
        self.z_total_dim = zy_dim + zx_dim + zay_dim + za_dim
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
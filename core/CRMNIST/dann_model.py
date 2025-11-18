import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

try:
    from sklearn.manifold import TSNE
except ImportError:
    print("Warning: scikit-learn not available. t-SNE visualization will be disabled.")
    TSNE = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Visualization plotting will be disabled.")
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    # Create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from core.CRMNIST.model import NModule
except ImportError as e:
    print(f"Warning: NModule not available due to missing dependency: {e}")
    # Fallback to basic nn.Module
    class NModule(nn.Module):
        def __init__(self):
            super().__init__()
        
        @classmethod
        def get_model_size(cls, model):
            """Fallback model size calculation"""
            return sum(p.numel() for p in model.parameters())


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from DANN paper
    Forward: identity transformation
    Backward: reverse gradient by multiplying by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class AugmentedDANN(NModule):
    def __init__(
        self,
        class_map,
        zy_dim=12,
        zd_dim=12,
        zdy_dim=12,
        y_dim=10,  # number of classes
        d_dim=6,   # number of domains
        in_channels=3,
        out_channels=32,
        kernel=3,
        stride=1,
        padding=0,
        lambda_reversal=1.0,
        sparsity_weight=0.05,
        alpha_y=1.0,
        alpha_d=1.0,
        beta_adv=0.15,
        image_size=28  # Image dimensions: 28 for CRMNIST, 96 for WILD
    ):
        super().__init__()

        self.name = 'dann'
        self.class_map = class_map

        # Model dimensions
        self.zy_dim = zy_dim
        self.zd_dim = zd_dim
        self.zdy_dim = zdy_dim
        self.y_dim = y_dim
        self.d_dim = d_dim
        self.image_size = image_size

        # Training parameters
        self.sparsity_weight_zdy_target = sparsity_weight  # Target weight for zdy sparsity (default 5.0)
        self.sparsity_weight_zdy_current = 0.0  # Current sparsity weight for zdy (increases from 0 to target)
        self.sparsity_weight_other_target = 1.0  # Target weight for zy and zd sparsity
        self.sparsity_weight_other_current = 0.0  # Current sparsity weight for zy and zd
        self.alpha_y = alpha_y
        self.alpha_d = alpha_d
        self.beta_adv = beta_adv

        # Single shared encoder that outputs to combined latent space
        # This matches the architecture pattern of NVAE/DIVA/DANN/IRM
        total_latent_dim = zy_dim + zd_dim + zdy_dim
        self.shared_encoder = self._create_encoder(in_channels, total_latent_dim, image_size)
        
        # System 1: Class-focused DANN components
        # Main class classifier (operates on Z_y ∪ Z_dy)
        self.class_classifier_main = nn.Sequential(
            nn.Linear(zy_dim + zdy_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, y_dim)
        )
        
        # System 2: Domain-focused DANN components
        # Main domain classifier (operates on Z_d ∪ Z_dy)
        self.domain_classifier_main = nn.Sequential(
            nn.Linear(zd_dim + zdy_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, d_dim)
        )
        
        # Separate GRL modules for fine-grained control
        self.grl_zy = GradientReversalLayer(lambda_reversal)
        self.grl_zd = GradientReversalLayer(lambda_reversal)
        
        # Adversarial classifiers without embedded GRL
        self.adversarial_domain_classifier = nn.Sequential(
            nn.Linear(zy_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, d_dim)
        )
        
        self.adversarial_class_classifier = nn.Sequential(
            nn.Linear(zd_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, y_dim)
        )
        
        self._init_weights()
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate model dimensions for consistency"""
        if self.zy_dim <= 0 or self.zd_dim <= 0 or self.zdy_dim <= 0:
            raise ValueError("All latent dimensions must be positive")

        if self.y_dim <= 0 or self.d_dim <= 0:
            raise ValueError("Output dimensions must be positive")

        # Validate shared encoder output dimension
        total_latent_dim = self.zy_dim + self.zd_dim + self.zdy_dim
        # Get the final layer of the shared encoder (should be the Linear layer)
        encoder_layers = list(self.shared_encoder.modules())
        final_linear = None
        for layer in reversed(encoder_layers):
            if isinstance(layer, nn.Linear):
                final_linear = layer
                break

        if final_linear is not None:
            if final_linear.out_features != total_latent_dim:
                raise ValueError(f"Shared encoder output dimension mismatch: "
                               f"expected {total_latent_dim}, got {final_linear.out_features}")

        # Validate classifier input dimensions match expected latent dimensions
        expected_class_input = self.zy_dim + self.zdy_dim
        expected_domain_input = self.zd_dim + self.zdy_dim

        # Check main classifiers
        class_first_layer = self.class_classifier_main[0]
        domain_first_layer = self.domain_classifier_main[0]

        if isinstance(class_first_layer, nn.Linear):
            if class_first_layer.in_features != expected_class_input:
                raise ValueError(f"Class classifier expects {expected_class_input} features, "
                               f"got {class_first_layer.in_features}")

        if isinstance(domain_first_layer, nn.Linear):
            if domain_first_layer.in_features != expected_domain_input:
                raise ValueError(f"Domain classifier expects {expected_domain_input} features, "
                               f"got {domain_first_layer.in_features}")

        # Check adversarial classifiers
        adv_domain_first = self.adversarial_domain_classifier[0]
        adv_class_first = self.adversarial_class_classifier[0]

        if isinstance(adv_domain_first, nn.Linear):
            if adv_domain_first.in_features != self.zy_dim:
                raise ValueError(f"Adversarial domain classifier expects {self.zy_dim} features, "
                               f"got {adv_domain_first.in_features}")

        if isinstance(adv_class_first, nn.Linear):
            if adv_class_first.in_features != self.zd_dim:
                raise ValueError(f"Adversarial class classifier expects {self.zd_dim} features, "
                               f"got {adv_class_first.in_features}")
    
    def _create_encoder(self, in_channels, out_dim, image_size):
        """
        Create a CNN encoder that outputs to combined or individual latent space.

        Architecture adapts to image size:
        - image_size=28 (CRMNIST): 28x28 -> 14x14 -> 7x7 -> flatten (9408) -> out_dim
        - image_size=96 (WILD): 96x96 -> 48x48 -> 24x24 -> 12x12 -> 6x6 -> flatten (18432) -> out_dim
        """
        if image_size == 28:
            # CRMNIST encoder: 28x28 -> 7x7
            return nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels, 96, kernel_size=5, stride=1, padding=2),
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
                # Block 5: Project to specific latent space
                nn.Linear(192 * 7 * 7, out_dim),
                nn.ReLU()
            )
        elif image_size == 96:
            # WILD encoder: 96x96 -> 6x6 (following qz_new pattern from model_wild.py)
            return nn.Sequential(
                # 96x96x3 -> 48x48x64
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),

                # 48x48x64 -> 24x24x128
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),

                # 24x24x128 -> 12x12x256
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),

                # 12x12x256 -> 6x6x512
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),

                nn.Flatten(),
                # 512 * 6 * 6 = 18432
                nn.Linear(512 * 6 * 6, out_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unsupported image_size={image_size}. Must be 28 (CRMNIST) or 96 (WILD)")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_features(self, x):
        """Extract features using shared encoder and split into latent spaces"""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if x.dim() != 4:
            raise ValueError(f"Input must be 4D tensor (batch, channels, height, width), got {x.dim()}D")

        if x.size(1) != 3:
            raise ValueError(f"Input must have 3 channels for RGB images, got {x.size(1)}")

        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f"Input must be {self.image_size}x{self.image_size} images, got {x.size(2)}x{x.size(3)}")

        # Single forward pass through shared encoder
        z_combined = self.shared_encoder(x)

        # Split the combined latent representation into separate spaces
        # Order: zy, zd, zdy
        zy = z_combined[:, :self.zy_dim]
        zd = z_combined[:, self.zy_dim:self.zy_dim + self.zd_dim]
        zdy = z_combined[:, self.zy_dim + self.zd_dim:]

        # Validate output dimensions
        if zy.size(1) != self.zy_dim:
            raise RuntimeError(f"Class latent output mismatch: expected {self.zy_dim}, got {zy.size(1)}")

        if zd.size(1) != self.zd_dim:
            raise RuntimeError(f"Domain latent output mismatch: expected {self.zd_dim}, got {zd.size(1)}")

        if zdy.size(1) != self.zdy_dim:
            raise RuntimeError(f"Interaction latent output mismatch: expected {self.zdy_dim}, got {zdy.size(1)}")

        return zy, zd, zdy
    
    def forward(self, y, x, d):
        """
        Standardized forward pass matching NVAE interface
        
        Args:
            y: class labels (for compatibility, not used in forward pass)
            x: input images
            d: domain labels (for compatibility, not used in forward pass)
            
        Returns:
            Tuple matching NVAE output format for compatibility
        """
        # Extract partitioned features
        zy, zd, zdy = self.extract_features(x)
        
        # Main predictions (use combined features)
        zy_zdy = torch.cat([zy, zdy], dim=1)
        y_pred_main = self.class_classifier_main(zy_zdy)
        
        zd_zdy = torch.cat([zd, zdy], dim=1)
        d_pred_main = self.domain_classifier_main(zd_zdy)
        
        # Adversarial predictions with explicit GRL application
        zy_reversed = self.grl_zy(zy)
        d_pred_adversarial = self.adversarial_domain_classifier(zy_reversed)
        
        zd_reversed = self.grl_zd(zd)
        y_pred_adversarial = self.adversarial_class_classifier(zd_reversed)
        
        # For compatibility with NVAE interface, return similar structure
        # We'll use dummy values for reconstruction since DANN doesn't generate
        device = x.device
        x_recon = torch.zeros_like(x)  # Dummy reconstruction
        z = torch.cat([zy, zd, zdy], dim=1)  # Combined latent representation
        
        # Dummy distributions for compatibility
        qz = None  # DANN doesn't use probabilistic latents
        pzy = None
        pzx = None
        pza = None
        pzay = None
        
        # Use main predictions as outputs
        y_hat = y_pred_main
        a_hat = d_pred_main
        
        # Return individual latent components for NVAE compatibility
        # AugmentedDANN has 3 latent spaces (zy, zd, zdy) but NVAE interface expects 4 (zy, zx, zay, za)
        # Mapping strategy:
        # zy -> zy (class-specific, unchanged)
        # zd -> za (domain-specific maps to auxiliary/attribute)
        # zdy -> zay (interaction maps to auxiliary-label interaction)
        # zd -> zx (domain-specific also maps to residual for compatibility)
        #
        # NOTE: zx and za both reference the SAME tensor (zd). This is intentional for interface
        # compatibility but means expressiveness evaluation will show identical results for both.
        # This does NOT affect model functionality, only interpretability of latent analysis.
        zx = zd   # Domain features as residual features (DUPLICATE of za)
        za = zd   # Domain features as auxiliary features (DUPLICATE of zx)
        zay = zdy # Interaction features as auxiliary-label interaction
        
        return x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za
    
    def dann_forward(self, x):
        """
        DANN-specific forward pass with clean gradient flow

        Returns:
            dict containing all predictions and features
        """
        # Extract features using shared encoder and split into latent spaces
        zy, zd, zdy = self.extract_features(x)
        
        # Main predictions (use combined features)
        zy_zdy = torch.cat([zy, zdy], dim=1)
        y_pred_main = self.class_classifier_main(zy_zdy)
        
        zd_zdy = torch.cat([zd, zdy], dim=1)
        d_pred_main = self.domain_classifier_main(zd_zdy)
        
        # Adversarial predictions with explicit GRL application
        zy_reversed = self.grl_zy(zy)
        d_pred_adversarial = self.adversarial_domain_classifier(zy_reversed)
        
        zd_reversed = self.grl_zd(zd)
        y_pred_adversarial = self.adversarial_class_classifier(zd_reversed)
        
        return {
            'y_pred_main': y_pred_main,           # Main class prediction
            'd_pred_main': d_pred_main,           # Main domain prediction
            'd_pred_adversarial': d_pred_adversarial,  # Domain prediction from Z_y (should fail)
            'y_pred_adversarial': y_pred_adversarial,  # Class prediction from Z_d (should fail)
            'zy': zy,
            'zd': zd,
            'zdy': zdy
        }
    
    def loss_function(self, y, x, d):
        """
        Compute the total loss for the augmented DANN matching NVAE interface
        
        Args:
            y: true class labels 
            x: input images
            d: true domain labels
            
        Returns:
            total_loss (scalar tensor)
        """
        outputs = self.dann_forward(x)
        
        # Convert one-hot to indices if necessary
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        if len(d.shape) > 1 and d.shape[1] > 1:
            d = torch.argmax(d, dim=1)
        
        # Main task losses (use combined features)
        loss_y_main = F.cross_entropy(outputs['y_pred_main'], y.long())
        loss_d_main = F.cross_entropy(outputs['d_pred_main'], d.long())
        
        # Adversarial losses (use pure features with GRL)
        loss_d_adversarial = F.cross_entropy(outputs['d_pred_adversarial'], d.long())
        loss_y_adversarial = F.cross_entropy(outputs['y_pred_adversarial'], y.long())

        # Sparsity penalties (L1 regularization) - all increase with training schedule
        sparsity_loss_zdy = torch.mean(torch.abs(outputs['zdy']))
        sparsity_loss_zy = torch.mean(torch.abs(outputs['zy']))
        sparsity_loss_zd = torch.mean(torch.abs(outputs['zd']))

        # Total loss
        total_loss = (
            self.alpha_y * loss_y_main +
            self.alpha_d * loss_d_main +
            self.beta_adv * (loss_d_adversarial + loss_y_adversarial) +
            self.sparsity_weight_zdy_current * sparsity_loss_zdy +
            self.sparsity_weight_other_current * (sparsity_loss_zy + sparsity_loss_zd)
        )

        return total_loss
    
    def detailed_loss(self, y, x, d):
        """
        Compute detailed loss breakdown for monitoring
        
        Returns:
            total_loss, loss_dict
        """
        outputs = self.dann_forward(x)
        
        # Convert one-hot to indices if necessary
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        if len(d.shape) > 1 and d.shape[1] > 1:
            d = torch.argmax(d, dim=1)
        
        # Main task losses (use combined features)
        loss_y_main = F.cross_entropy(outputs['y_pred_main'], y.long())
        loss_d_main = F.cross_entropy(outputs['d_pred_main'], d.long())
        
        # Adversarial losses (use pure features with GRL)
        loss_d_adversarial = F.cross_entropy(outputs['d_pred_adversarial'], d.long())
        loss_y_adversarial = F.cross_entropy(outputs['y_pred_adversarial'], y.long())

        # Sparsity penalties (L1 regularization) - all increase with training schedule
        sparsity_loss_zdy = torch.mean(torch.abs(outputs['zdy']))
        sparsity_loss_zy = torch.mean(torch.abs(outputs['zy']))
        sparsity_loss_zd = torch.mean(torch.abs(outputs['zd']))

        # Total loss
        total_loss = (
            self.alpha_y * loss_y_main +
            self.alpha_d * loss_d_main +
            self.beta_adv * (loss_d_adversarial + loss_y_adversarial) +
            self.sparsity_weight_zdy_current * sparsity_loss_zdy +
            self.sparsity_weight_other_current * (sparsity_loss_zy + sparsity_loss_zd)
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_y_main': loss_y_main.item(),
            'loss_d_main': loss_d_main.item(),
            'loss_d_adversarial': loss_d_adversarial.item(),
            'loss_y_adversarial': loss_y_adversarial.item(),
            'sparsity_loss_zdy': sparsity_loss_zdy.item(),
            'sparsity_loss_zy': sparsity_loss_zy.item(),
            'sparsity_loss_zd': sparsity_loss_zd.item(),
            'sparsity_weight_zdy': self.sparsity_weight_zdy_current,
            'sparsity_weight_other': self.sparsity_weight_other_current
        }

        return total_loss, loss_dict
    
    def classifier(self, x):
        """
        Predict class labels for compatibility with NVAE interface
        """
        with torch.no_grad():
            outputs = self.dann_forward(x)
            return torch.softmax(outputs['y_pred_main'], dim=1)
    
    def predict_class(self, x):
        """Predict class labels using the main class classifier"""
        return self.classifier(x)
    
    def predict_domain(self, x):
        """Predict domain labels using the main domain classifier"""
        with torch.no_grad():
            outputs = self.dann_forward(x)
            return torch.softmax(outputs['d_pred_main'], dim=1)
    
    def get_features(self, x):
        """Get feature representations for compatibility"""
        zy, zd, zdy = self.extract_features(x)
        return torch.cat([zy, zd, zdy], dim=1)
    
    def set_gradient_reversal_lambda(self, lambda_):
        """Update the gradient reversal parameter (useful for scheduling)"""
        self.grl_zy.set_lambda(lambda_)
        self.grl_zd.set_lambda(lambda_)
    
    def update_lambda_schedule(self, epoch, total_epochs):
        """
        Update gradient reversal lambda using standard DANN scheduling
        λ(p) = 2/(1+exp(-10p)) - 1 where p is training progress

        Also updates sparsity weights for all latent spaces using the same schedule:
        - zdy: increases from 0 to sparsity_weight_zdy_target (default 5.0)
        - zy, zd: increase from 0 to sparsity_weight_other_target (default 1.0)
        """
        # Ensure numerical stability
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")

        p = max(0.0, min(1.0, epoch / total_epochs))  # Clamp p to [0, 1]

        # Use numpy.clip to prevent overflow in exponential
        exp_arg = np.clip(-10 * p, -50, 50)  # Prevent extreme values
        try:
            lambda_val = 2 / (1 + np.exp(exp_arg)) - 1
            # Ensure lambda_val is in valid range
            lambda_val = np.clip(lambda_val, -1.0, 1.0)
        except (OverflowError, RuntimeWarning):
            # Fallback to safe value if computation fails
            lambda_val = -1.0 if p < 0.5 else 1.0

        self.set_gradient_reversal_lambda(lambda_val)

        # Update sparsity weights using same schedule: map [0, 1] to [0, target]
        # lambda_val ranges from 0 to ~1, so we use it directly as progress
        self.sparsity_weight_zdy_current = lambda_val * self.sparsity_weight_zdy_target
        self.sparsity_weight_other_current = lambda_val * self.sparsity_weight_other_target

        return lambda_val
    
    def visualize_latent_spaces(self, dataloader, device, save_path=None, max_samples=750, dataset_type="crmnist"):
        """
        Visualize all latent spaces using t-SNE
        Args:
            dataloader: DataLoader containing (x, y, c, r) tuples for CRMNIST or (x, y, metadata) for WILD
            device: torch device
            save_path: Optional path to save the visualization
            max_samples: Maximum number of samples to use for visualization
            dataset_type: Type of dataset ("crmnist" or "wild")
        """
        if TSNE is None:
            print("Error: scikit-learn not available. Cannot perform t-SNE visualization.")
            return
        
        if plt is None:
            print("Error: matplotlib not available. Cannot create visualization plots.")
            return
        
        self.eval()
        
        # Initialize lists to store latent vectors and labels
        zy_list, zd_list, zdy_list = [], [], []
        y_list, c_list, r_list = [], [], []
        
        # Initialize counters for each combination
        counts = {
            'digit': {i: 0 for i in range(10)},  # 10 digits
            'rotation': {i: 0 for i in range(6)},  # 6 rotations
            'color': {i: 0 for i in range(7)}  # 7 colors
        }
        
        # Target samples per category
        target_samples = min(200, max_samples // 30)
        total_collected = 0
        
        # Collect data in batches with improved memory management
        processed_batches = 0
        max_batches_per_pass = 50  # Limit memory usage
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if total_collected >= max_samples:
                    break

                # Limit number of batches to prevent memory issues
                if processed_batches >= max_batches_per_pass:
                    break

                try:
                    # Handle both CRMNIST and WILD data formats
                    if dataset_type == "wild":
                        x, y, metadata = batch
                        x = x.to(device)
                        y = y.to(device)
                        r = metadata[:, 0].to(device)  # Hospital ID from metadata
                        c = torch.zeros_like(y)  # Dummy color for WILD (not used)
                    else:
                        x, y, c, r = batch
                        x = x.to(device)
                        y = y.to(device)
                        c = c.to(device)
                        r = r.to(device)
                    
                    # Convert to indices if one-hot encoded
                    if len(y.shape) > 1:
                        y = torch.argmax(y, dim=1)
                    if len(c.shape) > 1:
                        c = torch.argmax(c, dim=1)
                    if len(r.shape) > 1:
                        r = torch.argmax(r, dim=1)
                    
                    # Create mask for samples we want to keep
                    keep_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
                    
                    # Check each sample with early termination
                    batch_collected = 0
                    for j in range(len(y)):
                        if total_collected >= max_samples or batch_collected >= 32:  # Limit per batch
                            break
                            
                        digit = y[j].item()
                        color = c[j].item()
                        rotation = r[j].item()
                        
                        # Only keep if we need more samples for this combination
                        if (counts['digit'][digit] < target_samples and 
                            counts['rotation'][rotation] < target_samples and 
                            counts['color'][color] < target_samples):
                            keep_mask[j] = True
                            counts['digit'][digit] += 1
                            counts['rotation'][rotation] += 1
                            counts['color'][color] += 1
                            total_collected += 1
                            batch_collected += 1
                    
                    # Apply mask to get only samples we want to keep
                    if keep_mask.any():
                        x_batch = x[keep_mask]
                        y_batch = y[keep_mask]
                        c_batch = c[keep_mask]
                        r_batch = r[keep_mask]
                    
                        # Get latent representations for this batch
                        zy, zd, zdy = self.extract_features(x_batch)
                        
                        # Move to CPU and store to save GPU memory
                        zy_list.append(zy.cpu())
                        zd_list.append(zd.cpu())
                        zdy_list.append(zdy.cpu())
                        y_list.append(y_batch.cpu())
                        c_list.append(c_batch.cpu())
                        r_list.append(r_batch.cpu())
                    
                    # Clear GPU cache periodically
                    if processed_batches % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    processed_batches += 1
                
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"Memory error in batch {i}: {e}")
                    # Try to free memory and continue
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    break
        
        # Check if we collected any data
        if not zy_list:
            raise ValueError("No data was collected for visualization. Check your dataloader and sampling criteria.")
        
        print(f"\nCollected {total_collected} samples total")
        
        # Concatenate tensors on CPU to save memory
        zy = torch.cat(zy_list, dim=0)
        zd = torch.cat(zd_list, dim=0)
        zdy = torch.cat(zdy_list, dim=0)
        y_labels = torch.cat(y_list, dim=0)
        c_labels = torch.cat(c_list, dim=0)
        r_labels = torch.cat(r_list, dim=0)
        
        # Clear the lists to free memory
        del zy_list, zd_list, zdy_list, y_list, c_list, r_list

        # Move to CPU for t-SNE and plotting
        zy = zy.cpu().numpy()
        zd = zd.cpu().numpy()
        zdy = zdy.cpu().numpy()
        y_labels = y_labels.cpu().numpy()
        c_labels = c_labels.cpu().numpy()
        r_labels = r_labels.cpu().numpy()
        
        # Apply t-SNE to each latent space with memory optimization
        def run_tsne(data, labels, max_iter=1000):
            try:
                # Use fewer iterations for faster computation if dataset is large
                n_iter = max_iter if data.shape[0] < 2000 else 500
                tsne = TSNE(n_components=2, random_state=42, n_iter=n_iter, 
                           learning_rate='auto', init='pca', method='barnes_hut')
                return tsne.fit_transform(data), labels
            except Exception as e:
                print(f"t-SNE failed: {e}. Using first 2 PCA components as fallback.")
                # Simple PCA fallback if t-SNE fails
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                return pca.fit_transform(data), labels
        
        # Prepare latent spaces for visualization
        latent_spaces = [(zy, y_labels), (zd, r_labels), (zdy, y_labels)]
        
        print("\nRunning t-SNE (this may take a few minutes)...")
        
        tsne_results = []
        for i, (space, labels) in enumerate(tqdm(latent_spaces, desc="Computing t-SNE", unit="space")):
            print(f"Processing latent space {i+1}/3 with {space.shape[0]} samples...")
            space_2d, sampled_labels = run_tsne(space, labels)
            tsne_results.append((space_2d, sampled_labels))
            
            # Free memory after each t-SNE computation
            del space
        
        # Unpack results
        zy_2d, y_labels = tsne_results[0]
        zd_2d, r_labels = tsne_results[1]
        zdy_2d, zdy_labels = tsne_results[2]
        
        # Define labels for legend
        rotation_angles = ['0°', '15°', '30°', '45°', '60°', '75°']
        color_labels = ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red']
        
        # Create figure with 3 rows and 3 columns for DANN
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Add overall title
        fig.suptitle('DANN Latent Space t-SNE',
                     fontsize=14, fontweight='bold', y=0.995)

        latent_spaces = [
            (zy_2d, y_labels, 'Class-Specific Space (zy)'),
            (zd_2d, r_labels, 'Domain-Specific Space (zd)'),
            (zdy_2d, zdy_labels, 'Domain-Class Interaction (zdy)')
        ]
        
        # Plot each latent space
        for col_idx, (space_2d, labels, title) in enumerate(latent_spaces):
            # Top row: color by digit label
            scatter1 = axes[0, col_idx].scatter(space_2d[:, 0], space_2d[:, 1],
                                          c=y_labels, cmap='tab10', vmin=0, vmax=9, alpha=0.4)
            axes[0, col_idx].set_title(f'{title}\nColored by Digit (Target Variable)',
                                      fontsize=10)
            axes[0, col_idx].set_xlabel('t-SNE Component 1', fontsize=9)
            axes[0, col_idx].set_ylabel('t-SNE Component 2', fontsize=9)
            axes[0, col_idx].legend(*scatter1.legend_elements(), title="Digits", fontsize=8)

            # Middle row: color by rotation
            scatter2 = axes[1, col_idx].scatter(space_2d[:, 0], space_2d[:, 1],
                                          c=r_labels, cmap='tab10',
                                          vmin=0, vmax=5, alpha=0.4)
            axes[1, col_idx].set_title(f'{title}\nColored by Rotation (Domain Variable)',
                                      fontsize=10)
            axes[1, col_idx].set_xlabel('t-SNE Component 1', fontsize=9)
            axes[1, col_idx].set_ylabel('t-SNE Component 2', fontsize=9)
            # Create custom legend for rotations
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=plt.cm.tab10(i/5),
                                        label=angle, markersize=10)
                             for i, angle in enumerate(rotation_angles)]
            axes[1, col_idx].legend(handles=legend_elements, title="Rotations", fontsize=8)
            
            # Bottom row: color by RGB color
            if len(c_labels.shape) > 1:
                rgb_colors = c_labels
            else:
                # Convert color indices to RGB
                rgb_colors = np.zeros((len(c_labels), 3))
                color_mappings = {
                    0: [0, 0, 1], 1: [0, 1, 0], 2: [1, 1, 0], 3: [0, 1, 1],
                    4: [1, 0, 1], 5: [1, 0.5, 0], 6: [1, 0, 0]
                }
                for i, color_idx in enumerate(c_labels):
                    rgb_colors[i] = color_mappings[color_idx]
            
            scatter3 = axes[2, col_idx].scatter(space_2d[:, 0], space_2d[:, 1],
                                          c=rgb_colors, alpha=0.4)
            axes[2, col_idx].set_title(f'{title}\nColored by Color (Spurious Variable)',
                                      fontsize=10)
            axes[2, col_idx].set_xlabel('t-SNE Component 1', fontsize=9)
            axes[2, col_idx].set_ylabel('t-SNE Component 2', fontsize=9)
            # Create custom legend for colors
            color_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, label=name, markersize=10)
                for color, name in zip(['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'red'],
                                     color_labels)
            ]
            axes[2, col_idx].legend(handles=color_elements, title="Colors", fontsize=8)

        # Adjust layout to accommodate suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            print(f"Latent space visualization saved to {save_path}")
        plt.close()
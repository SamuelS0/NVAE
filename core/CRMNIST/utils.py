import numpy as np
import random
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

random.seed(42)

class FixedRotate:
    def __init__(self, angle):
        # Convert angle to float if it's a string
        if isinstance(angle, str):
            try:
                self.angle = float(angle)
            except ValueError:
                raise ValueError(f"Invalid angle value: {angle}. Must be a number.")
        elif isinstance(angle, (int, float)):
            self.angle = float(angle)
        else:
            raise TypeError(f"Angle must be a number or string, got {type(angle)}")
        
        # Validate angle range (optional: normalize to 0-360)
        if not (-360 <= self.angle <= 360):
            raise ValueError(f"Angle {self.angle} is outside reasonable range [-360, 360]")
    
    def __call__(self, img):
        """
        Rotates an image.

        Args:
            img (PIL.Image.Image or torch.Tensor): image to be rotated
        Returns:
            img (PIL.Image.Image or torch.Tensor): rotated image in RGB format (3 channels)
        """
        if img is None:
            raise ValueError("Input image cannot be None")
            
        # Handle tensor input by converting to PIL
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Save original properties
            original_dtype = img.dtype
            original_device = img.device
            
            # Validate tensor
            if img.numel() == 0:
                raise ValueError("Input tensor is empty")
            
            # Special handling for different tensor formats
            if img.dim() == 3 and img.shape[0] == 3:
                # Already RGB, just make PIL image
                pil_img = transforms.ToPILImage()(img.cpu())
            elif img.dim() == 2:  # Single channel grayscale
                # Convert to PIL (single channel)
                pil_img = transforms.ToPILImage()(img.unsqueeze(0).cpu())
            elif img.dim() == 3 and img.shape[0] == 1:
                # Single channel with dimension
                pil_img = transforms.ToPILImage()(img.cpu())
            elif img.dim() == 3:
                # For other 3D tensors, assume [H,W,C] format
                img = img.permute(2, 0, 1)  # Convert to [C,H,W]
                pil_img = transforms.ToPILImage()(img.cpu())
            else:
                raise ValueError(f"Unsupported tensor shape: {img.shape}")
        else:
            pil_img = img
        
        # Always convert to RGB mode for PIL images
        if hasattr(pil_img, 'mode') and pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        try:
            rotated_img = F.rotate(pil_img, self.angle, fill=(0, 0, 0))
        except Exception as e:
            raise RuntimeError(f"Failed to rotate image: {e}")
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert back to tensor - will be RGB format (3 channels)
            result = transforms.ToTensor()(rotated_img).to(original_device)
            
            # Do NOT convert back to grayscale - keep as RGB (3 channels)
            
            # If original was int type (0-255), convert back
            if original_dtype in [torch.uint8, torch.int8, torch.int16]:
                result = (result * 255).to(original_dtype)
            
            return result
        
        return rotated_img
    
class AddColor:
    """
    Adds color to an image with call function.
    """

    def __init__(self, color, intensity):
        if not isinstance(color, str):
            raise TypeError(f"Color must be a string, got {type(color)}")
            
        self.color = color.lower()
        
        # Validate intensity
        if not isinstance(intensity, (int, float)):
            raise TypeError(f"Intensity must be a number, got {type(intensity)}")
        if intensity < 0:
            raise ValueError(f"Intensity must be non-negative, got {intensity}")
        if intensity > 10:  # Reasonable upper bound
            raise ValueError(f"Intensity {intensity} is too high (max 10)")
            
        self.intensity = float(intensity)
        
        self.channel_map = {
            'red': [0],
            'green': [1],
            'blue': [2],
            'yellow': [0, 1],
            'magenta': [0, 2],
            'cyan': [1, 2],
            'orange': []
        }

        if self.color not in self.channel_map:
            valid_colors = list(self.channel_map.keys())
            raise ValueError(f"Invalid color '{self.color}'. Valid colors are: {valid_colors}")
    
    def __call__(self, img):
        """
        Adds color to img.
        
        Args:
            img (PIL.Image.Image or torch.Tensor): image to be colored
        Returns: 
            img (PIL.Image.Image or torch.Tensor): colored image in RGB format (3 channels)
        """
        if img is None:
            raise ValueError("Input image cannot be None")
            
        # Handle tensor input by converting to PIL
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Save original properties
            original_dtype = img.dtype
            original_device = img.device
            
            # Validate tensor
            if img.numel() == 0:
                raise ValueError("Input tensor is empty")
            
            # Special handling for different tensor formats
            if img.dim() == 3 and img.shape[0] == 3:
                # Already RGB, just make PIL image
                pil_img = transforms.ToPILImage()(img.cpu())
            elif img.dim() == 2:  # Single channel grayscale
                # Convert to PIL (single channel)
                pil_img = transforms.ToPILImage()(img.unsqueeze(0).cpu())
            elif img.dim() == 3 and img.shape[0] == 1:
                # Single channel with dimension
                pil_img = transforms.ToPILImage()(img.cpu())
            elif img.dim() == 3:
                # For other 3D tensors, assume [H,W,C] format
                img = img.permute(2, 0, 1)  # Convert to [C,H,W]
                pil_img = transforms.ToPILImage()(img.cpu())
            else:
                raise ValueError(f"Unsupported tensor shape: {img.shape}")
        else:
            pil_img = img
            
        # Always ensure we have an RGB image for coloring
        if hasattr(pil_img, 'mode') and pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        try:
            np_img = np.array(pil_img).astype(np.float32) / 255.0
        except Exception as e:
            raise RuntimeError(f"Failed to convert image to numpy array: {e}")

        for channel in range(3):
            if channel in self.channel_map[self.color]:
                np_img[..., channel] *= self.intensity

            elif self.color == 'orange':
                # Create orange by using different intensities for R, G, B channels
                if channel == 0:  # Red channel
                    np_img[..., channel] *= self.intensity * 1.0  # Full red
                elif channel == 1:  # Green channel
                    np_img[..., channel] *= self.intensity * 0.6  # Medium green
                else:  # Blue channel
                    np_img[..., channel] *= self.intensity * 0.1  # Very low blue
            else:
                np_img[..., channel] *= 0.5

                
        np_img = np.clip(np_img, 0, 1)
        
        try:
            colored_img = Image.fromarray((np_img * 255).astype(np.uint8))
        except Exception as e:
            raise RuntimeError(f"Failed to create colored image: {e}")
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert back to tensor - will always be RGB (3 channels)
            result = transforms.ToTensor()(colored_img).to(original_device)
            
            # Do NOT convert back to single channel - keep as RGB (3 channels)
            
            # If original was int type (0-255), convert back
            if original_dtype in [torch.uint8, torch.int8, torch.int16]:
                result = (result * 255).to(original_dtype)
            
            return result
        
        return colored_img

def choose_label_subset(spec_data, chosen=False):
    """Generates y_c and subsets of five labels to be colored in each domain
    Args:
        spec_data: dict containing domain_data and other configuration
        chosen: boolean indicating if labels were already chosen

    Returns:
        y_c (int): The chosen label for special coloring
        subsets (dict): Dictionary mapping domain numbers to their subset of labels for coloring
    """

    # Ensure we have domain_data
    if 'domain_data' not in spec_data:
        raise ValueError("spec_data must contain 'domain_data' key")
        
    domain_data = spec_data['domain_data']

    if not chosen:
        y_c = random.choice(range(10))
        spec_data['y_c'] = y_c

        all_labels = list(range(10))
        if y_c in all_labels:
            all_labels.remove(y_c)

        spec_data['y_c'] = y_c
        subsets = {}
        for i in range(0,len(domain_data)):
            subsets[i] = random.sample(all_labels, 5)
            domain_data[i]['subset'] = subsets[i]

        return y_c, subsets
    else:
        # If already chosen, just return the existing y_c
        return spec_data.get('y_c')

def make_transform(domain_data, domain_number, style, transform_intensity = 1.5, transform_decay = 1):
    """
    Makes transform

    Args:
        domain_data (dict[int, dict]): Contains domain information in dict with keys: "rotation", "color"
                                    "intensity", "name", "number".
        domain_number (int): number label of domain for which to make the transform
        transform_intensity (float): intensity scalar
        transform_decay (float): decay scalar

    Returns:
        transform: transform for diven domain

    """
    domain_params = domain_data[domain_number]

    if style == 'domain_color':
        transform = transforms.Compose([
                    AddColor(domain_params['color'],
                             domain_params['intensity'] * transform_decay)
                    # transforms.Normalize((0.5, 0.5, 0.5),   #RGB mean
                    #                      (0.5, 0.5, 0.5))   #RGB std
                    ])
        return transform
        
    elif style == 'red' or style == 'unique_color':
        # Always use 'red' for unique color since that's what we've added to class_map
        transform = transforms.Compose([
                    AddColor('red',  # Use red for unique color
                             domain_params['intensity'] * transform_decay)
                    ])
        return transform
        
    elif style == 'rotate_only':
        # Convert rotation to int before applying decay
        rotation = int(domain_params['rotation'])
        transform = transforms.Compose([
                    FixedRotate(rotation * transform_decay)
                    # transforms.Normalize((0.5, 0.5, 0.5),   #RGB mean
                    #                      (0.5, 0.5, 0.5))   #RGB std
                    ])
        return transform

    # Default return in case style is not recognized
    return transforms.Compose([])

def rand_check(p):
    return random.uniform(0,1) >= p

def get_data(root='./data', train=True, download=True):
    """
    Download and return the MNIST dataset.
    
    Args:
        root (str): Directory to store the dataset
        train (bool): If True, returns the training set, otherwise the test set
        download (bool): If True, downloads the dataset
        
    Returns:
        MNIST dataset
    """
    try:
        return datasets.MNIST(root=root, train=train, download=download)
    except Exception as e:
        print(f"Error downloading MNIST dataset: {e}")
        return None

# Only load if needed, not at import time
def load_mnist():
    return get_data()

mnist = load_mnist()

def select_diverse_sample_batch(loader, args, samples_per_domain=10):
    """
    Select a diverse batch of samples with equal representation from each domain.
    Returns exactly 10 samples per rotation domain (0-5).
    
    Args:
        loader: DataLoader to select samples from
        args: Arguments from command line
        samples_per_domain: Number of samples to select per domain (default: 10)
        
    Returns:
        Tuple of (images, labels, color_labels, rotation_labels)
    """
    # Initialize dictionaries to store samples for each rotation domain
    rotation_samples = {i: [] for i in range(6)}  # 6 rotation domains (0-5)
    red_samples = []  # Track red images separately
    
    # Initialize lists to store all samples
    all_x, all_y, all_c, all_r = [], [], [], []
    
    # Track how many images we've processed
    images_processed = 0
    
    # Iterate through batches until we have enough samples or reach the end
    for batch_idx, (x, y, c, r) in enumerate(loader):
        images_processed += len(x)
        
        # Check domain assignments for each image in the batch
        for i in range(len(x)):
            # Check rotation domain - each sample belongs to one rotation domain
            if torch.max(r[i]) > 0:  # If there's a valid rotation assignment
                rotation_idx = torch.argmax(r[i]).item()
                
                # Verify this is a valid domain index
                if rotation_idx < 6:  # We have 6 rotation domains (0-5)
                    # Store this sample in its rotation domain
                    rotation_samples[rotation_idx].append((x[i], y[i], c[i], r[i]))
                else:
                    print(f"Warning: Invalid rotation domain index {rotation_idx} found. Skipping sample.")
            
            # Also check if it's a red image (these might overlap with rotation domains)
            if torch.max(c[i]) > 0 and torch.argmax(c[i]).item() == 6:  # Red is index 6
                red_samples.append((x[i], y[i], c[i], r[i]))
        
        # Print counts for debugging
        domain_counts = {domain: len(samples) for domain, samples in rotation_samples.items()}
        print(f"Processed {images_processed} images: {domain_counts}")
        print(f"Red images found: {len(red_samples)}")
        
        # Check if we have enough samples from each domain to proceed
        min_samples = min([len(samples) for samples in rotation_samples.values()])
        if min_samples >= samples_per_domain:
            print(f"Found at least {samples_per_domain} samples in each domain. Proceeding with selection.")
            break
    
    # If we didn't get enough samples, just use the first batch
    if min_samples < samples_per_domain:
        print(f"Warning: Could not find {samples_per_domain} samples in each domain.")
        print(f"Using the entire first batch instead.")
        return next(iter(loader))
    
    # Randomly select exactly samples_per_domain samples from each domain
    selected_x, selected_y, selected_c, selected_r = [], [], [], []
    
    for domain in range(6):
        domain_count = len(rotation_samples[domain])
        print(f"Domain {domain}: Selecting {samples_per_domain} samples from {domain_count} available")
        
        # Randomly sample without replacement
        if domain_count > samples_per_domain:
            selected_indices = random.sample(range(domain_count), samples_per_domain)
            selected_samples = [rotation_samples[domain][i] for i in selected_indices]
        else:
            # If we don't have enough, use what we have
            selected_samples = rotation_samples[domain][:samples_per_domain]
        
        # Add to our lists
        for x_i, y_i, c_i, r_i in selected_samples:
            selected_x.append(x_i)
            selected_y.append(y_i)
            selected_c.append(c_i)
            selected_r.append(r_i)
    
    # Convert lists to tensors
    selected_x = torch.stack(selected_x)
    selected_y = torch.stack(selected_y)
    selected_c = torch.stack(selected_c)
    selected_r = torch.stack(selected_r)
    
    # Verify selection has the right distribution
    rotation_counts = {}
    for i in range(len(selected_r)):
        rot_idx = torch.argmax(selected_r[i]).item()
        rotation_counts[rot_idx] = rotation_counts.get(rot_idx, 0) + 1
    
    # Print final counts
    print(f"Final selection: {len(selected_x)} total images")
    print(f"Domain distribution in selected batch: {rotation_counts}")
    
    return (selected_x, selected_y, selected_c, selected_r)

def visualize_reconstructions(model, epoch, batch_data, args, reconstructions_dir):
    """
    Visualize original images and their reconstructions, organized by domain.
    Each domain shows 10 samples with their reconstructions.
    
    Args:
        epoch: Current epoch number
        batch_data: Tuple of (x, y, c, r) tensors
    """
    x, y, c, r = batch_data
    if args.cuda:
        x, y, c, r = x.to(args.device), y.to(args.device), c.to(args.device), r.to(args.device)
    
    model.eval()
    with torch.no_grad():
        x_recon, _, _, _, _, _, _, _, _, _, _, _, _ = model.forward(y, x, r)
    
    # Get labels in the right format
    if len(y.shape) > 1 and y.shape[1] > 1:
        # One-hot format
        y_labels = y.max(1)[1].cpu().numpy()
    else:
        # Integer format
        y_labels = y.cpu().numpy()
    
    # Get color and rotation information
    color_indices = torch.argmax(c, dim=1).cpu().numpy() if torch.max(c) > 0 else np.zeros(len(x))
    rotation_indices = torch.argmax(r, dim=1).cpu().numpy() if torch.max(r) > 0 else np.zeros(len(x))
    
    # Create a figure with subplots for each domain
    num_domains = 6  # 6 rotation domains
    samples_per_domain = 10  # 10 samples per domain
    
    # Create a figure with 6 domains, each with 2 rows (original & reconstruction) and 10 columns (samples)
    fig, axes = plt.subplots(num_domains * 2, samples_per_domain, figsize=(20, 4 * num_domains))
    
    # Get rotation angles for titles
    rotation_angles = ['0°', '15°', '30°', '45°', '60°', '75°']
    
    # Color mapping for labels
    color_map = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'cyan', 4: 'magenta', 5: 'orange', 6: 'red'}
    
    # Organize images by domain
    domain_images = {i: [] for i in range(num_domains)}
    
    # Group images by their rotation domain
    for i in range(len(x)):
        domain = rotation_indices[i]
        if domain < num_domains:  # Ensure valid domain
            domain_images[domain].append(i)
    
    # Process each domain
    for domain in range(num_domains):
        domain_indices = domain_images[domain]
        
        # If we have enough images for this domain, select the exact number needed
        num_samples = min(samples_per_domain, len(domain_indices))
        
        # Print domain stats
        print(f"Domain {domain} ({rotation_angles[domain]}): {len(domain_indices)} images available, using {num_samples}")
        
        # Use consecutive samples since they're already randomly selected in select_diverse_sample_batch
        for i in range(num_samples):
            idx = domain_indices[i]
            
            # Check if this image has a valid color transformation
            has_color = torch.max(c[idx]) > 0
            if has_color:
                color_idx = torch.argmax(c[idx]).item()
                color = color_map.get(color_idx, 'none')
                color_label = f'Color: {color}'
            else:
                color_label = 'Color: none'
            
            # Display original (top row)
            img = x[idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[domain * 2, i].imshow(img)
            axes[domain * 2, i].set_title(f'Original\nDigit: {y_labels[idx]}\n{color_label}')
            axes[domain * 2, i].axis('off')
            
            # Display reconstruction (bottom row)
            img = x_recon[idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[domain * 2 + 1, i].imshow(img)
            axes[domain * 2 + 1, i].set_title(f'Recon\nDigit: {y_labels[idx]}\n{color_label}')
            axes[domain * 2 + 1, i].axis('off')
        
        # Fill remaining slots with empty plots
        for i in range(num_samples, samples_per_domain):
            axes[domain * 2, i].axis('off')
            axes[domain * 2 + 1, i].axis('off')
        
        # Add rotation angle as y-label for the domain
        axes[domain * 2, 0].set_ylabel(f'{rotation_angles[domain]}\nOriginal')
        axes[domain * 2 + 1, 0].set_ylabel(f'{rotation_angles[domain]}\nRecon')
    
    plt.suptitle(f'Reconstructions - Epoch {epoch}', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(reconstructions_dir, f'epoch_{epoch}.png'))
    plt.close()
    
    print(f"Saved reconstructions visualization for epoch {epoch}")

#Function to visualize conditional generation
def visualize_conditional_generation(model, device, output_dir):
    """Generate and visualize samples conditioned on digit classes"""
    model.eval()
    
    # Generate one image per class (digits 0-9)
    samples_per_class = 5
    
    # First, generate samples for each digit class
    plt.figure(figsize=(12, 6))
    
    # Generate one row of samples for each digit class (0-9)
    for digit in range(10):
        # Generate samples for this digit
        images, labels = model.generate(y=digit, num_samples=samples_per_class, device=device)
        
        # Display the generated images
        for i in range(samples_per_class):
            plt.subplot(10, samples_per_class, digit * samples_per_class + i + 1)
            img = images[i].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            if i == 0:
                plt.ylabel(f"Digit {digit}")
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conditional_generations.png'))
    plt.close()
    
    print(f"Conditional generation visualization saved to {os.path.join(output_dir, 'conditional_generations.png')}")

def save_domain_samples_visualization(x, y, c, r, epoch, output_dir):
    """
    Save the selected samples as a visualization grid, with a special row for red images.
    
    Args:
        x: Images tensor
        y: Digit labels
        c: Color labels
        r: Rotation labels
        epoch: Current epoch number
        output_dir: Directory to save the visualization
    """
    # Create a figure with subplots for each domain plus red images
    num_domains = 6  # 6 rotation domains
    samples_per_domain = 10  # Fixed number for consistent layout
    
    # Add an extra row for red images
    fig, axes = plt.subplots(num_domains + 1, samples_per_domain, figsize=(20, 14))
    
    # Get rotation angles for titles
    rotation_angles = ['0°', '15°', '30°', '45°', '60°', '75°']
    
    # First, display red images in the top row if any
    red_images_found = 0
    for i in range(len(x)):
        if torch.max(c[i]) > 0 and torch.argmax(c[i]).item() == 6:  # Red is index 6
            if red_images_found < samples_per_domain:
                img = x[i].cpu().detach().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[0, red_images_found].imshow(img)
                axes[0, red_images_found].set_title(f'Red\nDigit: {y[i].item()}')
                axes[0, red_images_found].axis('off')
                red_images_found += 1
    
    # Fill remaining red image slots with empty plots
    for i in range(red_images_found, samples_per_domain):
        axes[0, i].axis('off')
    
    # Label the red images row
    axes[0, 0].set_ylabel('Red Images')
    
    # Now display rotation domain images
    for domain in range(num_domains):
        domain_images_found = 0
        for i in range(len(x)):
            if torch.max(r[i]) > 0 and torch.argmax(r[i]).item() == domain:
                if domain_images_found < samples_per_domain:
                    img = x[i].cpu().detach().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    
                    # Get color information
                    color_idx = torch.argmax(c[i]).item() if torch.max(c[i]) > 0 else -1
                    color_map = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'cyan', 4: 'magenta', 5: 'orange', 6: 'red'}
                    color = color_map.get(color_idx, 'none')
                    
                    axes[domain + 1, domain_images_found].imshow(img)
                    axes[domain + 1, domain_images_found].set_title(f'Digit: {y[i].item()}\nColor: {color}')
                    axes[domain + 1, domain_images_found].axis('off')
                    domain_images_found += 1
        
        # Fill remaining slots with empty plots
        for i in range(domain_images_found, samples_per_domain):
            axes[domain + 1, i].axis('off')
        
        # Add rotation angle as y-label
        axes[domain + 1, 0].set_ylabel(f'{rotation_angles[domain]}')
    
    plt.suptitle(f'Domain Samples - Epoch {epoch}', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'domain_samples_epoch_{epoch}.png'))
    plt.close()
    
    print(f"Saved domain samples visualization for epoch {epoch}")

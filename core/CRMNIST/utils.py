import numpy as np
import random
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import datasets, transforms

random.seed(42)

class FixedRotate:
    def __init__(self, angle):
        # Convert angle to float if it's a string
        if isinstance(angle, str):
            self.angle = float(angle)
        else:
            self.angle = angle
    
    def __call__(self, img):
        """
        Rotates an image.

        Args:
            img (PIL.Image.Image or torch.Tensor): image to be rotated
        Returns:
            img (PIL.Image.Image or torch.Tensor): rotated image in RGB format (3 channels)
        """
        # Handle tensor input by converting to PIL
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Save original properties
            original_dtype = img.dtype
            original_device = img.device
            
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
        
        rotated_img = F.rotate(pil_img, self.angle, fill=(0, 0, 0))
        
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
        self.color = color.lower()
        self.intensity = intensity
        self.channel_map = {
            'red': [0],
            'green': [1],
            'blue': [2],
            'yellow': [0, 1],
            'magenta': [0, 2],
            'cyan': [1, 2]
        }

        if self.color not in self.channel_map:
            raise ValueError("invalid color")
    
    def __call__(self, img):
        """
        Adds color to img.
        
        Args:
            img (PIL.Image.Image or torch.Tensor): image to be colored
        Returns: 
            img (PIL.Image.Image or torch.Tensor): colored image in RGB format (3 channels)
        """
        # Handle tensor input by converting to PIL
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Save original properties
            original_dtype = img.dtype
            original_device = img.device
            
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
            
        np_img = np.array(pil_img).astype(np.float32) / 255.0

        for channel in range(3):
            if channel in self.channel_map[self.color]:
                np_img[..., channel] *= self.intensity
            else:
                np_img[..., channel] *= 0.5
                
        np_img = np.clip(np_img, 0, 1)
        colored_img = Image.fromarray((np_img * 255).astype(np.uint8))
        
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
        all_labels = list(range(10))
        if y_c in all_labels:
            all_labels.remove(y_c)
        
        subsets = {}
        for i in range(len(domain_data)):
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

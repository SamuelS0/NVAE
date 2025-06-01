import os
import numpy as np
import torch
import torch.utils.data as data_utils
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional, ToTensor
from PIL import Image
import torch.nn.functional as F
import core.CRMNIST.utils_crmnist

data_path = os.path.join('.','data')
crmnist_path = os.path.join(data_path,'crmnist')


class CRMNISTDataset(Dataset):
    def __init__(self, imgs, y_labels, c_labels, r_labels, transform=None, convert_y_to_one_hot=False):
        """
        Custom dataset for CRMNIST.

        Args:
            imgs (list[torch.Tensor]): List of tensors for images per domain.
            y_labels (list[torch.Tensor]): List of tensors for corresponding labels.
            c_labels (list[torch.Tensor]): Color transformation labels per domain.
            r_labels (list[torch.Tensor]): Rotation transformation labels per domain.
            transform (callable, optional): Optional transform to be applied.
            convert_y_to_one_hot (bool): Whether to convert integer y_labels to one-hot format.
        """
        self.num_y_classes = 10
        self.num_c_classes = c_labels[0].shape[1] if len(c_labels) > 0 else 6
        self.num_r_classes = r_labels[0].shape[1] if len(r_labels) > 0 else 5

        # Debug print to check array sizes before concatenation
        img_sizes = [img.shape[0] for img in imgs]
        c_label_sizes = [label.shape[0] for label in c_labels]
        r_label_sizes = [label.shape[0] for label in r_labels]
        y_label_sizes = [label.shape[0] for label in y_labels]
        
        print(f"Debug - Image sizes per domain: {img_sizes}, total: {sum(img_sizes)}")
        print(f"Debug - Color label sizes per domain: {c_label_sizes}, total: {sum(c_label_sizes)}")
        print(f"Debug - Rotation label sizes per domain: {r_label_sizes}, total: {sum(r_label_sizes)}")
        print(f"Debug - Y label sizes per domain: {y_label_sizes}, total: {sum(y_label_sizes)}")

        # Find any size mismatches
        for i, (img_size, c_size, r_size, y_size) in enumerate(zip(img_sizes, c_label_sizes, r_label_sizes, y_label_sizes)):
            if img_size != c_size or img_size != r_size or img_size != y_size:
                print(f"Size mismatch in domain {i}: imgs={img_size}, c_labels={c_size}, r_labels={r_size}, y_labels={y_size}")

        self.imgs = torch.cat(imgs)
        self.y_labels = torch.cat(y_labels)
        self.c_labels = torch.cat(c_labels)
        self.r_labels = torch.cat(r_labels)
        
        # Final check
        print(f"Final sizes: imgs={len(self.imgs)}, c_labels={len(self.c_labels)}, r_labels={len(self.r_labels)}, y_labels={len(self.y_labels)}")
        
        self.transform = transform
        self.convert_y_to_one_hot = convert_y_to_one_hot


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        y_label = self.y_labels[idx]
        c_label = self.c_labels[idx]
        r_label = self.r_labels[idx]

        if self.transform:
            img = self.transform(img)
            
        # Convert y_label to one-hot if requested
        if self.convert_y_to_one_hot:
            y_one_hot = torch.zeros(self.num_y_classes)
            y_one_hot[y_label] = 1
            y_label = y_one_hot
        
        return img, y_label, c_label, r_label



def generate_crmnist_dataset(spec_data, train, transform_intensity=1.5, transform_decay=1, p=0.5):
    """
    Generates CRMNIST dataset.
    
    Args:
        domain_data (dict[int, dict]): Contains domain information in dict with keys: "rotation", "color"
                                    "intensity", "name", "number", "y_c", "subset"
        train (bool): Whether to generate training or test dataset
        transform_intensity (float): intensity of transform
        transform_decay (float): decay of transform
        p: probability with which an image which may be colored is colored
    Returns:
        crmnist_dataset (dataset): 
            - if 'train' is 'true', returns training dataset.
            - if 'train' is 'false, returns test dataset.
        crmnist_domain_data: Modified domain dictionary that contains additional keys with transform info
    """

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(crmnist_path, exist_ok=True)

    mnist_train = datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
    
    if train:
        mnist_data = mnist_train
        print(f"Using training dataset with {len(mnist_train)} images")
    else:
        mnist_data = mnist_test
        print(f"Using test dataset with {len(mnist_test)} images")
    
    # Get images and labels - MNIST labels are already integers 0-9
    mnist_imgs = mnist_data.data.unsqueeze(1)  # Add channel dim
    mnist_labels = mnist_data.targets
    
    # Convert grayscale images to RGB (3 channels)
    # First normalize to [0,1] range
    mnist_imgs = mnist_imgs.float() / 255.0
    # Then expand to 3 channels [B, 1, H, W] -> [B, 3, H, W]
    mnist_imgs = mnist_imgs.repeat(1, 3, 1, 1)
    
    # imgs and labels are lists containing tensors of images and y labels,
    # each index corresponds to a domain
    imgs, labels, c_labels, r_labels = [],[],[],[]

    domain_data = spec_data['domain_data']
    class_map = spec_data['class_map']  # Use class_map from configuration
    
    num_domains = len(domain_data)
    y_c = spec_data['y_c']
    unique_color = spec_data['unique_color']
    
    print(f"\nDataset configuration:")
    print(f"Special digit (y_c) chosen for red color: {y_c}")
    print(f"Unique color: {unique_color}")
    print(f"Number of domains: {num_domains}")
    
    # Track statistics for red images
    total_red_images = 0
    red_images_per_domain = {i: 0 for i in range(num_domains)}

    # transform data for each domain
    for i in range(num_domains):
        r_transform = core.CRMNIST.utils_crmnist.make_transform(domain_data, i, 'rotate_only', transform_intensity, transform_decay)
        c_transform = core.CRMNIST.utils_crmnist.make_transform(domain_data, i, 'domain_color', transform_intensity, transform_decay)
        red_transform = core.CRMNIST.utils_crmnist.make_transform(domain_data, i, 'unique_color', transform_intensity, transform_decay)

        labels_subset = torch.tensor(domain_data[i]['subset'])

        subset_mask = torch.isin(mnist_labels, labels_subset)
        subset_imgs = mnist_imgs[subset_mask]
        subset_labels = mnist_labels[subset_mask]
        non_subset_imgs = mnist_imgs[~subset_mask]
        non_subset_labels = mnist_labels[~subset_mask]

        yc_imgs_mask = torch.isin(non_subset_labels, torch.tensor([y_c]))
        yc_non_subset_imgs = non_subset_imgs[yc_imgs_mask]
        yc_non_subset_labels = non_subset_labels[yc_imgs_mask]
        non_yc_non_subset_imgs = non_subset_imgs[~yc_imgs_mask]
        non_yc_non_subset_labels = non_subset_labels[~yc_imgs_mask]
        
        print(f"\nDomain {i}:")
        print(f"  Rotation: {domain_data[i]['rotation']}°")
        print(f"  Color: {domain_data[i]['color']}")
        print(f"  Subset size: {len(subset_imgs)}")
        print(f"  y_c images in subset: {len(yc_non_subset_imgs)}")
        print(f"  y_c images not in subset: {len(non_yc_non_subset_imgs)}")

        # Get unique color - handle if it's a list
        unique_color_val = unique_color[0] if isinstance(unique_color, list) else unique_color

        # Initialize label arrays with exact known lengths
        total_domain_images = len(subset_imgs) + len(yc_non_subset_imgs) + len(non_yc_non_subset_imgs)
        domain_c_labels = [None] * total_domain_images
        domain_r_labels = [None] * total_domain_images
        
        # Track red images in this domain
        domain_red_count = 0
        
        # Track current index for proper label assignment
        current_idx = 0

        # Process y_c images - apply color transformations
        for j in range(len(yc_non_subset_imgs)):
            color = None
            # Fix probability logic: apply transformation when random < p
            if random.uniform(0,1) < p:
                yc_non_subset_imgs[j] = red_transform(yc_non_subset_imgs[j])
                color = unique_color_val
                domain_red_count += 1
                total_red_images += 1
                if domain_red_count <= 5:  # Print details for first 5 red images in this domain
                    print(f"  Created red image with digit {yc_non_subset_labels[j].item()} in domain {i}")
            
            domain_c_labels[current_idx + j] = color
        
        current_idx += len(yc_non_subset_imgs)
        
        # Add labels for non-y_c images
        for j in range(len(non_yc_non_subset_imgs)):
            domain_c_labels[current_idx + j] = None  # No color transform
            
        current_idx += len(non_yc_non_subset_imgs)
        
        # Process subset images - apply color transformations
        for j in range(len(subset_imgs)):
            color = None

            if random.uniform(0,1) < p:
                subset_imgs[j] = c_transform(subset_imgs[j])
                color = domain_data[i]['color']
            
            domain_c_labels[current_idx + j] = color

        red_images_per_domain[i] = domain_red_count
        print(f"  Red images created in this domain: {domain_red_count}")
        
        # Concatenate all images from this domain
        domain_imgs = torch.cat([yc_non_subset_imgs, non_yc_non_subset_imgs, subset_imgs])
        domain_y_labels = torch.cat([yc_non_subset_labels, non_yc_non_subset_labels, subset_labels])
        
        rotation_value = domain_data[i]['rotation']
        for j in range(len(domain_imgs)):
            domain_imgs[j] = r_transform(domain_imgs[j])
            domain_r_labels[j] = rotation_value
        
        # Verification step to ensure label arrays match image count
        assert len(domain_c_labels) == len(domain_imgs), f"Color labels mismatch in domain {i}: {len(domain_c_labels)} vs {len(domain_imgs)}"
        assert len(domain_r_labels) == len(domain_imgs), f"Rotation labels mismatch in domain {i}: {len(domain_r_labels)} vs {len(domain_imgs)}"
                
        # Initialize tensors to store indices instead of one-hot directly
        domain_c_indices = torch.full((len(domain_c_labels),), -1, dtype=torch.long)
        domain_r_indices = torch.full((len(domain_r_labels),), -1, dtype=torch.long)
        
        for idx, color in enumerate(domain_c_labels):
            if color is not None and color in class_map:
                domain_c_indices[idx] = class_map[color]
                
        for idx, rotation in enumerate(domain_r_labels):
            # All rotation values should be valid now
            if rotation in class_map:
                domain_r_indices[idx] = class_map[rotation]
            else:
                print(f"Warning: Invalid rotation value '{rotation}' - assigning domain rotation {domain_data[i]['rotation']}")
                # Fallback to domain rotation if for some reason the value is invalid
                domain_r_indices[idx] = class_map[domain_data[i]['rotation']]
        
        # Convert valid indices to one-hot, handling -1 (no transform) cases
        valid_c_mask = domain_c_indices >= 0
        valid_r_mask = domain_r_indices >= 0  # All rotations should be valid now
        
        # Get number of classes from the highest indices in class_map
        num_c_classes = max([v for k, v in class_map.items() if not k.isdigit()]) + 1
        num_r_classes = max([v for k, v in class_map.items() if k.isdigit()]) + 1
        
        # Verify all rotation indices are valid
        if not valid_r_mask.all():
            invalid_count = (~valid_r_mask).sum().item()
            print(f"Warning: {invalid_count} rotation indices in domain {i} are invalid. Fixing...")
            domain_r_indices[~valid_r_mask] = class_map[domain_data[i]['rotation']]
            valid_r_mask = domain_r_indices >= 0  # Update mask after fixing
        
        domain_c_labels_tensor = torch.zeros(len(domain_c_labels), num_c_classes)
        domain_r_labels_tensor = torch.zeros(len(domain_r_labels), num_r_classes)
        
        if valid_c_mask.any():
            # Convert one_hot output to float to match destination tensor type
            one_hot_c = F.one_hot(domain_c_indices[valid_c_mask], num_classes=num_c_classes).float()
            domain_c_labels_tensor[valid_c_mask] = one_hot_c
            
        if valid_r_mask.any():
            # Convert one_hot output to float to match destination tensor type
            one_hot_r = F.one_hot(domain_r_indices[valid_r_mask], num_classes=num_r_classes).float()
            domain_r_labels_tensor[valid_r_mask] = one_hot_r

        imgs.append(domain_imgs)
        labels.append(domain_y_labels)  # Keep as integer labels (0-9)
        c_labels.append(domain_c_labels_tensor)  # One-hot encoded
        r_labels.append(domain_r_labels_tensor)  # One-hot encoded
    
    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"Total red images created: {total_red_images}")
    for domain, count in red_images_per_domain.items():
        print(f"Red images in domain {domain}: {count}")
    
    # Create dataset with integer y labels and one-hot c/r labels
    dataset = CRMNISTDataset(imgs, labels, c_labels, r_labels, convert_y_to_one_hot=False)
    return dataset


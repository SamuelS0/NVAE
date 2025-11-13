import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from typing import Dict
import datetime

def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--methods', '-m', nargs='+', type=str.lower)
    parser.add_argument('--hidden_dim', '-hd', type=int)
    parser.add_argument('--steps', '-s', type=int)
    parser.add_argument('--out', '-o', type=pathlib.Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)

    return parser

#def visualize_batch(images, labels, domain_dict):
    #domain_labels is tensor of integer domain labels for each image in batch
    digits, domain_labels = labels

    #map integers to domain names
    domains_name_dict = {id: name for id, name in enumerate(domain_dict.keys())}


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        # Handle both a single size parameter or multiple arguments
        if len(args) == 1 and (isinstance(args[0], tuple) or isinstance(args[0], list)):
            # If a single tuple/list is passed
            self.size = args[0]
        else:
            # If multiple separate arguments are passed
            self.size = args

    def forward(self, tensor):
        # Ensure input is a tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # For -1 size, we need to infer the shape from other dimensions
        if isinstance(self.size, tuple) and self.size[0] == -1:
            # First dim is batch size (preserved), remaining dims are flattened
            batch_size = tensor.size(0)
            return tensor.view(batch_size, -1)
        
        # Try to reshape, with better error handling
        try:
            return tensor.view(self.size)
        except RuntimeError:
            current_shape = list(tensor.shape)
            raise RuntimeError(f"Cannot reshape tensor of shape {current_shape} to {self.size}")

# NOTE: The kl_divergence function was removed because it's never used in the codebase.
# Models use PyTorch's built-in log_prob() method for KL divergence calculation instead.
# If you need KL divergence, use: torch.distributions.kl_divergence(p, q)

# Function to calculate additional metrics
def _calculate_metrics(model, y, x, r, mode='train'):
    """Calculate metrics for a batch, handling both generative and discriminative models."""
    with torch.no_grad():
        # Check if this is a DANN model by looking for the dann_forward method
        if hasattr(model, 'dann_forward'):
            # Handle DANN model (AugmentedDANN)
            outputs = model.dann_forward(x)

            # Convert one-hot to indices if necessary
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_true = torch.argmax(y, dim=1)
            else:
                y_true = y.long()

            if len(r.shape) > 1 and r.shape[1] > 1:
                a_true = torch.argmax(r, dim=1)
            else:
                a_true = r.long()

            # Main task accuracies
            y_pred = outputs['y_pred_main'].argmax(dim=1)
            y_accuracy = (y_pred == y_true).float().mean().item()

            a_pred = outputs['d_pred_main'].argmax(dim=1)
            a_accuracy = (a_pred == a_true).float().mean().item()

            return {
                'recon_mse': 0.0,  # DANN doesn't do reconstruction
                'y_accuracy': y_accuracy,
                'a_accuracy': a_accuracy
            }
        else:
            # Handle generative models (NVAE, DIVA)
            x_recon, _, _, _, _, _, _, y_hat, a_hat, _, _, _, _ = model.forward(y, x, r)

            # Reconstruction MSE
            recon_mse = torch.nn.functional.mse_loss(x_recon, x).item()

            # Classification accuracy
            _, y_pred = y_hat.max(1)
            if len(y.shape) > 1 and y.shape[1] > 1:
                _, y_true = y.max(1)
            else:
                y_true = y.long()
            y_accuracy = (y_pred == y_true).float().mean().item()

            # Attribute accuracy
            _, a_pred = a_hat.max(1)
            if len(r.shape) > 1 and r.shape[1] > 1:
                _, a_true = r.max(1)
            else:
                a_true = r.long()
            a_accuracy = (a_pred == a_true).float().mean().item()

            return {
                'recon_mse': recon_mse,
                'y_accuracy': y_accuracy,
                'a_accuracy': a_accuracy
            }

def sample_nvae(model, dataloader, num_samples, device):
    """
    Sample data from NVAE model and collect domain information.
    
    Args:
        model: NVAE model
        dataloader: DataLoader containing (x, y, c, r) tuples
        num_samples: Maximum number of samples to collect
        device: torch device
    
    Returns:
        zy_list, za_list, zay_list, zx_list: Lists of latent vectors
        y_list: List of digit labels
        domain_dict: Dictionary of domain labels
        labels_dict: Dictionary of domain label names
    """
    model.eval()
    zy_list, za_list, zay_list, zx_list = [], [], [], []
    y_list = []
    domain_dict = {"color": [], 'rotation': []}
    labels_dict = {
        "color": ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red'],
        'rotation': ['0°', '15°', '30°', '45°', '60°', '75°']
    }
    
    # Initialize counters for each combination
    counts = {}
    for digit in range(10):
        for color in range(7):
            for rotation in range(6):
                counts[(digit, color, rotation)] = 0
    
    # Target samples per combination - ensure we get enough for visualization
    target_samples = 50  # Minimum samples per combination for good visualization (matches DIVA/DANN)
    total_collected = 0
    all_combinations_satisfied = False
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Sampling NVAE data")
        for x, y, c, r in pbar:
            # Only check max_samples after we have enough samples for each combination
            if all_combinations_satisfied and total_collected >= num_samples:
                break
                
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
            
            # Check each sample
            for j in range(len(y)):
                digit = y[j].item()
                color = c[j].item()
                rotation = r[j].item()
                
                # Check if we need more samples for this specific combination
                combination = (digit, color, rotation)
                should_keep = counts[combination] < target_samples
                
                if should_keep:
                    keep_mask[j] = True
                    counts[combination] += 1
                    total_collected += 1
            
            # Apply mask to get only samples we want to keep
            if keep_mask.any():
                x_batch = x[keep_mask]
                y_batch = y[keep_mask]
                c_batch = c[keep_mask]
                r_batch = r[keep_mask]
                
                # Get latent representations
                z_loc, _ = model.qz(x_batch)
                zy = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                zay = z_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
                za = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
                
                # Store latent vectors and labels
                zy_list.append(zy.cpu())
                za_list.append(za.cpu())
                zay_list.append(zay.cpu())
                zx_list.append(zx.cpu())
                y_list.append(y_batch.cpu())
                domain_dict["color"].append(c_batch.cpu())
                domain_dict["rotation"].append(r_batch.cpu())
            
            # Check if we have enough samples for all combinations
            all_combinations_satisfied = all(count >= target_samples for count in counts.values())
            
            # Update progress bar with more detailed information
            min_count = min(counts.values())
            max_count = max(counts.values())
            pbar.set_postfix({
                'total': total_collected,
                'min_count': min_count,
                'max_count': max_count,
                'target': target_samples,
                'satisfied': all_combinations_satisfied
            })
    
    # Print detailed statistics about the sampling
    print("\nSampling Statistics:")
    print(f"Total samples collected: {total_collected}")
    print(f"Target samples per combination: {target_samples}")
    print(f"Minimum samples per combination: {min(counts.values())}")
    print(f"Maximum samples per combination: {max(counts.values())}")
    
    # Print distribution of samples across digits
    digit_counts = {i: 0 for i in range(10)}
    for (digit, _, _), count in counts.items():
        digit_counts[digit] += count
    print("\nSamples per digit:")
    for digit, count in digit_counts.items():
        print(f"Digit {digit}: {count} samples")
    
    # Print distribution of samples across colors
    color_counts = {i: 0 for i in range(7)}
    for (_, color, _), count in counts.items():
        color_counts[color] += count
    print("\nSamples per color:")
    for color, count in color_counts.items():
        print(f"{labels_dict['color'][color]}: {count} samples")
    
    # Print distribution of samples across rotations
    rotation_counts = {i: 0 for i in range(6)}
    for (_, _, rotation), count in counts.items():
        rotation_counts[rotation] += count
    print("\nSamples per rotation:")
    for rotation, count in rotation_counts.items():
        print(f"{labels_dict['rotation'][rotation]}: {count} samples")
    
    return zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict

def sample_diva(model, dataloader, num_samples, device):
    """
    Sample data from DIVA model and collect domain information.
    DIVA is similar to NVAE but doesn't have a zay latent space.
    
    Args:
        model: DIVA model
        dataloader: DataLoader containing (x, y, c, r) tuples
        num_samples: Maximum number of samples to collect
        device: torch device
    
    Returns:
        zy_list, za_list, zay_list, zx_list: Lists of latent vectors
        y_list: List of digit labels
        domain_dict: Dictionary of domain labels
        labels_dict: Dictionary of domain label names
    """
    model.eval()
    zy_list, za_list, zx_list = [], [], []
    y_list = []
    domain_dict = {"color": [], 'rotation': []}
    labels_dict = {
        "color": ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red'],
        'rotation': ['0°', '15°', '30°', '45°', '60°', '75°']
    }
    
    # Initialize counters for each combination
    counts = {}
    for digit in range(10):
        for color in range(7):
            for rotation in range(6):
                counts[(digit, color, rotation)] = 0
    
    # Target samples per combination - ensure we get enough for visualization
    target_samples = 50  # Minimum samples per combination for good visualization
    total_collected = 0
    all_combinations_satisfied = False
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Sampling DIVA data")
        for x, y, c, r in pbar:
            # Only check max_samples after we have enough samples for each combination
            if all_combinations_satisfied and total_collected >= num_samples:
                break
                
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
            
            # Check each sample
            for j in range(len(y)):
                digit = y[j].item()
                color = c[j].item()
                rotation = r[j].item()
                
                # Check if we need more samples for this specific combination
                combination = (digit, color, rotation)
                should_keep = counts[combination] < target_samples
                
                if should_keep:
                    keep_mask[j] = True
                    counts[combination] += 1
                    total_collected += 1
            
            # Apply mask to get only samples we want to keep
            if keep_mask.any():
                x_batch = x[keep_mask]
                y_batch = y[keep_mask]
                c_batch = c[keep_mask]
                r_batch = r[keep_mask]
                
                # Get latent representations
                z_loc, _ = model.qz(x_batch)
                zy = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                za = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
                
                # Store latent vectors and labels
                zy_list.append(zy.cpu())
                za_list.append(za.cpu())
                zx_list.append(zx.cpu())
                y_list.append(y_batch.cpu())
                domain_dict["color"].append(c_batch.cpu())
                domain_dict["rotation"].append(r_batch.cpu())
            
            # Check if we have enough samples for all combinations
            all_combinations_satisfied = all(count >= target_samples for count in counts.values())
            
            # Update progress bar with more detailed information
            min_count = min(counts.values())
            max_count = max(counts.values())
            pbar.set_postfix({
                'total': total_collected,
                'min_count': min_count,
                'max_count': max_count,
                'target': target_samples,
                'satisfied': all_combinations_satisfied
            })
    
    # Print detailed statistics about the sampling
    print("\nSampling Statistics:")
    print(f"Total samples collected: {total_collected}")
    print(f"Target samples per combination: {target_samples}")
    print(f"Minimum samples per combination: {min(counts.values())}")
    print(f"Maximum samples per combination: {max(counts.values())}")
    
    # Print distribution of samples across digits
    digit_counts = {i: 0 for i in range(10)}
    for (digit, _, _), count in counts.items():
        digit_counts[digit] += count
    print("\nSamples per digit:")
    for digit, count in digit_counts.items():
        print(f"Digit {digit}: {count} samples")
    
    # Print distribution of samples across colors
    color_counts = {i: 0 for i in range(7)}
    for (_, color, _), count in counts.items():
        color_counts[color] += count
    print("\nSamples per color:")
    for color, count in color_counts.items():
        print(f"{labels_dict['color'][color]}: {count} samples")
    
    # Print distribution of samples across rotations
    rotation_counts = {i: 0 for i in range(6)}
    for (_, _, rotation), count in counts.items():
        rotation_counts[rotation] += count
    print("\nSamples per rotation:")
    for rotation, count in rotation_counts.items():
        print(f"{labels_dict['rotation'][rotation]}: {count} samples")
    
    # Return None for zay_list since DIVA doesn't have it
    return zy_list, za_list, None, zx_list, y_list, domain_dict, labels_dict

def sample_dann(model, dataloader, num_samples, device, model_variant='dann'):
    """
    Sample data from DANN or AugmentedDANN model and collect domain information.

    Args:
        model: DANN or AugmentedDANN model
        dataloader: DataLoader containing (x, y, c, r) tuples
        num_samples: Maximum number of samples to collect
        device: torch device
        model_variant: 'dann' or 'dann_augmented' (affects progress bar description only)

    Returns:
        zy_list, za_list, zay_list, zx_list: Lists of latent vectors
        y_list: List of digit labels
        domain_dict: Dictionary of domain labels
        labels_dict: Dictionary of domain label names
    """
    model.eval()
    zy_list, zd_list, zdy_list = [], [], []
    y_list = []
    domain_dict = {"color": [], "rotation": []}  # DANN uses both color and rotation
    labels_dict = {
        "color": ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red'],
        "rotation": ['0°', '15°', '30°', '45°', '60°', '75°']
    }
    
    # Initialize counters for each combination
    counts = {}
    for digit in range(10):
        for color in range(7):
            for rotation in range(6):
                counts[(digit, color, rotation)] = 0
    
    # Target samples per combination - ensure we get enough for visualization
    target_samples = 50  # Minimum samples per combination for good visualization
    total_collected = 0
    all_combinations_satisfied = False

    with torch.no_grad():
        desc_prefix = "Augmented " if model_variant == 'dann_augmented' else ""
        pbar = tqdm(dataloader, desc=f"Sampling {desc_prefix}DANN data")
        for x, y, c, r in pbar:
            # Only check max_samples after we have enough samples for each combination
            if all_combinations_satisfied and total_collected >= num_samples:
                break
                
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
            
            # Check each sample
            for j in range(len(y)):
                digit = y[j].item()
                color = c[j].item()
                rotation = r[j].item()
                
                # Check if we need more samples for this specific combination
                combination = (digit, color, rotation)
                should_keep = counts[combination] < target_samples
                
                if should_keep:
                    keep_mask[j] = True
                    counts[combination] += 1
                    total_collected += 1
            
            # Apply mask to get only samples we want to keep
            if keep_mask.any():
                x_batch = x[keep_mask]
                y_batch = y[keep_mask]
                c_batch = c[keep_mask]
                r_batch = r[keep_mask]
                
                # Get latent representations from Augmented DANN
                zy, zd, zdy = model.extract_features(x_batch)
                
                # Store the actual latent spaces
                zy_list.append(zy.cpu())
                zd_list.append(zd.cpu())
                zdy_list.append(zdy.cpu())
                y_list.append(y_batch.cpu())
                domain_dict["color"].append(c_batch.cpu())
                domain_dict["rotation"].append(r_batch.cpu())
            
            # Check if we have enough samples for all combinations
            all_combinations_satisfied = all(count >= target_samples for count in counts.values()) if counts else False
            
            # Update progress bar with more detailed information
            min_count = min(counts.values()) if counts else 0
            max_count = max(counts.values()) if counts else 0
            pbar.set_postfix({
                'total': total_collected,
                'min_count': min_count,
                'max_count': max_count,
                'target': target_samples,
                'satisfied': all_combinations_satisfied
            })
    
    # Print detailed statistics about the sampling
    print("\nSampling Statistics:")
    print(f"Total samples collected: {total_collected}")
    print(f"Target samples per combination: {target_samples}")
    if counts:
        print(f"Minimum samples per combination: {min(counts.values())}")
        print(f"Maximum samples per combination: {max(counts.values())}")
    
    # Print distribution of samples across digits
    digit_counts = {i: 0 for i in range(10)}
    for (digit, _, _), count in counts.items():
        digit_counts[digit] += count
    print("\nSamples per digit:")
    for digit, count in digit_counts.items():
        print(f"Digit {digit}: {count} samples")
    
    # Print distribution of samples across colors
    color_counts = {i: 0 for i in range(7)}
    for (_, color, _), count in counts.items():
        color_counts[color] += count
    print("\nSamples per color:")
    for color, count in color_counts.items():
        print(f"{labels_dict['color'][color]}: {count} samples")
    
    # Print distribution of samples across rotations
    rotation_counts = {i: 0 for i in range(6)}
    for (_, _, rotation), count in counts.items():
        rotation_counts[rotation] += count
    print("\nSamples per rotation:")
    for rotation, count in rotation_counts.items():
        print(f"{labels_dict['rotation'][rotation]}: {count} samples")
    
    # Return in the expected format: zy, za, zay, zx
    # For DANN: zy=zy, za=zd, zay=zdy, zx=zd (domain features)
    return zy_list, zd_list, zdy_list, zd_list, y_list, domain_dict, labels_dict

def sample_wild(model, dataloader, max_samples=5000, device=None):
    """
    Sample from a WILD model and collect latent representations.
    
    Args:
        model: WILD model
        dataloader: DataLoader containing (x, y, domain) tuples
        max_samples: Maximum number of samples to collect
        device: torch device
        
    Returns:
        Lists of latent vectors and corresponding labels
    """
    model.eval()
    
    # Initialize lists to store latent vectors and labels
    zy_list, za_list, zay_list, zx_list = [], [], [], []
    y_list = []
    domain_dict = {"hospital": []}
    labels_dict = {"digit": [], "hospital": []}
    
    # Initialize counters for each combination
    counts = {}
    for digit in range(10):  # 10 digits
        for hospital in range(6):  # 6 hospitals
            counts[(digit, hospital)] = 0
    
    # Target samples per combination
    target_samples = min(50, max_samples // 60)  # 60 combinations total (10 digits * 6 hospitals)
    total_collected = 0
    
    # Collect data in batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Early stopping if we've collected enough samples
            if total_collected >= max_samples:
                break
                
            x, y, domain = process_batch(batch, device, dataset_type='wild')
            
            # Convert to indices if one-hot encoded
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            if len(domain.shape) > 1:
                domain = torch.argmax(domain, dim=1)
            
            # Create mask for samples we want to keep
            keep_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
            
            # Check each sample
            for j in range(len(y)):
                if total_collected >= max_samples:
                    break
                    
                digit = y[j].item()
                hospital = domain[j].item()
                
                # Only keep if we need more samples for this combination
                if counts[(digit, hospital)] < target_samples:
                    keep_mask[j] = True
                    counts[(digit, hospital)] += 1
                    total_collected += 1
            
            # Apply mask to get only samples we want to keep
            if keep_mask.any():
                x_batch = x[keep_mask]
                y_batch = y[keep_mask]
                domain_batch = domain[keep_mask]
                
                # Get latent representations for this batch
                z_loc, _ = model.qz(x_batch)
                zy = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                if model.diva:
                    zay = None
                else:
                    zay = z_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
                za = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
                
                # Move to CPU and store
                zy_list.append(zy.cpu())
                za_list.append(za.cpu())
                if not model.diva:
                    zay_list.append(zay.cpu())
                zx_list.append(zx.cpu())
                y_list.append(y_batch.cpu())
                
                # Convert hospital IDs to strings for better visualization
                hospital_labels = [f"Hospital {h.item() + 1}" for h in domain_batch]
                domain_dict["hospital"].append(hospital_labels)
                labels_dict["digit"].append(y_batch.cpu())
                labels_dict["hospital"].append(hospital_labels)
                
                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Memory management: limit the number of batches we process
            if batch_idx > 100:  # Process at most 100 batches
                break
    
    # Check if we collected any data
    if not zy_list:
        raise ValueError("No data was collected for visualization. Check your dataloader and sampling criteria.")
    
    print(f"\nCollected {total_collected} samples total")
    
    # Print sample counts
    print("\nNumber of samples per combination:")
    for (digit, hospital), count in counts.items():
        print(f"Digit {digit}, Hospital {hospital + 1}: {count} samples")
    
    # Calculate total samples
    total_samples = sum(counts.values())
    print(f"\nTotal samples: {total_samples}")
    if counts:
        print(f"Average samples per combination: {total_samples / len(counts):.1f}")
    else:
        print("No samples collected")
    
    return zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict

def sample_crmnist(model, dataloader, num_samples, device):
    """
    Sample data from CRMNIST model and collect domain information.
    Handles both NVAE and DIVA models (DIVA doesn't have zay latent space).
    
    Args:
        model: CRMNIST VAE model (NVAE or DIVA)
        dataloader: DataLoader containing (x, y, c, r) tuples
        num_samples: Maximum number of samples to collect
        device: torch device
    
    Returns:
        zy_list, za_list, zay_list, zx_list: Lists of latent vectors (zay_list is dummy for DIVA)
        y_list: List of digit labels
        domain_dict: Dictionary of domain labels
        labels_dict: Dictionary of domain label names
    """
    model.eval()
    zy_list, za_list, zay_list, zx_list = [], [], [], []
    y_list = []
    domain_dict = {"color": [], 'rotation': []}
    labels_dict = {
        "color": ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red'],
        'rotation': ['0°', '15°', '30°', '45°', '60°', '75°']
    }
    
    # Check if model is in DIVA mode
    is_diva = hasattr(model, 'diva') and model.diva
    
    # Initialize counters for each combination
    counts = {}
    for digit in range(10):
        for color in range(7):
            for rotation in range(6):
                counts[(digit, color, rotation)] = 0
    
    # Target samples per combination - ensure we get enough for visualization
    target_samples = 50  # Minimum samples per combination for good visualization
    total_collected = 0
    all_combinations_satisfied = False
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Sampling CRMNIST data ({'DIVA' if is_diva else 'NVAE'} mode)")
        for x, y, c, r in pbar:
            # Only check max_samples after we have enough samples for each combination
            if all_combinations_satisfied and total_collected >= num_samples:
                break
                
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
            
            # Check each sample
            for j in range(len(y)):
                digit = y[j].item()
                color = c[j].item()
                rotation = r[j].item()
                
                # Check if we need more samples for this specific combination
                combination = (digit, color, rotation)
                should_keep = counts[combination] < target_samples
                
                if should_keep:
                    keep_mask[j] = True
                    counts[combination] += 1
                    total_collected += 1
            
            # Apply mask to get only samples we want to keep
            if keep_mask.any():
                x_batch = x[keep_mask]
                y_batch = y[keep_mask]
                c_batch = c[keep_mask]
                r_batch = r[keep_mask]
                
                # Get latent representations
                z_loc, _ = model.qz(x_batch)
                zy = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                za = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
                
                # Handle zay - DIVA models don't have it
                if is_diva:
                    # For DIVA, create empty placeholder to maintain consistent return structure
                    zay = torch.zeros(zy.shape[0], 1, device=zy.device)
                else:
                    zay = z_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
                
                # Store latent vectors and labels
                zy_list.append(zy.cpu())
                za_list.append(za.cpu())
                zay_list.append(zay.cpu())
                zx_list.append(zx.cpu())
                y_list.append(y_batch.cpu())
                domain_dict["color"].append(c_batch.cpu())
                domain_dict["rotation"].append(r_batch.cpu())
            
            # Check if we have enough samples for all combinations
            all_combinations_satisfied = all(count >= target_samples for count in counts.values())
            
            # Update progress bar with more detailed information
            min_count = min(counts.values())
            max_count = max(counts.values())
            pbar.set_postfix({
                'total': total_collected,
                'min_count': min_count,
                'max_count': max_count,
                'target': target_samples,
                'satisfied': all_combinations_satisfied
            })
    
    # Print detailed statistics about the sampling
    print("\nSampling Statistics:")
    print(f"Total samples collected: {total_collected}")
    print(f"Target samples per combination: {target_samples}")
    print(f"Minimum samples per combination: {min(counts.values())}")
    print(f"Maximum samples per combination: {max(counts.values())}")
    
    # Print distribution of samples across digits
    digit_counts = {i: 0 for i in range(10)}
    for (digit, _, _), count in counts.items():
        digit_counts[digit] += count
    print("\nSamples per digit:")
    for digit, count in digit_counts.items():
        print(f"Digit {digit}: {count} samples")
    
    # Print distribution of samples across colors
    color_counts = {i: 0 for i in range(7)}
    for (_, color, _), count in counts.items():
        color_counts[color] += count
    print("\nSamples per color:")
    for color, count in color_counts.items():
        print(f"{labels_dict['color'][color]}: {count} samples")
    
    # Print distribution of samples across rotations
    rotation_counts = {i: 0 for i in range(6)}
    for (_, _, rotation), count in counts.items():
        rotation_counts[rotation] += count
    print("\nSamples per rotation:")
    for rotation, count in rotation_counts.items():
        print(f"{labels_dict['rotation'][rotation]}: {count} samples")
    
    return zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict

def visualize_latent_spaces(model, dataloader, device, type = "nvae", save_path=None, max_samples=5000):
    """
    Unified function to visualize latent spaces using t-SNE with balanced sampling.
    
    This function works for all model types (NVAE, DIVA, DANN, WILD, CRMNIST) and 
    automatically handles differences like DIVA not having a zay latent space.
    
    For CRMNIST/NVAE/DIVA:
    - Uses balanced sampling to collect equal samples per (digit × color × rotation) combination
    - Ensures good representation of all data variations
    - Automatically detects if model is in DIVA mode
    
    Args:
        model: Model with latent spaces (zy, zx, [zay], za) - zay optional for DIVA
        dataloader: DataLoader containing (x, y, *domains) tuples
        device: torch device
        type: Model type ("nvae", "diva", "dann", "wild", "crmnist")
        save_path: Optional path to save the visualization
        max_samples: Maximum number of samples to use for visualization
    """
    model.eval()
    
    # Get data and domain information from helper functions
    if type == "nvae":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_nvae(model, dataloader, max_samples, device)
    elif type == "diva":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_diva(model, dataloader, max_samples, device)
    elif type == "dann":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_dann(model, dataloader, max_samples, device, 'dann')
    elif type == "dann_augmented":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_dann(model, dataloader, max_samples, device, 'dann_augmented')
    elif type == "wild":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_wild(model, dataloader, max_samples, device)
    elif type == "crmnist":
        zy_list, za_list, zay_list, zx_list, y_list, domain_dict, labels_dict = sample_crmnist(model, dataloader, max_samples, device)
    else:  # Default to NVAE
        raise ValueError(f"Invalid model type: {type}")
    
    # Concatenate tensors
    zy = torch.cat(zy_list, dim=0)
    za = torch.cat(za_list, dim=0)
    zx = torch.cat(zx_list, dim=0)
    y_labels = torch.cat(y_list, dim=0)
    
    # Handle domain labels differently for WILD dataset
    if type == "wild":
        # For WILD, domain_dict contains lists of strings
        hospital_labels = []
        for batch_labels in domain_dict["hospital"]:
            hospital_labels.extend(batch_labels)
        domain_dict["hospital"] = hospital_labels
    else:
        # For other datasets, concatenate tensors as before
        for domain_name in domain_dict:
            domain_dict[domain_name] = torch.cat(domain_dict[domain_name], dim=0)
    
    # Print sample counts for all collected samples
    print("\nNumber of samples per category:")
    print("\nDigits:")
    digit_counts = torch.bincount(y_labels)
    for digit, count in enumerate(digit_counts):
        print(f"Digit {digit}: {count} samples")
    
    for domain_name in domain_dict.keys():
        print(f"\n{domain_name.capitalize()}:")
        if type == "wild":
            # Count occurrences of each hospital label
            from collections import Counter
            hospital_counts = Counter(domain_dict[domain_name])
            for hospital, count in sorted(hospital_counts.items()):
                print(f"{hospital}: {count} samples")
        else:
            domain_counts = torch.bincount(domain_dict[domain_name])
            for value, count in enumerate(domain_counts):
                print(f"{labels_dict[domain_name][value]}: {count} samples")
    
    # Convert to numpy
    zy = zy.numpy()
    za = za.numpy()
    zx = zx.numpy()
    y_labels = y_labels.numpy()
    
    # Convert domain labels to numeric values for WILD dataset
    if type == "wild":
        # Create a mapping from hospital labels to numeric values
        # Extract hospital numbers from labels (e.g., "Hospital 1" -> 1)
        hospital_numbers = [int(label.split()[-1]) for label in domain_dict["hospital"]]
        unique_hospitals = sorted(set(hospital_numbers))
        hospital_to_num = {num: i for i, num in enumerate(unique_hospitals)}
        domain_dict["hospital"] = np.array([hospital_to_num[num] for num in hospital_numbers])
    else:
        for domain_name in domain_dict:
            domain_dict[domain_name] = domain_dict[domain_name].numpy()
    
    # Run t-SNE
    print("\nRunning t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    
    # Apply t-SNE to each latent space
    latent_spaces = [(zy, 'Label-specific (zy)')]
    
    # Handle different model architectures
    if type == "dann":
        # For DANN: zy=class-specific, za=domain-specific, zay=interaction
        if zay_list[0] is not None:
            zay = torch.cat(zay_list, dim=0).numpy()
            latent_spaces.append((zay, 'Domain-Class Interaction (zdy)'))
        latent_spaces.extend([
            (za, 'Domain-specific (zd)'),
            (zx, 'Combined Features (zd)')  # zx is same as za for DANN
        ])
    elif not (hasattr(model, 'diva') and model.diva):
        # For NVAE (non-DIVA mode)
        zay = torch.cat(zay_list, dim=0).numpy()
        latent_spaces.append((zay, 'Domain-Label (zay)'))
        latent_spaces.extend([
            (za, 'Domain-specific (za)'),
            (zx, 'Residual (zx)')
        ])
    else:
        # For DIVA mode
        latent_spaces.extend([
            (za, 'Domain-specific (za)'),
            (zx, 'Residual (zx)')
        ])
    
    tsne_results = []
    for space, title in tqdm(latent_spaces, desc="Computing t-SNE", unit="space"):
        space_2d = tsne.fit_transform(space)
        tsne_results.append((space_2d, title))
    
    # Create figure
    n_cols = len(latent_spaces)
    n_rows = 1 + len(domain_dict)  # One row for digits, one for each domain
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Plot each latent space
    for col_idx, (space_2d, title) in enumerate(tsne_results):
        # First row: color by digit
        scatter = axes[0, col_idx].scatter(space_2d[:, 0], space_2d[:, 1], 
                                         c=y_labels, cmap='tab10', alpha=0.7)
        axes[0, col_idx].set_title(f'{title}\nColored by Digit')
        axes[0, col_idx].legend(*scatter.legend_elements(), title="Digits")
        
        # Additional rows: color by each domain
        for row_idx, (domain_name, domain_values) in enumerate(domain_dict.items(), 1):
            scatter = axes[row_idx, col_idx].scatter(space_2d[:, 0], space_2d[:, 1],
                                                   c=domain_values, cmap='tab10',
                                                   vmin=0, vmax=len(unique_hospitals)-1 if type == "wild" else len(labels_dict[domain_name])-1,
                                                   alpha=0.7)
            axes[row_idx, col_idx].set_title(f'{title}\nColored by {domain_name.capitalize()}')
            
            # Create custom legend for domain values
            if type == "wild":
                # For WILD, use the original hospital numbers
                # Get the actual colors used in the scatter plot
                norm = plt.Normalize(0, len(unique_hospitals)-1)
                colors = plt.cm.tab10(norm(range(len(unique_hospitals))))
                
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[i],
                              label=f"Hospital {h}", markersize=10)
                    for i, h in enumerate(unique_hospitals)
                ]
            else:
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.tab10(i/len(labels_dict[domain_name])),
                              label=labels_dict[domain_name][i], markersize=10)
                    for i in range(len(labels_dict[domain_name]))
                ]
            axes[row_idx, col_idx].legend(handles=legend_elements, title=domain_name.capitalize())
    
    plt.tight_layout()
    if save_path:
        print(f"Saving latent space visualization to {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"Latent space visualization saved to {save_path}")
    plt.close()

def get_model_name(args, model_type=None):
    """
    Generate a model name based on the arguments and model type.
    
    Args:
        args: Arguments object containing model parameters
        model_type: Type of model ('nvae', 'diva', 'dann', or 'irm')
    
    Returns:
        str: Model name string
    """
    if model_type is None:
        raise ValueError("Model type must be specified")
    
    # Create parameter string
    param_str = f"alpha1-{args.alpha_1}_alpha2-{args.alpha_2}_zy{args.zy_dim}_zx{args.zx_dim}_zay{args.zay_dim}_za{args.za_dim}_b1-{args.beta_1}_b2-{args.beta_2}_b3-{args.beta_3}_b4-{args.beta_4}_ep{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{param_str}_{timestamp}"

def process_batch(batch, device, dataset_type='crmnist'):
    """
    Process a batch of data for different dataset types.
    
    Args:
        batch: Tuple of (x, y, c, d) where:
            x: input images
            y: labels (could be one-hot encoded)
            c: color labels (could be one-hot encoded)
            d: domain labels (could be one-hot encoded)
        device: torch device to move tensors to
        dataset_type: Type of dataset ('crmnist', 'irm', 'dann', etc.)
        
    Returns:
        Tuple of (x, y, d) where:
            x: processed input images
            y: processed labels (converted to indices if one-hot)
            d: processed domain labels (converted to indices if one-hot)
    """

    

    # Dataset-specific processing
    if dataset_type == 'crmnist':
        # IRM specific processing if needed
        x, y, c, d = batch
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)
    elif dataset_type == 'wild':
        x, y, metadata = batch
        x = x.to(device)
        y = y.to(device)
        #metadata = metadata.to(device)
        d = metadata[:, 0].to(device)
    
    # Convert one-hot encoded labels to indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    if len(d.shape) > 1 and d.shape[1] > 1:
        d = torch.argmax(d, dim=1)
    return x, y, d 

def balanced_sample_for_visualization(model, dataloader, device, model_type="generic", 
                                     max_samples=5000, target_samples_per_combination=50,
                                     feature_extractor_fn=None, dataset_type="crmnist"):
    """
    Generic balanced sampling function for visualization that works with different model types.
    
    Args:
        model: The model to extract features from
        dataloader: DataLoader containing (x, y, c, r) tuples for CRMNIST or (x, y, metadata) for WILD
        device: torch device
        model_type: Type of model ("nvae", "dann", "irm", "diva", "generic")
        max_samples: Maximum total samples to collect
        target_samples_per_combination: Target samples per unique combination
        feature_extractor_fn: Custom function to extract features (optional)
        dataset_type: Type of dataset ("crmnist" or "wild")
    
    Returns:
        features_dict: Dictionary containing extracted features/latents
        labels_dict: Dictionary containing labels (y, c, r) - for WILD, c is dummy, r is hospital
        sampling_stats: Dictionary with sampling statistics
    """
    model.eval()
    
    # Initialize storage
    features_dict = {
        'zy': [], 'za': [], 'zay': [], 'zx': [], 'features': []
    }
    labels_dict = {
        'y': [], 'c': [], 'r': []
    }
    
    # Initialize counters for balanced sampling
    # For CRMNIST: combinations of (digit, color, rotation)
    # For WILD: combinations of (label, hospital) - 2 labels × 5 hospitals = 10 combinations
    # For simpler models: combinations of (digit, rotation)
    use_color = model_type in ["nvae", "diva"] and dataset_type == "crmnist"  # Models that use color information
    
    counts = {}
    if dataset_type == "wild":
        # WILD dataset: label × hospital combinations
        for label in range(2):  # Normal (0), Tumor (1)
            for hospital in range(5):  # 5 hospitals (0, 1, 2, 3, 4)
                counts[(label, hospital)] = 0
    elif use_color:
        # Full combinations: digit × color × rotation
        for digit in range(10):
            for color in range(7):
                for rotation in range(6):
                    counts[(digit, color, rotation)] = 0
    else:
        # Simplified combinations: digit × rotation
        for digit in range(10):
            for rotation in range(6):
                counts[(digit, rotation)] = 0
    
    total_collected = 0
    all_combinations_satisfied = False
    
    print(f"Starting balanced sampling for {model_type} model...")
    print(f"Target samples per combination: {target_samples_per_combination}")
    print(f"Total combinations: {len(counts)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stop if we have enough balanced samples
            if all_combinations_satisfied and total_collected >= max_samples:
                break
            
            # Process batch according to dataset type
            if dataset_type == "wild":
                x, y, metadata = batch
                x = x.to(device)
                y = y.to(device)
                r = metadata[:, 0].to(device)  # Hospital ID from metadata
                c = torch.zeros_like(y)  # Dummy color for WILD (not used)
            else:
                # CRMNIST format
                x, y, c, r = batch
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                r = r.to(device)
            
            # Convert to indices if one-hot encoded
            if len(y.shape) > 1:
                y_indices = torch.argmax(y, dim=1)
            else:
                y_indices = y.long()
                
            if len(c.shape) > 1:
                c_indices = torch.argmax(c, dim=1)
            else:
                c_indices = c.long()
                
            if len(r.shape) > 1:
                r_indices = torch.argmax(r, dim=1)
            else:
                r_indices = r.long()
            
            # Create mask for samples we want to keep
            keep_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
            
            # Check each sample for balanced sampling
            for j in range(len(y)):
                if dataset_type == "wild":
                    label = y_indices[j].item()
                    hospital = r_indices[j].item()
                    combination = (label, hospital)
                else:
                    digit = y_indices[j].item()
                    rotation = r_indices[j].item()
                    
                    if use_color:
                        color = c_indices[j].item()
                        combination = (digit, color, rotation)
                    else:
                        combination = (digit, rotation)
                
                if counts[combination] < target_samples_per_combination:
                    keep_mask[j] = True
                    counts[combination] += 1
                    total_collected += 1
            
            # Apply mask and extract features only for selected samples
            if keep_mask.any():
                x_selected = x[keep_mask]
                y_selected = y[keep_mask]
                c_selected = c[keep_mask]
                r_selected = r[keep_mask]
                
                # Extract features based on model type
                if feature_extractor_fn:
                    # Use custom feature extractor
                    extracted_features = feature_extractor_fn(model, x_selected, y_selected, c_selected, r_selected)
                else:
                    # Use default feature extraction based on model type
                    extracted_features = _extract_features_by_model_type(
                        model, model_type, x_selected, y_selected, c_selected, r_selected
                    )
                
                # Store features
                for key, value in extracted_features.items():
                    if value is not None:
                        features_dict[key].append(value.cpu())
                
                # Store labels
                labels_dict['y'].append(y_selected.cpu())
                labels_dict['c'].append(c_selected.cpu())
                labels_dict['r'].append(r_selected.cpu())
            
            # Check if all combinations are satisfied
            all_combinations_satisfied = all(count >= target_samples_per_combination for count in counts.values())
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                min_count = min(counts.values())
                print(f"Batch {batch_idx}: collected {total_collected} samples, min_count: {min_count}")
    
    # Concatenate all collected data
    final_features = {}
    for key, value_list in features_dict.items():
        if value_list:  # Only concatenate if we have data
            final_features[key] = torch.cat(value_list, dim=0)
        else:
            final_features[key] = None
    
    final_labels = {}
    for key, value_list in labels_dict.items():
        if value_list:
            final_labels[key] = torch.cat(value_list, dim=0)
    
    # Sampling statistics
    sampling_stats = {
        'total_collected': total_collected,
        'target_per_combination': target_samples_per_combination,
        'min_per_combination': min(counts.values()),
        'max_per_combination': max(counts.values()),
        'num_combinations': len(counts),
        'all_satisfied': all_combinations_satisfied
    }
    
    print(f"\nSampling completed:")
    print(f"Total samples collected: {total_collected}")
    print(f"Min samples per combination: {sampling_stats['min_per_combination']}")
    print(f"Max samples per combination: {sampling_stats['max_per_combination']}")
    
    return final_features, final_labels, sampling_stats

def _extract_features_by_model_type(model, model_type, x, y, c, r):
    """
    Extract features based on model type.
    
    Returns:
        Dictionary with extracted features for different latent spaces
    """
    features = {
        'zy': None, 'za': None, 'zay': None, 'zx': None, 'features': None
    }
    
    if model_type == "nvae":
        # NVAE has separate latent spaces
        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model.forward(y, x, r)
        features['zy'] = zy
        features['za'] = za
        features['zay'] = zay
        features['zx'] = zx
        features['features'] = z  # Full latent vector
        
    elif model_type == "diva":
        # DIVA is similar to NVAE but without zay
        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model.forward(y, x, r)
        features['zy'] = zy
        features['za'] = za
        features['zay'] = None  # DIVA doesn't have zay
        features['zx'] = zx
        features['features'] = z
        
    elif model_type == "dann":
        # DANN only has one feature space
        if hasattr(model, 'get_features'):
            single_features = model.get_features(x)
        else:
            # Fallback: use forward pass and extract features
            y_logits, domain_predictions = model.forward(x, y, r)
            single_features = model.feature_extractor(x)
        
        # Use the same features for all latent spaces in visualization
        features['zy'] = single_features
        features['za'] = single_features
        features['zay'] = single_features
        features['zx'] = single_features
        features['features'] = single_features
        
    elif model_type == "irm":
        # IRM is similar to DANN
        if hasattr(model, 'get_features'):
            single_features = model.get_features(x)
        else:
            logits, single_features = model.forward(x, y, r)
        
        features['zy'] = single_features
        features['za'] = single_features
        features['zay'] = single_features
        features['zx'] = single_features
        features['features'] = single_features
        
    else:
        # Generic case: try to extract features
        if hasattr(model, 'get_features'):
            single_features = model.get_features(x)
            features['features'] = single_features
        elif hasattr(model, 'feature_extractor'):
            single_features = model.feature_extractor(x)
            features['features'] = single_features
        else:
            raise ValueError(f"Don't know how to extract features from model type: {model_type}")
    
    return features 

def crmnist_color_aware_sampling(model, dataset, device, model_type="nvae", 
                                max_samples=5000, target_samples_per_color=100,
                                target_samples_per_combination=50):
    """
    Intelligent color-aware sampling for CRMNIST dataset that knows where to look for each color.
    Uses the same logic as the color visualization script to efficiently find colored samples.
    
    Args:
        model: The model to extract features from
        dataset: CRMNIST dataset (not DataLoader) to allow indexed access
        device: torch device
        model_type: Type of model ("nvae", "dann", "irm", "diva")
        max_samples: Maximum total samples to collect
        target_samples_per_color: Target samples per color type
        target_samples_per_combination: Target samples per (digit, color, rotation) combination
    
    Returns:
        features_dict: Dictionary containing extracted features/latents
        labels_dict: Dictionary containing labels (y, c, r)
        sampling_stats: Dictionary with sampling statistics
    """
    model.eval()
    
    print(f"Starting intelligent color-aware sampling for CRMNIST...")
    print(f"Dataset size: {len(dataset)}")
    print(f"Target samples per color: {target_samples_per_color}")
    
    # Initialize storage
    features_dict = {
        'zy': [], 'za': [], 'zay': [], 'zx': [], 'features': []
    }
    labels_dict = {
        'y': [], 'c': [], 'r': []
    }
    
    # Color mapping
    color_idx_to_name = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'cyan', 4: 'magenta', 5: 'orange', 6: 'red'}
    color_samples = {color: [] for color in color_idx_to_name.values()}
    
    # Track combination counts for balanced sampling
    combination_counts = {}
    for digit in range(10):
        for color in range(7):
            for rotation in range(6):
                combination_counts[(digit, color, rotation)] = 0
    
    total_images = len(dataset)
    domain_size = 10000  # Each domain has 10000 images
    num_domains = 6
    
    print("Sampling by domain to ensure color diversity...")
    
    with torch.no_grad():
        # Sample from each domain intelligently
        for domain in range(num_domains):
            start_idx = domain * domain_size
            end_idx = min((domain + 1) * domain_size, total_images)
            
            print(f"\nProcessing Domain {domain} (indices {start_idx}-{end_idx-1})...")
            
            domain_samples_collected = 0
            domain_color_counts = {color: 0 for color in color_idx_to_name.values()}
            
            # Sample red images from start of domain (first ~1200 images)
            red_samples_needed = target_samples_per_color - len(color_samples['red'])
            if red_samples_needed > 0:
                for i in range(min(1200, end_idx - start_idx)):
                    idx = start_idx + i
                    img, y_label, c_label, r_label = dataset[idx]
                    
                    if torch.max(c_label) > 0:
                        color_idx = torch.argmax(c_label).item()
                        color_name = color_idx_to_name.get(color_idx, f"unknown_{color_idx}")
                        
                        if color_name == 'red' and len(color_samples['red']) < target_samples_per_color:
                            # Check combination balance
                            digit = y_label.item() if len(y_label.shape) == 0 else torch.argmax(y_label).item()
                            rotation = torch.argmax(r_label).item() if torch.max(r_label) > 0 else -1
                            combination = (digit, color_idx, rotation)
                            
                            if combination_counts[combination] < target_samples_per_combination:
                                # Extract features
                                x_tensor = img.unsqueeze(0).to(device)
                                y_tensor = y_label.unsqueeze(0).to(device) if len(y_label.shape) == 0 else y_label.unsqueeze(0).to(device)
                                c_tensor = c_label.unsqueeze(0).to(device)
                                r_tensor = r_label.unsqueeze(0).to(device)
                                
                                extracted_features = _extract_features_by_model_type(
                                    model, model_type, x_tensor, y_tensor, c_tensor, r_tensor
                                )
                                
                                # Store features
                                for key, value in extracted_features.items():
                                    if value is not None:
                                        features_dict[key].append(value.cpu())
                                
                                # Store labels
                                labels_dict['y'].append(y_tensor.cpu())
                                labels_dict['c'].append(c_tensor.cpu())
                                labels_dict['r'].append(r_tensor.cpu())
                                
                                color_samples['red'].append(idx)
                                domain_color_counts['red'] += 1
                                domain_samples_collected += 1
                                combination_counts[combination] += 1
                                
                                if len(color_samples['red']) >= target_samples_per_color:
                                    break
            
            # Sample domain-specific colors from middle section (1200+)
            expected_domain_color = ['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange'][domain]
            domain_color_needed = target_samples_per_color - len(color_samples[expected_domain_color])
            
            if domain_color_needed > 0:
                search_start = start_idx + 1200
                for idx in range(search_start, min(search_start + 5000, end_idx)):
                    img, y_label, c_label, r_label = dataset[idx]
                    
                    if torch.max(c_label) > 0:
                        color_idx = torch.argmax(c_label).item()
                        color_name = color_idx_to_name.get(color_idx, f"unknown_{color_idx}")
                        
                        if color_name == expected_domain_color and len(color_samples[expected_domain_color]) < target_samples_per_color:
                            # Check combination balance
                            digit = y_label.item() if len(y_label.shape) == 0 else torch.argmax(y_label).item()
                            rotation = torch.argmax(r_label).item() if torch.max(r_label) > 0 else -1
                            combination = (digit, color_idx, rotation)
                            
                            if combination_counts[combination] < target_samples_per_combination:
                                # Extract features
                                x_tensor = img.unsqueeze(0).to(device)
                                y_tensor = y_label.unsqueeze(0).to(device) if len(y_label.shape) == 0 else y_label.unsqueeze(0).to(device)
                                c_tensor = c_label.unsqueeze(0).to(device)
                                r_tensor = r_label.unsqueeze(0).to(device)
                                
                                extracted_features = _extract_features_by_model_type(
                                    model, model_type, x_tensor, y_tensor, c_tensor, r_tensor
                                )
                                
                                # Store features
                                for key, value in extracted_features.items():
                                    if value is not None:
                                        features_dict[key].append(value.cpu())
                                
                                # Store labels
                                labels_dict['y'].append(y_tensor.cpu())
                                labels_dict['c'].append(c_tensor.cpu())
                                labels_dict['r'].append(r_tensor.cpu())
                                
                                color_samples[expected_domain_color].append(idx)
                                domain_color_counts[expected_domain_color] += 1
                                domain_samples_collected += 1
                                combination_counts[combination] += 1
                                
                                if len(color_samples[expected_domain_color]) >= target_samples_per_color:
                                    break
            
            print(f"  Domain {domain} collected: {domain_samples_collected} samples")
            print(f"  Colors found: {domain_color_counts}")
            
            # Check if we have enough samples overall
            total_collected = sum(len(samples) for samples in color_samples.values())
            if total_collected >= max_samples:
                break
    
    # Concatenate all collected features and labels
    for key in features_dict:
        if features_dict[key]:
            features_dict[key] = torch.cat(features_dict[key], dim=0)
        else:
            features_dict[key] = None
    
    for key in labels_dict:
        if labels_dict[key]:
            labels_dict[key] = torch.cat(labels_dict[key], dim=0)
    
    # Create sampling statistics
    final_color_counts = {color: len(samples) for color, samples in color_samples.items()}
    total_collected = sum(final_color_counts.values())
    
    sampling_stats = {
        'total_samples': total_collected,
        'color_distribution': final_color_counts,
        'combination_counts': combination_counts,
        'target_samples_per_color': target_samples_per_color,
        'domains_processed': min(num_domains, 6)
    }
    
    print(f"\n=== INTELLIGENT SAMPLING COMPLETE ===")
    print(f"Total samples collected: {total_collected}")
    print(f"Color distribution: {final_color_counts}")
    
    return features_dict, labels_dict, sampling_stats 
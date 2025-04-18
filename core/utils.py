import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn


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

def kl_divergence(loc, scale): # assumes scale is logscale
    batch_size = loc.size(0)
    if loc.dim() == 4:
        loc = loc.view(batch_size, -1)
    if scale.dim() == 4:
        scale = scale.view(batch_size, -1)
    
    # KL divergence formula for normal distribution
    kls = -0.5 * (1 + 2 * scale - loc.pow(2) - scale.exp().pow(2))
    total_kl = kls.sum(1).mean(0)
    dim_wise_kls = kls.mean(0)
    mean_kl = kls.mean(1).mean(0)

    return total_kl, dim_wise_kls, mean_kl
                               
# Function to calculate additional metrics
def calculate_metrics(model, y, x, r):
    """Calculate additional metrics beyond just loss"""
    with torch.no_grad():
        x_recon, _, _, _, _, _, _, y_hat, a_hat, _, _, _, _ = model.forward(y, x, r)
        
        # Reconstruction MSE
        recon_mse = F.mse_loss(x_recon, x).item()
        
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
    

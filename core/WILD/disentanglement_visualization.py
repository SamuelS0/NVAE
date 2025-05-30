import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random


def visualize_disentanglement(model, dataloader, device, save_path=None, num_variations=7, num_examples=3):
    """
    Visualize disentanglement by showing how changing each latent space affects generation
    while keeping others fixed. This demonstrates the quality of learned disentangled representations.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader containing (x, y, c, r) tuples
        device: torch device
        save_path: Optional path to save visualization
        num_variations: Number of variations to show per latent space (default: 7)
        num_examples: Number of base examples to use (default: 3)
    """
    model.eval()
    
    # Get diverse base examples
    base_examples = _select_diverse_examples(dataloader, device, num_examples)
    
    # Create visualizations for each base example
    for example_idx, (x_base, y_base, c_base, r_base) in enumerate(base_examples):
        print(f"Creating disentanglement visualization for example {example_idx + 1}/{num_examples}")
        
        # Get base latent representation
        with torch.no_grad():
            z_loc, z_scale = model.qz(x_base.unsqueeze(0))
            
            # Extract individual latent components
            zy_base = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
            zx_base = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
            za_base = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
            
            if model.diva:
                zay_base = None
                num_latent_spaces = 3
            else:
                zay_base = z_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
                num_latent_spaces = 4
        
        # Create figure for this example
        fig, axes = plt.subplots(num_latent_spaces + 1, num_variations, 
                                figsize=(2 * num_variations, 2 * (num_latent_spaces + 1)))
        
        # Show original image in the first row, center column
        center_col = num_variations // 2
        original_img = x_base.cpu().permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)
        
        # Clear the first row and show original in center
        for col in range(num_variations):
            axes[0, col].axis('off')
        axes[0, center_col].imshow(original_img)
        axes[0, center_col].set_title('Original Image', fontsize=10, fontweight='bold')
        axes[0, center_col].axis('off')
        
        # Add example information
        digit_label = y_base.item() if len(y_base.shape) == 0 else torch.argmax(y_base).item()
        color_label = torch.argmax(c_base).item() if torch.max(c_base) > 0 else -1
        rotation_label = torch.argmax(r_base).item() if torch.max(r_base) > 0 else -1
        
        color_names = ['Blue', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Orange', 'Red']
        color_name = color_names[color_label] if 0 <= color_label < len(color_names) else 'None'
        rotation_angle = rotation_label * 15 if rotation_label >= 0 else 0
        
        fig.suptitle(f'Disentanglement Analysis - Example {example_idx + 1}\n'
                    f'Digit: {digit_label}, Color: {color_name}, Rotation: {rotation_angle}°', 
                    fontsize=12, fontweight='bold')
        
        # Generate variations for each latent space
        latent_spaces = [
            ('zy', zy_base, 'Label-specific (zy)\nShould change: Digit identity'),
            ('za', za_base, 'Domain-specific (za)\nShould change: Rotation/Domain'),
        ]
        
        if not model.diva:
            latent_spaces.append(('zay', zay_base, 'Domain-Label (zay)\nShould change: Interaction effects'))
        
        latent_spaces.append(('zx', zx_base, 'Residual (zx)\nShould change: Style/Noise'))
        
        # Create variations for each latent space
        for row_idx, (space_name, base_latent, description) in enumerate(latent_spaces):
            row = row_idx + 1  # Skip the original image row
            
            # Generate variations by interpolating around the base latent
            variations = _generate_latent_variations(base_latent, num_variations)
            
            for col, varied_latent in enumerate(variations):
                # Reconstruct image with varied latent
                if space_name == 'zy':
                    img_recon = model.px(varied_latent, zx_base, zay_base, za_base)
                elif space_name == 'za':
                    img_recon = model.px(zy_base, zx_base, zay_base, varied_latent)
                elif space_name == 'zay':
                    img_recon = model.px(zy_base, zx_base, varied_latent, za_base)
                elif space_name == 'zx':
                    img_recon = model.px(zy_base, varied_latent, zay_base, za_base)
                
                # Display the reconstructed image
                img = img_recon[0].cpu().detach().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # Add variation strength as title for center column
                if col == center_col:
                    axes[row, col].set_title('Original', fontsize=8)
                else:
                    variation_strength = (col - center_col) * 0.5
                    axes[row, col].set_title(f'{variation_strength:+.1f}σ', fontsize=8)
            
            # Add row label
            axes[row, 0].set_ylabel(description, fontsize=9, ha='right', va='center')
        
        plt.tight_layout()
        
        # Save individual example
        if save_path:
            base_path = save_path.replace('.png', f'_example_{example_idx + 1}.png')
            plt.savefig(base_path, bbox_inches='tight', dpi=150)
            print(f"Disentanglement visualization saved to {base_path}")
        
        plt.close()
    
    # Create a summary visualization showing all examples side by side
    if len(base_examples) > 1:
        _create_summary_visualization(model, base_examples, device, save_path, num_variations)


def _select_diverse_examples(dataloader, device, num_examples=3):
    """
    Select diverse examples from the dataloader for disentanglement analysis.
    Tries to get examples with different digits, colors, and rotations.
    """
    examples = []
    seen_combinations = set()
    
    # Try to find diverse examples
    for batch_idx, (x, y, metadata) in enumerate(dataloader):
        hospital_id = metadata[:, 0]
        x, y, hospital_id = x.to(device), y.to(device), hospital_id.to(device)
        
        for i in range(len(x)):
            # Get labels
            digit = y[i].item() if len(y[i].shape) == 0 else torch.argmax(y[i]).item()
            color = torch.argmax(c[i]).item() if torch.max(c[i]) > 0 else -1
            rotation = torch.argmax(r[i]).item() if torch.max(r[i]) > 0 else -1
            
            combination = (digit, color, rotation)
            
            # Add if we haven't seen this combination and need more examples
            if combination not in seen_combinations and len(examples) < num_examples:
                examples.append((x[i], y[i], c[i], r[i]))
                seen_combinations.add(combination)
                
                if len(examples) >= num_examples:
                    break
        
        if len(examples) >= num_examples:
            break
    
    # If we couldn't find enough diverse examples, just take the first few
    if len(examples) < num_examples:
        print(f"Warning: Could only find {len(examples)} diverse examples, using first batch")
        x, y, c, r = next(iter(dataloader))
        x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
        examples = [(x[i], y[i], c[i], r[i]) for i in range(min(num_examples, len(x)))]
    
    return examples


def _generate_latent_variations(base_latent, num_variations):
    """
    Generate variations of a latent vector by adding scaled noise.
    The center variation is the original, others are scaled perturbations.
    """
    variations = []
    center_idx = num_variations // 2
    
    for i in range(num_variations):
        if i == center_idx:
            # Center is the original
            variations.append(base_latent.clone())
        else:
            # Create variation by adding scaled noise
            scale = (i - center_idx) * 0.5  # Scale from -1.5 to +1.5
            noise = torch.randn_like(base_latent) * 0.3  # Small noise
            varied = base_latent + scale * noise
            variations.append(varied)
    
    return variations


def _create_summary_visualization(model, base_examples, device, save_path, num_variations):
    """
    Create a summary visualization showing all examples in a compact grid.
    """
    num_examples = len(base_examples)
    num_latent_spaces = 3 if model.diva else 4
    
    # Create a large figure with all examples
    fig, axes = plt.subplots(num_examples * (num_latent_spaces + 1), num_variations, 
                            figsize=(2 * num_variations, 1.5 * num_examples * (num_latent_spaces + 1)))
    
    center_col = num_variations // 2
    
    for example_idx, (x_base, y_base, c_base, r_base) in enumerate(base_examples):
        base_row = example_idx * (num_latent_spaces + 1)
        
        with torch.no_grad():
            z_loc, _ = model.qz(x_base.unsqueeze(0))
            zy_base = z_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
            zx_base = z_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
            za_base = z_loc[:, model.za_index_range[0]:model.za_index_range[1]]
            zay_base = None if model.diva else z_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
        
        # Show original
        for col in range(num_variations):
            axes[base_row, col].axis('off')
        
        original_img = x_base.cpu().permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)
        axes[base_row, center_col].imshow(original_img)
        axes[base_row, center_col].set_title(f'Ex {example_idx + 1}', fontsize=8)
        axes[base_row, center_col].axis('off')
        
        # Generate variations for each latent space
        latent_spaces = [('zy', zy_base), ('za', za_base)]
        if not model.diva:
            latent_spaces.append(('zay', zay_base))
        latent_spaces.append(('zx', zx_base))
        
        for space_idx, (space_name, base_latent) in enumerate(latent_spaces):
            row = base_row + space_idx + 1
            variations = _generate_latent_variations(base_latent, num_variations)
            
            for col, varied_latent in enumerate(variations):
                if space_name == 'zy':
                    img_recon = model.px(varied_latent, zx_base, zay_base, za_base)
                elif space_name == 'za':
                    img_recon = model.px(zy_base, zx_base, zay_base, varied_latent)
                elif space_name == 'zay':
                    img_recon = model.px(zy_base, zx_base, varied_latent, za_base)
                elif space_name == 'zx':
                    img_recon = model.px(zy_base, varied_latent, zay_base, za_base)
                
                img = img_recon[0].cpu().detach().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
            
            # Add space label
            if example_idx == 0:  # Only label for first example
                axes[row, 0].set_ylabel(space_name, fontsize=8)
    
    plt.suptitle('Disentanglement Summary - All Examples', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        summary_path = save_path.replace('.png', '_summary.png')
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        print(f"Summary disentanglement visualization saved to {summary_path}")
    
    plt.close()


def visualize_latent_interpolation(model, dataloader, device, save_path=None, num_steps=7):
    """
    Visualize smooth interpolation between two different images in latent space.
    This shows how the latent space is structured and whether transitions are smooth.
    """
    model.eval()
    
    # Get two diverse examples
    examples = _select_diverse_examples(dataloader, device, num_examples=2)
    if len(examples) < 2:
        print("Warning: Need at least 2 examples for interpolation")
        return
    
    (x1, y1, c1, r1), (x2, y2, c2, r2) = examples[:2]
    
    with torch.no_grad():
        # Get latent representations
        z1_loc, _ = model.qz(x1.unsqueeze(0))
        z2_loc, _ = model.qz(x2.unsqueeze(0))
        
        # Extract components
        zy1 = z1_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
        zx1 = z1_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
        za1 = z1_loc[:, model.za_index_range[0]:model.za_index_range[1]]
        
        zy2 = z2_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
        zx2 = z2_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
        za2 = z2_loc[:, model.za_index_range[0]:model.za_index_range[1]]
        
        if model.diva:
            zay1 = zay2 = None
            num_spaces = 3
        else:
            zay1 = z1_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
            zay2 = z2_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
            num_spaces = 4
        
        # Create interpolation steps
        alphas = torch.linspace(0, 1, num_steps).to(device)
        
        # Create figure
        fig, axes = plt.subplots(num_spaces + 1, num_steps, figsize=(2 * num_steps, 2 * (num_spaces + 1)))
        
        # Show original images in first row
        for step in range(num_steps):
            axes[0, step].axis('off')
        
        # Show source and target
        img1 = x1.cpu().permute(1, 2, 0).numpy()
        img2 = x2.cpu().permute(1, 2, 0).numpy()
        axes[0, 0].imshow(np.clip(img1, 0, 1))
        axes[0, 0].set_title('Source', fontsize=10)
        axes[0, -1].imshow(np.clip(img2, 0, 1))
        axes[0, -1].set_title('Target', fontsize=10)
        
        # Interpolate each latent space separately
        space_names = ['zy', 'za']
        space_pairs = [(zy1, zy2), (za1, za2)]
        
        if not model.diva:
            space_names.append('zay')
            space_pairs.append((zay1, zay2))
        
        space_names.append('zx')
        space_pairs.append((zx1, zx2))
        
        for space_idx, (space_name, (z_start, z_end)) in enumerate(zip(space_names, space_pairs)):
            row = space_idx + 1
            
            for step, alpha in enumerate(alphas):
                # Interpolate this space, keep others from source
                if space_name == 'zy':
                    zy_interp = (1 - alpha) * zy1 + alpha * zy2
                    img_recon = model.px(zy_interp, zx1, zay1, za1)
                elif space_name == 'za':
                    za_interp = (1 - alpha) * za1 + alpha * za2
                    img_recon = model.px(zy1, zx1, zay1, za_interp)
                elif space_name == 'zay':
                    zay_interp = (1 - alpha) * zay1 + alpha * zay2
                    img_recon = model.px(zy1, zx1, zay_interp, za1)
                elif space_name == 'zx':
                    zx_interp = (1 - alpha) * zx1 + alpha * zx2
                    img_recon = model.px(zy1, zx_interp, zay1, za1)
                
                img = img_recon[0].cpu().detach().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[row, step].imshow(img)
                axes[row, step].axis('off')
                
                if step == 0:
                    axes[row, step].set_ylabel(f'{space_name}\ninterpolation', fontsize=9)
        
        plt.suptitle('Latent Space Interpolation', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            interp_path = save_path.replace('.png', '_interpolation.png')
            plt.savefig(interp_path, bbox_inches='tight', dpi=150)
            print(f"Interpolation visualization saved to {interp_path}")
        
        plt.close()


def visualize_factor_traversal(model, device, save_path=None, num_steps=7):
    """
    Visualize factor traversal by systematically varying each dimension of each latent space.
    This helps identify which dimensions control which factors.
    """
    model.eval()
    
    # Sample a base point from the prior
    with torch.no_grad():
        # Sample from standard normal for all latent spaces
        zy_base = torch.randn(1, model.zy_dim).to(device)
        zx_base = torch.randn(1, model.zx_dim).to(device)
        za_base = torch.randn(1, model.za_dim).to(device)
        
        if model.diva:
            zay_base = None
            latent_spaces = [('zy', zy_base), ('za', za_base), ('zx', zx_base)]
        else:
            zay_base = torch.randn(1, model.zay_dim).to(device)
            latent_spaces = [('zy', zy_base), ('za', za_base), ('zay', zay_base), ('zx', zx_base)]
        
        # For each latent space, show traversal of first few dimensions
        max_dims_per_space = 3  # Show first 3 dimensions of each space
        
        for space_name, base_latent in latent_spaces:
            num_dims = min(max_dims_per_space, base_latent.shape[1])
            
            fig, axes = plt.subplots(num_dims, num_steps, figsize=(2 * num_steps, 2 * num_dims))
            if num_dims == 1:
                axes = axes.reshape(1, -1)
            
            for dim in range(num_dims):
                # Create traversal values
                traversal_values = torch.linspace(-3, 3, num_steps).to(device)
                
                for step, value in enumerate(traversal_values):
                    # Modify only this dimension
                    modified_latent = base_latent.clone()
                    modified_latent[0, dim] = value
                    
                    # Generate image
                    if space_name == 'zy':
                        img_recon = model.px(modified_latent, zx_base, zay_base, za_base)
                    elif space_name == 'za':
                        img_recon = model.px(zy_base, zx_base, zay_base, modified_latent)
                    elif space_name == 'zay':
                        img_recon = model.px(zy_base, zx_base, modified_latent, za_base)
                    elif space_name == 'zx':
                        img_recon = model.px(zy_base, modified_latent, zay_base, za_base)
                    
                    img = img_recon[0].cpu().detach().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[dim, step].imshow(img)
                    axes[dim, step].axis('off')
                    
                    if step == 0:
                        axes[dim, step].set_ylabel(f'Dim {dim}', fontsize=9)
                    if dim == 0:
                        axes[dim, step].set_title(f'{value:.1f}', fontsize=8)
            
            plt.suptitle(f'Factor Traversal - {space_name.upper()} Space', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                traversal_path = save_path.replace('.png', f'_traversal_{space_name}.png')
                plt.savefig(traversal_path, bbox_inches='tight', dpi=150)
                print(f"Factor traversal for {space_name} saved to {traversal_path}")
            
            plt.close() 
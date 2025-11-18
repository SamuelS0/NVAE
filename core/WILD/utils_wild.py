import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from PIL import Image
#from model_diva import DIVA_VAE

def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images from WILDS dataset."""
    transform = transforms.ToTensor()

    # batch is a list of tuples (x, y, metadata)
    xs, ys, metadatas = [], [], []
    for item in batch:
        x, y, metadata = item
        # Convert PIL Image to tensor if needed
        if isinstance(x, Image.Image):
            x = transform(x)
        xs.append(x)
        ys.append(y)
        metadatas.append(metadata)

    # Stack into batches
    xs = torch.stack(xs)
    ys = torch.tensor(ys) if not isinstance(ys[0], torch.Tensor) else torch.stack(ys)
    metadatas = torch.stack(metadatas) if isinstance(metadatas[0], torch.Tensor) else torch.tensor(metadatas)

    return xs, ys, metadatas

def select_diverse_sample_batch(loader, data_type='id_val', samples_per_domain=10, seed=None):
    # Create local random number generator for reproducibility
    rng = random.Random(seed) if seed is not None else random

    # Initialize dictionaries to store samples for each domain AND label
    if data_type == 'train':
        domain_list = [0,3,4]
    elif data_type == 'id_val':
        domain_list = [0,3,4]
    elif data_type == 'val':
        domain_list = [1]
    elif data_type == 'test':
        domain_list = [2]
    domain_samples = {i: {'0': [], '1': []} for i in domain_list}  # Split by hospital and label
    
    for batch_idx, (x, y, metadata) in enumerate(loader):
        hospital_ids = metadata[:, 0]

        for i in range(len(x)):
            hospital_id = int(hospital_ids[i].item())

            # Validate hospital ID is in valid range for Camelyon17 (0-4)
            assert 0 <= hospital_id < 5, f"Invalid hospital ID {hospital_id} (expected 0-4 for Camelyon17)"

            label = str(int(y[i].item()))  # Convert label to string for dict key

            if hospital_id not in domain_samples:
                domain_samples[hospital_id] = {'0': [], '1': []}
            
            # Store sample in appropriate hospital and label bucket
            domain_samples[hospital_id][label].append((x[i], y[i], metadata[i]))
        
        # Check if we have enough samples of each label for each domain
        if all(len(samples['0']) >= samples_per_domain//2 and 
                len(samples['1']) >= samples_per_domain//2 
                for samples in domain_samples.values()):
            break
        

    
    # Select balanced samples for each domain
    selected_x, selected_y, selected_metadata = [], [], []
    for hospital_id, label_samples in domain_samples.items():
        for label in ['0', '1']:
            samples = label_samples[label]
            n_samples = min(samples_per_domain//2, len(samples))
            selected = rng.sample(samples, n_samples)
            
            for x_i, y_i, metadata_i in selected:
                selected_x.append(x_i)
                selected_y.append(y_i)
                selected_metadata.append(metadata_i)
    
    # Convert lists to tensors
    selected_x = torch.stack(selected_x)
    selected_y = torch.stack(selected_y)
    selected_metadata = torch.stack(selected_metadata)
    
    # Print distribution of labels for debugging
    # Use minlength=2 to ensure both class counts are available (prevents IndexError)
    label_dist = torch.bincount(selected_y.long(), minlength=2)
    print(f"Label distribution in batch - Normal: {label_dist[0]}, Tumor: {label_dist[1]}")
    
    return selected_x, selected_y, selected_metadata


def visualize_reconstructions(model, epoch, batch_data, image_dir, args):
    """
    Visualize original images and their reconstructions, organized by domain.
    Each domain shows 10 samples with their reconstructions.
    
    Args:
        epoch: Current epoch number
        batch_data: Tuple of (x, y, c, r) tensors
    """
    
    x, y, metadata = batch_data
    hospital_id = metadata[:, 0]
    if args.cuda:
        x, y, hospital_id = x.to(args.device), y.to(args.device), hospital_id.to(args.device)
    
    model.eval()
    with torch.no_grad():
        #if args.model == 'vae':
        x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model.forward(y, x, hospital_id)
        ''' elif args.model == 'diva':
            x_recon, d_hat, y_hat, qz, pz, z_q = model.forward(hospital_id, x, y)'''
    
    # Get labels in the right format
    if len(y.shape) > 1 and y.shape[1] > 1:
        # One-hot format
        y_labels = y.max(1)[1].cpu().numpy()
    else:
        # Integer format
        y_labels = y.cpu().numpy()
    
    # Get hospital IDs
    hospital_indices = hospital_id.cpu().numpy()
    
    # Create a figure with subplots for each domain
    num_domains = 5  # 5 hospitals
    samples_per_domain = 10  # 10 samples per domain
    
    # Create a figure with 5 domains, each with 2 rows (original & reconstruction) and 10 columns (samples)
    fig, axes = plt.subplots(num_domains * 2, samples_per_domain, figsize=(20, 4 * num_domains))
    
    # Get hospital names for titles (0-indexed to match Camelyon17 hospital IDs)
    hospital_names = ['Hospital 0', 'Hospital 1', 'Hospital 2', 'Hospital 3', 'Hospital 4']
    
    # Organize images by domain
    domain_images = {i: [] for i in range(num_domains)}
    
    # Group images by their hospital domain
    for i in range(len(x)):
        domain = hospital_indices[i]
        if domain < num_domains:  # Ensure valid domain
            domain_images[domain].append(i)
    
    # Process each domain
    for domain in range(num_domains):
        domain_indices = domain_images[domain]
        
        # If we have enough images for this domain, select the exact number needed
        num_samples = min(samples_per_domain, len(domain_indices))
        
        # Print domain stats
        print(f"Domain {domain} ({hospital_names[domain]}): {len(domain_indices)} images available, using {num_samples}")
        
        # Use consecutive samples since they're already randomly selected in select_diverse_sample_batch
        for i in range(num_samples):
            idx = domain_indices[i]
            
            # Display original (top row)
            img = x[idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[domain * 2, i].imshow(img)
            axes[domain * 2, i].set_title(f'Original\nLabel: {y_labels[idx]}')
            axes[domain * 2, i].axis('off')
            
            # Display reconstruction (bottom row)
            img = x_recon[idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[domain * 2 + 1, i].imshow(img)
            axes[domain * 2 + 1, i].set_title(f'Recon\nLabel: {y_labels[idx]}')
            axes[domain * 2 + 1, i].axis('off')
        
        # Fill remaining slots with empty plots
        for i in range(num_samples, samples_per_domain):
            axes[domain * 2, i].axis('off')
            axes[domain * 2 + 1, i].axis('off')
        
        # Add hospital name as y-label for the domain
        axes[domain * 2, 0].set_ylabel(f'{hospital_names[domain]}\nOriginal')
        axes[domain * 2 + 1, 0].set_ylabel(f'{hospital_names[domain]}\nRecon')
    
    plt.suptitle(f'Histopathology Image Reconstruction - Epoch {epoch}',
                 y=1.005, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(image_dir)
    plt.close()
    
    print(f"Saved reconstructions visualization for epoch {epoch}")

def visualize_conditional_generation(model, device, output_dir):
    """Generate and visualize samples conditioned on class labels (tumor/normal) and hospitals"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    samples_per_class = 5
    # Hospital names (0-indexed to match Camelyon17 hospital IDs)
    hospital_names = ['Hospital 0', 'Hospital 1', 'Hospital 2', 'Hospital 3', 'Hospital 4']
    class_names = ['Normal', 'Tumor']
    
    # Create a figure with rows for each class and columns for hospitals
    fig = plt.figure(figsize=(25, 8))
    
    # Create grid of subplots with extra space for labels
    gs = plt.GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    for label in range(2):  # 0: Normal, 1: Tumor
        for hospital in range(5):  # 5 hospitals
            # Generate samples for this class and hospital
            images = model.generate(y=label, a=hospital, num_samples=1, device=device)
            
            # Create subplot
            ax = fig.add_subplot(gs[label, hospital])
            
            # Display sample image
            img = images[0].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            
            # Add both class and hospital labels
            ax.set_title(f"{class_names[label]}\n{hospital_names[hospital]}", pad=10)
    
    plt.suptitle("Conditional Image Generation by Tissue Type and Hospital",
                 y=1.05, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'conditional_generations.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Conditional generation visualization saved to {os.path.join(output_dir, 'conditional_generations.png')}")

def save_domain_samples_visualization(x, y, metadata, epoch, output_dir):
    """
    Save the selected samples as a visualization grid, organized by hospital ID.

    Args:
        x: Images tensor
        y: Class labels (0=Normal, 1=Tumor)
        metadata: Metadata tensor containing hospital IDs and other info
        epoch: Current epoch number
        output_dir: Directory to save the visualization
    """
    # Create a figure with subplots for each hospital
    num_hospitals = 5  # 5 hospitals
    samples_per_hospital = 10  # Fixed number for consistent layout
    hospital_id = metadata[:, 0]  # Extract hospital IDs from metadata
    fig, axes = plt.subplots(num_hospitals, samples_per_hospital, figsize=(20, 4 * num_hospitals))

    # Organize images by hospital
    hospital_images = {i: [] for i in range(num_hospitals)}
    for i in range(len(x)):
        hospital = int(hospital_id[i].item())
        if hospital < num_hospitals:  # Ensure valid hospital ID
            hospital_images[hospital].append(i)
    # Process each hospital
    for hospital in range(num_hospitals):
        hospital_indices = hospital_images[hospital]

        # If we have enough images for this hospital, select the exact number needed
        num_samples = min(samples_per_hospital, len(hospital_indices))

        # Print hospital stats
        print(f"Hospital {hospital}: {len(hospital_indices)} images available, using {num_samples}")

        for i in range(num_samples):
            idx = hospital_indices[i]

            # Display image
            img = x[idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[hospital, i].imshow(img)
            axes[hospital, i].set_title(f"Label: {y[idx].item()}")
            axes[hospital, i].axis('off')

        # Fill remaining slots with empty plots
        for i in range(num_samples, samples_per_hospital):
            axes[hospital, i].axis('off')

        # Add hospital ID as y-label
        axes[hospital, 0].set_ylabel(f"Hospital {hospital}")

    plt.suptitle(f"Hospital Domain Sample Distribution - Epoch {epoch}",
                 y=1.005, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"domain_samples_epoch_{epoch}.png"))
    plt.close()

    print(f"Saved domain samples visualization for epoch {epoch}")

def generate_images_latent(model, device, data_type, output_dir, batch_data, mode, args):
    """Generate and visualize reconstructions using different latent spaces from forward pass"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    x, y, metadata = batch_data

    hospital_id = metadata[:, 0]
    
    # Move to device
    x = x.to(device)
    y = y.to(device)
    hospital_id = hospital_id.to(device)
    
    # Get latent representations through forward pass
    with torch.no_grad():
        if args.model == 'vae':
            x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model.forward(y, x, hospital_id)
            
            if mode == 'only':
                # Generate different reconstructions by zeroing out different latent variables
                zy_only = model.px(zy, torch.zeros_like(zx), torch.zeros_like(zay), torch.zeros_like(za))
                zx_only = model.px(torch.zeros_like(zy), zx, torch.zeros_like(zay), torch.zeros_like(za))
                zay_only = model.px(torch.zeros_like(zy), torch.zeros_like(zx), zay, torch.zeros_like(za))
                za_only = model.px(torch.zeros_like(zy), torch.zeros_like(zx), torch.zeros_like(zay), za)
                y_only = model.px(zy, torch.zeros_like(zx), zay, torch.zeros_like(za))
                a_only = model.px(torch.zeros_like(zy), torch.zeros_like(zx), zay, za)
                
                latent_names = [
                    'Original Image',
                    'Full Reconstruction\n(all latents)',
                    'Label Space (zy)\ncaptures tumor/normal',
                    'Content Space (zx)\ncaptures image details',
                    'Label-Hospital Space (zay)\ncaptures class-domain interaction',
                    'Hospital Space (za)\ncaptures domain style',
                    'zy + zay\ncaptures label',
                    'za + zay\ncaptures domain style'
                ]
                images_list = [x, x_recon, zy_only, zx_only, zay_only, za_only, y_only, a_only]
            elif mode == 'without':
                # Generate different reconstructions by zeroing out different latent variables
                no_zy_only = model.px(torch.zeros_like(zy), zx, zay, za)
                no_zx_only = model.px(zy, torch.zeros_like(zx), zay, za)
                no_zay_only = model.px(zy, zx, torch.zeros_like(zay), za)
                no_za_only = model.px(zy, zx, zay, torch.zeros_like(za))
                no_y_only = model.px(torch.zeros_like(zy), zx, torch.zeros_like(zay), za)
                no_a_only = model.px(zy, zx, torch.zeros_like(zay), torch.zeros_like(za))
                
                latent_names = [
                    'Original Image',
                    'Full Reconstruction (all latents)',
                    'no (zy)\ncaptures tumor/normal',
                    'no (zx)\ncaptures image details',
                    'no (zay)\ncaptures class-domain interaction',
                    'no (za)\ncaptures domain style',
                    'no (zy + zay)\nno label',
                    'no (za + zay)\nno domain'
                ]
                images_list = [x, x_recon, no_zy_only, no_zx_only, no_zay_only, no_za_only, no_y_only, no_a_only]
        
        elif args.model == 'diva':
            x_recon, z, qz, pzy, pzx, pza, _, y_hat, a_hat, zy, zx, _, za = model.forward(y, x, hospital_id)
            
            if mode == 'only':
                # Generate different reconstructions by zeroing out different latent variables
                zy_only = model.px(zy, torch.zeros_like(zx), None, torch.zeros_like(za))
                zx_only = model.px(torch.zeros_like(zy), zx, None, torch.zeros_like(za))
                za_only = model.px(torch.zeros_like(zy), torch.zeros_like(zx), None, za)
                
                latent_names = [
                    'Original Image',
                    'Full Reconstruction\n(all latents)',
                    'Label Space (zy)\ncaptures tumor/normal',
                    'Content Space (zx)\ncaptures image details',
                    'Hospital Space (za)\ncaptures domain style',
                ]
                images_list = [x, x_recon, zy_only, zx_only, za_only]
            elif mode == 'without':
                # Generate different reconstructions by zeroing out different latent variables
                no_zy_only = model.px(torch.zeros_like(zy), zx, None, za)
                no_zx_only = model.px(zy, torch.zeros_like(zx), None, za)
                no_za_only = model.px(zy, zx, None, torch.zeros_like(za))
                
                latent_names = [
                    'Original Image',
                    'Full Reconstruction (all latents)',
                    'no (zy)\ncaptures tumor/normal',
                    'no (zx)\ncaptures image details',
                    'no (za)\ncaptures domain style',
                ]
                images_list = [x, x_recon, no_zy_only, no_zx_only, no_za_only]
    
    fig_width = x.size(0) * 3 + 3  # Add extra width for the label column
    fig_height = len(images_list) * 2  # Each row gets 2 units of height
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create a grid with an extra column for labels
    gs = gridspec.GridSpec(len(images_list), x.size(0) + 1, width_ratios=[1] + [3] * x.size(0), figure=fig)

    axes = []
    for row_idx in range(len(images_list)):
        row_axes = []
        for col_idx in range(x.size(0) + 1):  # Include the extra column for labels
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
        axes.append(row_axes)

    # Add images and labels
    for row_idx, (name, images) in enumerate(zip(latent_names, images_list)):
        # Add label to the first column of the row
        axes[row_idx][0].text(0.5, 0.5, name, fontsize=12, ha='center', va='center', wrap=True)
        axes[row_idx][0].axis('off')  # Turn off the axis for the label column

        # Add images to the remaining columns
        for col_idx in range(x.size(0)):
            img = images[col_idx].cpu().detach().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[row_idx][col_idx + 1].imshow(img)
            axes[row_idx][col_idx + 1].axis('off')  # Turn off the axis for the image columns

            # Add sample info only to the first row
            if row_idx == 0:
                axes[row_idx][col_idx + 1].set_title(f"Sample {col_idx}\nLabel: {y[col_idx].item()}\nHospital: {hospital_id[col_idx].item()}")

    # Save the figure
    save_dir = os.path.join(output_dir, f'latent_reconstructions_{data_type}_{mode}.png')
    plt.suptitle(f"Latent Space Ablation - {args.model.upper()} ({mode} mode)",
                 y=1.02, fontsize=13, fontweight='bold')
    plt.subplots_adjust(left=0.2)
    plt.savefig(save_dir, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Latent reconstructions saved to {save_dir}")

def prepare_data(dataset, args):
    """
    Prepare datasets and data loaders using standard WILDS splits.

    Args:
        dataset: WILDS dataset
        args: Command-line arguments
    """
    '''transform = transforms.Compose(
        [transforms.Resize((448, 448) if args.resolution == 'high' else (64, 64)),
         transforms.ToTensor()]
    )'''

    '''transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor()]
    )'''
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )


    train_data = dataset.get_subset("train", transform=transform)
    val_data = dataset.get_subset(args.val_type, transform=transform)
    test_data = dataset.get_subset("test", transform=transform)

    # Create data loaders using standard WILDS functions
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)

    print("\nHospital distribution:")
    print("Train hospitals:", np.unique(train_data.dataset.metadata_array[train_data.indices, 0]))
    print("Val hospitals:", np.unique(val_data.dataset.metadata_array[val_data.indices, 0]))
    print("Test hospitals:", np.unique(test_data.dataset.metadata_array[test_data.indices, 0]))

    return train_loader, val_loader, test_loader

'''def initialize_model(args, num_classes, num_domains):
    """Initialize the appropriate model type."""
    if args.model == 'vae':
        return VAE(
            class_map=None,
            zy_dim=args.zy_dim,
            zx_dim=args.zx_dim,
            zay_dim=args.zay_dim,
            za_dim=args.za_dim,
            y_dim=num_classes,
            a_dim=num_domains,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            beta_3=args.beta_3,
            beta_4=args.beta_4,
            alpha_1=args.alpha_1,
            alpha_2=args.alpha_2,
            recon_weight=args.recon_weight,
            device=args.device
        )
    elif args.model == 'diva':
        return DIVA_VAE(args)'''


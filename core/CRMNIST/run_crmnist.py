import json
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import core.CRMNIST.utils
from core.CRMNIST.data_generation import generate_crmnist_dataset, CRMNIST
from core.CRMNIST.model import VAE
from tqdm import tqdm
from core.train import train
"""
CRMNIST VAE training script.

This script trains a Variational Autoencoder (VAE) on the CRMNIST dataset,
a custom version of MNIST with color and rotation transformations.

Run with:
python -m core.CRMNIST.run_crmnist --out results/ --config conf/crmnist.json
"""

def run_experiment(args):
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    reconstructions_dir = os.path.join(args.out, 'reconstructions')
    os.makedirs(reconstructions_dir, exist_ok=True)
    models_dir = os.path.join(args.out, 'models')
    os.makedirs(models_dir, exist_ok=True)
    domain_samples_dir = os.path.join(args.out, 'domain_samples')
    os.makedirs(domain_samples_dir, exist_ok=True)
    
    # Log some information
    print(f"Starting CRMNIST VAE training...")
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.out}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # Load configuration from JSON
    with open(args.config, 'r') as file:
        spec_data = json.load(file)
        
    # Prepare domain data
    domain_data = {int(key): value for key, value in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data
    class_map = spec_data['class_map']
    
    # Choose labels subset if not already chosen
    y_c, subsets = core.CRMNIST.utils.choose_label_subset(spec_data)
    spec_data['y_c'] = y_c
    
    # Generate dataset
    train_dataset = CRMNIST(spec_data, train=True, 
                           transform_intensity=args.intensity,
                           transform_decay=args.intensity_decay)
    
    test_dataset = CRMNIST(spec_data, train=False,
                          transform_intensity=args.intensity,
                          transform_decay=args.intensity_decay)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get dataset dimensions
    num_y_classes = train_dataset.dataset.num_y_classes
    num_r_classes = train_dataset.dataset.num_r_classes
    
    print(f"Dataset dimensions: y_dim={num_y_classes}, r_dim={num_r_classes}")
    
    # Initialize model
    model = VAE(class_map=class_map,
               zy_dim=args.zy_dim,
               zx_dim=args.zx_dim,
               zay_dim=args.zay_dim,
               za_dim=args.za_dim,
               y_dim=num_y_classes,
               a_dim=num_r_classes,
               beta_1=args.beta_1,
               beta_2=args.beta_2,
               beta_3=args.beta_3,
               beta_4=args.beta_4)
    
    # Move model to device
    if args.cuda:
        model = model.to(args.device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping setup
    patience = 5  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    def select_diverse_sample_batch(loader, args, samples_per_domain=10):
        """
        Select a diverse batch of samples with equal representation from each domain.
        Returns exactly 10 samples per rotation domain (0-4).
        
        Args:
            loader: DataLoader to select samples from
            args: Arguments from command line
            samples_per_domain: Number of samples to select per domain (default: 10)
            
        Returns:
            Tuple of (images, labels, color_labels, rotation_labels)
        """
        # Initialize dictionaries to store samples for each rotation domain
        rotation_samples = {i: [] for i in range(5)}  # 5 rotation domains (0-4)
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
                    
                    # Store this sample in its rotation domain
                    rotation_samples[rotation_idx].append((x[i], y[i], c[i], r[i]))
                
                # Also check if it's a red image (these might overlap with rotation domains)
                if torch.max(c[i]) > 0 and torch.argmax(c[i]).item() == 5:  # Red is index 5
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
        
        for domain in range(5):
            # Randomly sample without replacement
            if len(rotation_samples[domain]) > samples_per_domain:
                selected_indices = random.sample(range(len(rotation_samples[domain])), samples_per_domain)
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
        
        # Print final counts
        print(f"Final selection: {len(selected_x)} total images, {samples_per_domain} per rotation domain")
        
        return (selected_x, selected_y, selected_c, selected_r)

    def visualize_reconstructions(epoch, batch_data):
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
        num_domains = 5  # 5 rotation domains
        samples_per_domain = 10  # 10 samples per domain
        
        # Create a figure with 5 domains, each with 2 rows (original & reconstruction) and 10 columns (samples)
        fig, axes = plt.subplots(num_domains * 2, samples_per_domain, figsize=(20, 4 * num_domains))
        
        # Get rotation angles for titles
        rotation_angles = ['0°', '15°', '30°', '45°', '60°']
        
        # Color mapping for labels
        color_map = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'cyan', 4: 'magenta', 5: 'red'}
        
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
    
    # Function to visualize conditional generation
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
        num_domains = 5  # 5 rotation domains
        samples_per_domain = 10  # Fixed number for consistent layout
        
        # Add an extra row for red images
        fig, axes = plt.subplots(num_domains + 1, samples_per_domain, figsize=(20, 12))
        
        # Get rotation angles for titles
        rotation_angles = ['0°', '15°', '30°', '45°', '60°']
        
        # First, display red images in the top row if any
        red_images_found = 0
        for i in range(len(x)):
            if torch.max(c[i]) > 0 and torch.argmax(c[i]).item() == 5:  # Red is index 5
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
                        color_map = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'cyan', 4: 'magenta', 5: 'red'}
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
    
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
        
        # Select a diverse sample batch with images from all domains
        sample_batch = select_diverse_sample_batch(test_loader, args)
        
        with torch.no_grad():
            for batch_idx, (x, y, c, r) in enumerate(test_loader):
                x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
                
                loss = model.loss_function(y, x, r)
                test_loss += loss.item()
                
                # Calculate additional metrics
                batch_metrics = calculate_metrics(model, y, x, r)
                for k, v in batch_metrics.items():
                    metrics_sum[k] += v
        
        # Average metrics
        test_loss /= len(test_loader)
        metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}
        
        print(f'Test Loss: {test_loss:.4f}')
        for k, v in metrics_avg.items():
            print(f'Test {k}: {v:.4f}')
        
        return test_loss, metrics_avg, sample_batch
    
    # Train the model
    training_metrics = train(args, model, optimizer, train_loader, test_loader, args.device)

    # Load best model for final evaluation
    if training_metrics['best_model_state'] is not None:
        model.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Save the final model
    final_model_path = os.path.join(models_dir, f'model_checkpoint_epoch_{epoch+1}.pt')
    torch.save(model.state_dict(), final_model_path)
    
    # Final evaluation
    print("\nEvaluating model on test set...")
    
    # Add tqdm progress bar for final evaluation
    model.eval()
    test_loss = 0
    metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
    
    # Select a diverse sample batch with images from all domains
    sample_batch = select_diverse_sample_batch(test_loader, args)
    
    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")
    
    with torch.no_grad():
        for batch_idx, (x, y, c, r) in test_pbar:
            x, y, c, r = x.to(args.device), y.to(args.device), c.to(args.device), r.to(args.device)
            
            loss = model.loss_function(y, x, r)
            test_loss += loss.item()
            
            # Calculate additional metrics
            batch_metrics = calculate_metrics(model, y, x, r)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            
            # Update progress bar with current loss
            test_pbar.set_postfix(loss=loss.item())
    
    # Average metrics
    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}
    
    print(f'Test Loss: {test_loss:.4f}')
    for k, v in metrics_avg.items():
        print(f'Test {k}: {v:.4f}')
    
    final_test_loss, final_metrics, sample_batch = test_loss, metrics_avg, sample_batch
    
    # Generate final reconstructions
    visualize_reconstructions('final', sample_batch)
    
    # Generate and visualize conditional samples
    # visualize_conditional_generation(model, args.device, reconstructions_dir)
    
    # Save training results as JSON
    results = {
        'final_test_loss': final_test_loss,
        'final_metrics': final_metrics,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
    
    results_path = os.path.join(args.out, 'results.json')
    with open(results_path, 'w') as f:
        # Convert values to strings since some may not be JSON serializable
        serializable_results = {
            k: str(v) if not isinstance(v, dict) else {k2: str(v2) for k2, v2 in v.items()}
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return model


if __name__ == "__main__":
    parser = core.utils.get_parser('CRMNIST')
    parser.add_argument('--intensity', '-i', type=float, default=1.5)
    parser.add_argument('--intensity_decay', '-d', type=float, default=1.0)
    parser.add_argument('--config', type=str, default='../conf/crmnist.json')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--zy_dim', type=int, default=32)
    parser.add_argument('--zx_dim', type=int, default=32)
    parser.add_argument('--zay_dim', type=int, default=32)
    parser.add_argument('--za_dim', type=int, default=32)
    parser.add_argument('--beta_1', type=float, default=1.0)
    parser.add_argument('--beta_2', type=float, default=1.0)
    parser.add_argument('--beta_3', type=float, default=1.0)
    parser.add_argument('--beta_4', type=float, default=1.0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    
    args = parser.parse_args()
    
    # Set up CUDA if available
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.cuda:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Run experiment
    model = run_experiment(args)

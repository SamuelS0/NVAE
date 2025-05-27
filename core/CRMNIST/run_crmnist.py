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
from core.CRMNIST.data_generation import generate_crmnist_dataset
from core.CRMNIST.model import VAE
from core.train import train
from core.CRMNIST.utils import select_diverse_sample_batch, visualize_reconstructions, visualize_conditional_generation
from core.test import test

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
    spec_data['y_c'], subsets = core.CRMNIST.utils.choose_label_subset(spec_data)
    # Update domain_data with subsets
    for i, subset in subsets.items():
        if i in domain_data:
            domain_data[i]['subset'] = subset
            
    # Generate dataset
    train_dataset = generate_crmnist_dataset(spec_data, train=True)
    test_dataset = generate_crmnist_dataset(spec_data, train=False)
    
    # Create validation split from training data to avoid data leakage
    train_size = len(train_dataset)
    val_size = int(0.2 * train_size)  # Use 20% for validation
    train_size = train_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get dataset dimensions
    num_y_classes = spec_data['num_y_classes']
    num_r_classes = spec_data['num_r_classes']

    print(f"Dataset dimensions: y_dim={num_y_classes}, r_dim=num_domains={num_r_classes}")
    
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

    patience = 5

    # Train the model
    training_metrics = train(args, model, optimizer, train_loader, val_loader, args.device, patience)

    # Load best model for final evaluation
    if training_metrics['best_model_state'] is not None:
        model.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Save the final model
    final_model_path = os.path.join(models_dir, f"model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Final evaluation
    print("\nEvaluating model on test set...")
    
    # Select a diverse sample batch with images from all domains
    sample_batch = select_diverse_sample_batch(test_loader, args)
    
    test_loss, metrics_avg = test(model, test_loader, args.device)
    
    print(f'Test Loss: {test_loss:.4f}')
    for k, v in metrics_avg.items():
        print(f'Test {k}: {v:.4f}')
    
    final_test_loss, final_metrics, sample_batch = test_loss, metrics_avg, sample_batch
    
    # Generate final reconstructions
    visualize_reconstructions(model, 'final', sample_batch, args, reconstructions_dir)
    
    # Generate and visualize conditional samples
    visualize_conditional_generation(model, args.device, reconstructions_dir)
    
    # Save training results as JSON
    results = {
        'final_test_loss': final_test_loss,
        'final_metrics': final_metrics,
        'best_val_loss': training_metrics['best_val_loss'],
        'epochs_trained': training_metrics['epochs_trained']
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

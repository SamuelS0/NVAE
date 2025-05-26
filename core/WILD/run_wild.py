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
from argparse import ArgumentParser
from model_wild import VAE, VAE_LowRes
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import matplotlib.gridspec as gridspec
import sys
from model_diva import DIVA_VAE
from utils_wild import (
    prepare_data, 
    visualize_reconstructions,
    select_diverse_sample_batch,
    generate_images_latent
)
from trainer import WILDTrainer
from train import train
from test import test
"""
WILD VAE training script.

This script trains a Variational Autoencoder (VAE) on the Camelyon17-wilds dataset,


train hospital IDs: [0, 3, 4]
id_val hospital IDs: [0, 3, 4]
val hospital IDs: [1]
test hospital IDs: [2]




command: python -B WILD/run_wild.py --out results_low_res_vae/ --batch_size 128 --model vae --resolution low --epochs 50
"""

def run_experiment(dataset, args):
    #print all the arguments
    print('args: ', args)
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    reconstructions_dir = os.path.join(args.out, 'reconstructions')
    os.makedirs(reconstructions_dir, exist_ok=True)
    models_dir = os.path.join(args.out, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Log some information
    print(f"Starting WILD VAE training...")
    #print(f"Config file: {args.config}")
    print(f"Output directory: {args.out}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    train_loader, val_loader, test_loader = prepare_data(dataset, args)

    num_classes = 2
    num_domains = 5
    print(f"Dataset dimensions: y_dim={num_classes}, r_dim={num_domains}")

    
    # Initialize model
    model = initialize_model(args, num_classes, num_domains)
    
    # Move model to device
    if args.cuda:
        model = model.to(args.device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping setup
    patience = 10  # Number of epochs to wait for improvement

    training_metrics = train(args, model, optimizer, train_loader, val_loader, args.device, patience)
     
    # Save the best model
    final_model_path = os.path.join(models_dir, f"model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Load best model for final evaluation
    if training_metrics['best_model_state'] is not None:
        model.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Run final evaluation on both validation and test sets
    print("\nRunning final evaluation...")
    val_loss, val_metrics = run_evaluation(model, args, dataset, args.out, eval_type='id_val')
    test_loss, test_metrics = run_evaluation(model, args, dataset, args.out, eval_type='test')
    
    # Save training results as JSON
    results = {
        'final_val_loss': val_loss,
        'final_val_metrics': val_metrics,
        'final_test_loss': test_loss,
        'final_test_metrics': test_metrics,
        'best_validation_accuracy': training_metrics['best_val_accuracy'],
        'total_epochs_trained': training_metrics['total_epochs_trained']
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

def run_evaluation(model, args, dataset, output_dir, eval_type='test'):
    """
    Run evaluation on a trained model using either validation or test data.
    
    Args:
        model: Trained VAE model
        args: Command line arguments
        dataset: WILDS dataset
        eval_type: Type of evaluation ('test' or 'id_val')
    """
    print(f"\nEvaluating model on {eval_type} set...")
    
    # Create output directories if they don't exist
    eval_dir = os.path.join(output_dir, f'{eval_type}_evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Prepare data loader
    transform = transforms.Compose([transforms.ToTensor()])
    eval_data = dataset.get_subset(eval_type, transform=transform)
    eval_loader = get_train_loader("standard", eval_data, batch_size=args.batch_size)
    
    # Select diverse sample batch for visualization
    eval_sample_batch = select_diverse_sample_batch(eval_loader, data_type=eval_type, samples_per_domain=10)
    
    # Run evaluation
    eval_loss, metrics_avg = test(model, args.device, eval_loader, args)
    
    # Print results
    print(f'{eval_type.capitalize()} Loss: {eval_loss:.4f}')
    for k, v in metrics_avg.items():
        print(f'{eval_type.capitalize()} {k}: {v:.4f}')
    
    # Generate reconstructions
    recon_dir = os.path.join(eval_dir, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)
    image_path = os.path.join(recon_dir, f'{eval_type}_reconstructions.png')
    visualize_reconstructions(model, eval_type, eval_sample_batch, image_dir=image_path, args=args)
    
    # Generate latent space visualizations
    latent_recon_dir = os.path.join(eval_dir, 'latent_recon')
    os.makedirs(latent_recon_dir, exist_ok=True)
    generate_images_latent(model, args.device, eval_type, latent_recon_dir, eval_sample_batch, mode='without', args=args)
    generate_images_latent(model, args.device, eval_type, latent_recon_dir, eval_sample_batch, mode='only', args=args)
    
    # Save evaluation results
    results = {
        'eval_loss': float(eval_loss),
        'metrics': {k: float(v) for k, v in metrics_avg.items()},
        'eval_type': eval_type
    }
    
    results_path = os.path.join(eval_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    
    return eval_loss, metrics_avg

def get_args():
    # Base argument parser (common arguments)
    parser = ArgumentParser(description='WILD VAE Training')
    parser.add_argument('--config', type=str, default='configs/vae.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'diva'],
                        help='Model type: vae or diva')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='Enable CUDA training')
    parser.add_argument('--out', type=str, default='results', help='Output directory')
    parser.add_argument('--out_1', type=str, default='results_1', help='Output directory')
    parser.add_argument('--out_2', type=str, default='results_2', help='Output directory')
    parser.add_argument('--resolution', type=str, default='low', choices=['high', 'low'],
                        help='Image resolution: high or low')
    parser.add_argument('--val_type', type=str, default='id_val', help='Validation type: id_val or val')

    # Model-specific arguments
    parser.add_argument('--zy_dim', type=int, default=128, help='Latent dimension for zy (VAE only)')
    parser.add_argument('--zx_dim', type=int, default=128, help='Latent dimension for zx (VAE only)')
    parser.add_argument('--zay_dim', type=int, default=129, help='Latent dimension for zay (VAE only)')
    parser.add_argument('--za_dim', type=int, default=128, help='Latent dimension for za (VAE only)')
    parser.add_argument('--beta_1', type=float, default=1.0, help='Beta 1 for VAE loss')
    parser.add_argument('--beta_2', type=float, default=1.0, help='Beta 2 for VAE loss')
    parser.add_argument('--beta_3', type=float, default=1.0, help='Beta 3 for VAE loss')
    parser.add_argument('--beta_4', type=float, default=1.0, help='Beta 4 for VAE loss')
    parser.add_argument('--alpha_1', type=float, default=1.0, help='y label loss multiplier')
    parser.add_argument('--alpha_2', type=float, default=1.0, help='domain label loss multiplier')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight for reconstruction loss (VAE only)')
    parser.add_argument('--d_dim', type=int, default=5, help='Domain dimension for DIVA_VAE')
    parser.add_argument('--y_dim', type=int, default=2, help='Class dimension for DIVA_VAE')
    parser.add_argument('--z_dim', type=int, default=64,
                    help='size of latent space 1')
    parser.add_argument('--x_dim', type=int, default=96 * 96 * 3,
                    help='input size after flattening')
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=75000.0,
                        help='Multiplier for auxiliary loss on y (DIVA_VAE only)')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=100000,
                        help='Multiplier for auxiliary loss on d (DIVA_VAE only)')
    parser.add_argument('--beta_scale', type=float, default=1.0, help='Beta for KL divergence')
    #beta annealingstore true, when --beta_annealing it is true
    parser.add_argument('--beta_annealing', action='store_true', help='Beta annealing')
    return parser.parse_args()

def initialize_model(args, num_classes, num_domains):
    
    model = VAE(class_map=None,
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
                device=args.device, 
                resolution=args.resolution, 
                model=args.model)
    
    return model




if __name__ == "__main__":
    args = get_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    dataset = get_dataset(
            dataset="camelyon17", 
            download=False, 
            root_dir='/midtier/cocolab/scratch/ofn9004/WILD',
            unlabeled=False
        )
    # Run experiment
    model = run_experiment(dataset, args)
    num_classes = 2
    num_domains = 5
    run_evaluation(model, args, dataset, args.out, 'id_val')
    run_evaluation(model, args, dataset, args.out, 'test')
   

    #python -B WILD/run_wild.py --out test_vae --batch_size 256 --model vae --resolution low --epochs 20 --recon_weight 10 --alpha_1 100 --alpha_2 100 --beta_scale 2 --zy_dim 64 --zx_dim 64 --zay_dim 10 --za_dim 64 --beta_annealing
    #python -B WILD/run_wild.py --out test_diva --batch_size 256 --model diva --resolution low --epochs 20 --recon_weight 10 --alpha_1 100 --alpha_2 100 --beta_scale 2 --zy_dim 64 --zx_dim 64 --zay_dim 10 --za_dim 64 --beta_annealing









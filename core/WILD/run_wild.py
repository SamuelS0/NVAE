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
from core.WILD.model_wild import VAE
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import matplotlib.gridspec as gridspec
import sys
import time
#from model_diva import DIVA_VAE
from core.WILD.utils_wild import (
    prepare_data, 
    visualize_reconstructions,
    select_diverse_sample_batch,
    generate_images_latent
)
from core.train import train
from core.test import test
from core.utils import visualize_latent_spaces
from core.comparison.train import train_nvae, train_diva, train_dann, train_irm
"""
WILD VAE training script.

This script trains a Variational Autoencoder (VAE) on the Camelyon17-wilds dataset,


train hospital IDs: [0, 3, 4]
id_val hospital IDs: [0, 3, 4]
val hospital IDs: [1]
test hospital IDs: [2]




command: python -B WILD/run_wild.py --out results_low_res_vae/ --batch_size 128 --model vae --resolution low --epochs 50
# '/midtier/cocolab/scratch/ofn9004/WILD'
"""

def run_experiment(dataset, args):
    # Create output directories with progress indication
    print("🚀 Setting up experiment...")
    setup_tasks = [
        ("Creating output directories", lambda: os.makedirs(args.out, exist_ok=True)),
        ("Creating reconstructions directory", lambda: os.makedirs(os.path.join(args.out, 'reconstructions'), exist_ok=True)),
        ("Creating models directory", lambda: os.makedirs(os.path.join(args.out, 'models'), exist_ok=True))
    ]
    
    for desc, task in tqdm(setup_tasks, desc="Setup", unit="task"):
        task()
        time.sleep(0.1)  # Small delay for visual feedback
    
    reconstructions_dir = os.path.join(args.out, 'reconstructions')
    models_dir = os.path.join(args.out, 'models')
    
    # Log some information
    print(f"\n📊 Experiment Configuration:")
    print(f"  Output directory: {args.out}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    
    # Data preparation with progress bar
    print("\n📁 Preparing datasets...")
    with tqdm(total=3, desc="Data Prep", unit="dataset") as pbar:
        pbar.set_postfix_str("Loading train/val/test loaders")
        train_loader, val_loader, test_loader = prepare_data(dataset, args)
        pbar.update(3)

    num_classes = 2
    num_domains = 5
    print(f"\n🏥 Dataset dimensions: y_dim={num_classes}, domains={num_domains}")

    # Model initialization with progress bar
    print("\n🧠 Initializing model...")
    with tqdm(total=3, desc="Model Init", unit="step") as pbar:
        pbar.set_postfix_str("Creating model architecture")
        model = initialize_model(args, num_classes, num_domains)
        pbar.update(1)
        
        pbar.set_postfix_str("Moving to device")
        if args.cuda:
            model = model.to(args.device)
        pbar.update(1)
        
        pbar.set_postfix_str("Setting up optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        pbar.update(1)
    
    #Save model parameters
    model_params = {
        'recon_weight': args.recon_weight,
        'zy_dim': args.zy_dim,
        'zx_dim': args.zx_dim,
        'zay_dim': args.zay_dim,
        'za_dim': args.za_dim,
        'y_dim': num_classes,
        'a_dim': num_domains,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'beta_3': args.beta_3,
        'beta_4': args.beta_4,
        'alpha_1': args.alpha_1,
        'alpha_2': args.alpha_2,
        'beta_scale': args.beta_scale,
        'diva': args.model
        }
    with open(os.path.join(args.out, 'model_params.json'), 'w') as f:
        json.dump(model_params, f)
    
    # Early stopping setup
    patience = 10  # Number of epochs to wait for improvement
    
    print(f"\n🎯 Starting training for {args.epochs} epochs with patience={patience}...")
    training_metrics = train(args, model, optimizer, train_loader, val_loader, args.device, patience, trainer_class=WILDTrainer)
     
    # Save the best model
    print("\n💾 Saving final model...")
    final_model_path = os.path.join(models_dir, f"model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(model.state_dict(), final_model_path)
    
    if training_metrics['best_model_state'] is not None:
        model.load_state_dict(training_metrics['best_model_state'])
        print("✅ Loaded best model for final evaluation")
    
    # Final evaluation with progress indication
    print("\n🧪 Evaluating model on test set...")
    
    # Select a diverse sample batch with images from all domains
    print("  📊 Selecting diverse test samples...")
    test_sample_batch = select_diverse_sample_batch(test_loader, data_type='test', samples_per_domain=10)
    
    print("  🔍 Running test evaluation...")
    test_loss, metrics_avg = test(model, test_loader, dataset_type='wild', device=args.device)
    
    print(f'\n📈 Final Test Results:')
    print(f'  Test Loss: {test_loss:.4f}')
    for k, v in metrics_avg.items():
        print(f'  Test {k}: {v:.4f}')
    
    final_test_loss, final_metrics, test_sample_batch = test_loss, metrics_avg, test_sample_batch
    
    #Generate final reconstructions with progress
    print("\n🎨 Generating visualizations...")
    with tqdm(total=2, desc="Visualizations", unit="plot") as pbar:
        pbar.set_postfix_str("Test reconstructions")
        image_dir = os.path.join(reconstructions_dir, f'test_reconstructions.png')
        visualize_reconstructions(model, 'test', test_sample_batch, image_dir=image_dir, args=args)
        pbar.update(1)
        
        pbar.set_postfix_str("Saving results")
        # Save training results as JSON
        results = {
            'final_test_loss': final_test_loss,
            'final_metrics': final_metrics,
            'best_val_accuracy': training_metrics['best_val_accuracy'],
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
        pbar.update(1)
    
    print(f"💾 Results saved to {results_path}")
    
    return model

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
    parser.add_argument('--resolution', type=str, default='low', choices=['high', 'low'],
                        help='Image resolution: high or low')
    parser.add_argument('--val_type', type=str, default='id_val', help='Validation type: id_val or val')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='None', 
                        help='Directory to store/load dataset (default: ~/data/wilds)')
    parser.add_argument('--download', action='store_true', default=True,
                        help='Download dataset if not found locally')

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
    parser.add_argument('--num_y_classes', type=int, default=2, help='Class dimension for DIVA_VAE')
    parser.add_argument('--num_r_classes', type=int, default=5, help='Domain dimension for DIVA_VAE')
    parser.add_argument('--z_dim', type=int, default=64,
                    help='size of latent space 1')
    parser.add_argument('--x_dim', type=int, default=96 * 96 * 3,
                    help='input size after flattening')
    parser.add_argument('--beta_scale', type=float, default=1.0, help='Beta for KL divergence')
    #beta annealingstore true, when --beta_annealing it is true
    parser.add_argument('--beta_annealing', action='store_true', help='Beta annealing')
    parser.add_argument('--dataset', type=str, default='wild')
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
    print("🔬 WILD VAE Training Pipeline")
    print("=" * 50)
    
    # Parse arguments
    args = get_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # Setup data directory
    if args.data_dir is None:
        # Use default data directory in user's home folder
        args.data_dir = os.path.expanduser("~/data/wilds")
    
    # Create data directory if it doesn't exist
    print(f"📁 Data directory: {args.data_dir}")
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Check if dataset exists locally
    dataset_path = os.path.join(args.data_dir, "camelyon17_v1.0")
    dataset_exists = os.path.exists(dataset_path)
    
    if not dataset_exists and args.download:
        print("⚠️  Dataset not found locally. Will download (~10GB) - this may take a while...")
        print("☕ Grab a coffee! First download can take 30+ minutes depending on your connection.")
    elif not dataset_exists and not args.download:
        print("❌ Dataset not found and download is disabled.")
        print(f"   Please download manually or use --download flag")
        print(f"   Expected location: {dataset_path}")
        exit(1)
    elif dataset_exists:
        print("✅ Dataset found locally, skipping download.")
    
    # Dataset loading with progress
    print(f"\n📦 Loading Camelyon17 dataset from {args.data_dir}...")
    with tqdm(total=1, desc="Dataset", unit="dataset") as pbar:
        if dataset_exists:
            pbar.set_postfix_str("Loading from local cache")
        else:
            pbar.set_postfix_str("Downloading and loading (this will take a while)")
        
        try:
            dataset = get_dataset(
                    dataset="camelyon17", 
                    download=args.download, 
                    root_dir=args.data_dir,
                    unlabeled=False
                )
            pbar.update(1)
        except Exception as e:
            pbar.close()
            print(f"❌ Error loading dataset: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Check your internet connection")
            print("2. Ensure you have enough disk space (~10GB)")
            print("3. Try a different data directory with --data_dir /path/to/data")
            print("4. Check write permissions for the data directory")
            exit(1)
    
    print("✅ Dataset loaded successfully!")
    
    # Run main experiment
    #print("\n🎯 Running main experiment...")
    #model = run_experiment(dataset, args)
    
    
    
    
    
    
    
    #run comparison experiments
    print("\n🎯 Running comparison experiments...")
    
    train_loader, val_loader, test_loader = prepare_data(dataset, args)
    spec_data = {'class_map': None, 'num_y_classes': 2, 'num_r_classes': 5}
    
    #run dann
    train_dann(args, spec_data, train_loader, test_loader, dataset='wild')

    #run irm
    train_irm(args, spec_data, train_loader, test_loader, dataset='wild')

    #run diva
    train_diva(args, spec_data, train_loader, test_loader, dataset='wild')

    #run nvae   
    train_nvae(args, spec_data, train_loader, test_loader, dataset='wild')
    
    







    # Post-training analysis with progress bars
    print("\n🔬 Running post-training analysis...")
    
    # model.eval()
    
    #Prepare validation data for latent analysis
    # print("  📊 Preparing validation data for latent analysis...")
    # transform = transforms.Compose([transforms.ToTensor()])
    # final_val_data = dataset.get_subset(args.val_type, transform=transform)
    # val_loader = get_train_loader("standard", final_val_data, batch_size=10)
    # val_x, val_y, val_metadata = next(iter(val_loader))
    
    # # # Generate latent space analysis with progress
    # # latent_recon_dir = os.path.join(args.out, 'latent_visualization')
    # # os.makedirs(latent_recon_dir, exist_ok=True)

    # latent_space_dir = os.path.join(args.out, 'latent_space', 'wild_latent_space')
    # os.makedirs(latent_space_dir, exist_ok=True)

    # visualize_latent_spaces(model=model, dataloader=val_loader, device=args.device, type='wild', save_path=latent_space_dir)
    
    # latent_analysis_tasks = [
    #     ("Latent analysis (without components)", lambda: generate_images_latent(
    #         model, args.device, 'id_val', latent_recon_dir, (val_x, val_y, val_metadata), 'without', args)),
    #     ("Latent analysis (individual components)", lambda: generate_images_latent(
    #         model, args.device, 'id_val', latent_recon_dir, (val_x, val_y, val_metadata), 'only', args))
    # ]
    
    # for desc, task in tqdm(latent_analysis_tasks, desc="Latent Analysis", unit="analysis"):
    #     task()
    
    # print("\n🎉 Training and analysis complete!")
    # print(f"📁 All results saved to: {args.out}")
    # print(f"💾 Dataset cached at: {args.data_dir}")
    # print("=" * 50)




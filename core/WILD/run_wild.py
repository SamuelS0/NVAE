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
from core.WILD.disentanglement_analysis import analyze_disentanglement
from core.comparison.train import train_nvae, train_diva, train_dann, train_irm, train_staged_nvae
from core.WILD.trainer import WILDTrainer
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
    patience = 100  # Number of epochs to wait for improvement
    
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
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory to store/load dataset (default: ~/data/wilds)')
    parser.add_argument('--download', action='store_true', default=True,
                        help='Download dataset if not found locally')

    # Model-specific arguments
    parser.add_argument('--zy_dim', type=int, default=128, help='Latent dimension for zy (VAE only)')
    parser.add_argument('--zx_dim', type=int, default=128, help='Latent dimension for zx (VAE only)')
    parser.add_argument('--zay_dim', type=int, default=128, help='Latent dimension for zay (VAE only)')
    parser.add_argument('--za_dim', type=int, default=128, help='Latent dimension for za (VAE only)')
    parser.add_argument('--beta_1', type=float, default=1.0, help='Beta 1 for VAE loss')
    parser.add_argument('--beta_2', type=float, default=1.0, help='Beta 2 for VAE loss')
    parser.add_argument('--beta_3', type=float, default=1.0, help='Beta 3 for VAE loss')
    parser.add_argument('--beta_4', type=float, default=1.0, help='Beta 4 for VAE loss')
    parser.add_argument('--alpha_1', type=float, default=1000.0, help='y label loss multiplier')
    parser.add_argument('--alpha_2', type=float, default=1000.0, help='domain label loss multiplier')
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
    # Model selection arguments
    parser.add_argument('--models', type=str, nargs='+', default=['nvae', 'diva', 'dann', 'irm'],
                   choices=['nvae', 'diva', 'dann', 'irm'],
                   help='Which models to train and test (default: all)')
    parser.add_argument('--skip_training', action='store_true', default=False,
                   help='Skip training and only do visualization (requires pre-trained models)')
    parser.add_argument('--dataset', type=str, default='wild')
    
    # Staged training arguments
    parser.add_argument('--staged_training', action='store_true', help='Use staged training for better disentanglement')
    parser.add_argument('--stage1_epochs', type=int, default=None, help='Number of epochs for stage 1 (za, zy only)')
    parser.add_argument('--stage2_epochs', type=int, default=None, help='Number of epochs for stage 2 (gradual zay)')
    parser.add_argument('--use_independence_penalty', action='store_true', default=True, help='Use independence penalty in stage 2')
    parser.add_argument('--no_independence_penalty', dest='use_independence_penalty', action='store_false', help='Disable independence penalty in stage 2')
    parser.add_argument('--independence_penalty', type=float, default=10.0, help='Weight for independence penalty in stage 2')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    
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
    
    
    
    
    
    
    
    # =============================================================================
    # MULTI-MODEL COMPARISON EXPERIMENTS
    # =============================================================================
    print("\n🎯 Running multi-model comparison experiments...")
    
    # Prepare data and directories
    train_loader, val_loader, test_loader = prepare_data(dataset, args)
    spec_data = {'class_map': None, 'num_y_classes': 2, 'num_r_classes': 5}
    
    os.makedirs(args.out, exist_ok=True)
    models_dir = os.path.join(args.out, 'models')
    os.makedirs(models_dir, exist_ok=True)
    latent_space_dir = os.path.join(args.out, 'latent_space')
    os.makedirs(latent_space_dir, exist_ok=True)
    
    # Dictionary to store all trained models
    trained_models = {}
    
    # Model parameters for saving
    model_params = {
        'zy_dim': args.zy_dim,
        'zx_dim': args.zx_dim,
        'zay_dim': args.zay_dim,
        'za_dim': args.za_dim,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'beta_3': args.beta_3,
        'beta_4': args.beta_4,
        'alpha_1': args.alpha_1,
        'alpha_2': args.alpha_2,
    }
    
    print(f"\nSelected models to run: {args.models}")
    if args.skip_training:
        print("⚠️  Skipping training - will only do visualization")
    
    # =============================================================================
    # 1. TRAIN AND TEST NVAE MODEL
    # =============================================================================
    if 'nvae' in args.models:
        print("\n" + "="*60)
        print("🔥 TRAINING NVAE MODEL")
        print("="*60)
        
        model_params['diva'] = False
        with open(os.path.join(args.out, 'nvae_model_params.json'), 'w') as f:
            json.dump(model_params, f)
        
        if not args.skip_training:
            # Choose training method based on arguments
            if args.staged_training:
                print("🎯 Using staged training for better disentanglement...")
                nvae_model, nvae_metrics = train_staged_nvae(args, spec_data, train_loader, val_loader, dataset='wild')
            else:
                print("🎯 Using standard training...")
                nvae_model, nvae_metrics = train_nvae(args, spec_data, train_loader, val_loader, dataset='wild')
            
            trained_models['nvae'] = nvae_model
            
            # Save NVAE model
            nvae_model_path = os.path.join(models_dir, f"nvae_model_epoch_{nvae_metrics['best_model_epoch']}.pt")
            torch.save(nvae_model.state_dict(), nvae_model_path)
            print(f"NVAE model saved to: {nvae_model_path}")
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained NVAE model for visualization...")
            # You would implement model loading here
            # nvae_model = load_nvae_model(...)
            
        # Visualize NVAE latent spaces
        if 'nvae_model' in locals():
            print("🎨 Generating NVAE latent space visualization...")
            nvae_latent_path = os.path.join(latent_space_dir, 'nvae_latent_spaces.png')
            nvae_model.visualize_latent_spaces(test_loader, args.device, save_path=nvae_latent_path)
    
    # =============================================================================
    # 2. TRAIN AND TEST DIVA MODEL
    # =============================================================================
    if 'diva' in args.models:
        print("\n" + "="*60)
        print("🔥 TRAINING DIVA MODEL")
        print("="*60)
        
        model_params['diva'] = True
        with open(os.path.join(args.out, 'diva_model_params.json'), 'w') as f:
            json.dump(model_params, f)
        
        if not args.skip_training:
            diva_model, diva_metrics = train_diva(args, spec_data, train_loader, val_loader, dataset='wild')
            trained_models['diva'] = diva_model
            
            # Save DIVA model
            diva_model_path = os.path.join(models_dir, f"diva_model_epoch_{diva_metrics['best_model_epoch']}.pt")
            torch.save(diva_model.state_dict(), diva_model_path)
            print(f"DIVA model saved to: {diva_model_path}")
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained DIVA model for visualization...")
            # diva_model = load_diva_model(...)
            
        # Visualize DIVA latent spaces
        if 'diva_model' in locals():
            print("🎨 Generating DIVA latent space visualization...")
            diva_latent_path = os.path.join(latent_space_dir, 'diva_latent_spaces.png')
            diva_model.visualize_latent_spaces_diva(test_loader, args.device, save_path=diva_latent_path)
    
    # =============================================================================
    # 3. TRAIN AND TEST DANN MODEL
    # =============================================================================
    if 'dann' in args.models:
        print("\n" + "="*60)
        print("🔥 TRAINING DANN MODEL")
        print("="*60)
        
        with open(os.path.join(args.out, 'dann_model_params.json'), 'w') as f:
            json.dump(model_params, f)
        
        if not args.skip_training:
            dann_model, dann_metrics = train_dann(args, spec_data, train_loader, val_loader, dataset='wild')
            trained_models['dann'] = dann_model
            
            # Save DANN model
            dann_model_path = os.path.join(models_dir, f"dann_model_epoch_{dann_metrics['best_model_epoch']}.pt")
            torch.save(dann_model.state_dict(), dann_model_path)
            print(f"DANN model saved to: {dann_model_path}")
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained DANN model for visualization...")
            # dann_model = load_dann_model(...)
            
        # Visualize DANN latent space
        if 'dann_model' in locals():
            print("🎨 Generating DANN latent space visualization...")
            dann_latent_path = os.path.join(latent_space_dir, 'dann_latent_spaces.png')
            dann_model.visualize_latent_space(test_loader, args.device, save_path=dann_latent_path)
    
    # =============================================================================
    # 4. TRAIN AND TEST IRM MODEL
    # =============================================================================
    if 'irm' in args.models:
        print("\n" + "="*60)
        print("🔥 TRAINING IRM MODEL")
        print("="*60)
        
        with open(os.path.join(args.out, 'irm_model_params.json'), 'w') as f:
            json.dump(model_params, f)
        
        if not args.skip_training:
            irm_model, irm_metrics = train_irm(args, spec_data, train_loader, val_loader, dataset='wild')
            trained_models['irm'] = irm_model
            
            # Save IRM model
            irm_model_path = os.path.join(models_dir, f"irm_model_epoch_{irm_metrics['best_model_epoch']}.pt")
            torch.save(irm_model.state_dict(), irm_model_path)
            print(f"IRM model saved to: {irm_model_path}")
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained IRM model for visualization...")
            # irm_model = load_irm_model(...)
            
        # Visualize IRM latent space
        if 'irm_model' in locals():
            print("🎨 Generating IRM latent space visualization...")
            irm_latent_path = os.path.join(latent_space_dir, 'irm_latent_spaces.png')
            if hasattr(irm_model, 'visualize_latent_space'):
                irm_model.visualize_latent_space(test_loader, args.device, save_path=irm_latent_path)
            else:
                # Use the generic visualization function
                from core.utils import balanced_sample_for_visualization
                features_dict, labels_dict, stats = balanced_sample_for_visualization(
                    model=irm_model, dataloader=test_loader, device=args.device, 
                    model_type="irm", max_samples=3000, dataset_type="wild"
                )
                print(f"IRM features extracted: {features_dict['features'].shape}")
    
    # =============================================================================
    # 5. POST-TRAINING ANALYSIS (for NVAE/DIVA models only)
    # =============================================================================
    if ('nvae' in args.models and 'nvae_model' in locals()) or ('diva' in args.models and 'diva_model' in locals()):
        print("\n" + "="*60)
        print("🔬 RUNNING POST-TRAINING ANALYSIS")
        print("="*60)
        
        # Choose the model for analysis (prefer NVAE if available)
        analysis_model = locals().get('nvae_model') or locals().get('diva_model')
        if analysis_model:
            analysis_model.eval()
            
            # Prepare validation data for latent analysis
            print("  📊 Preparing validation data for latent analysis...")
            transform = transforms.Compose([transforms.ToTensor()])
            final_val_data = dataset.get_subset(args.val_type, transform=transform)
            val_loader_analysis = get_train_loader("standard", final_val_data, batch_size=10)
            val_x, val_y, val_metadata = next(iter(val_loader_analysis))
            
            # Generate latent space analysis with progress
            latent_recon_dir = os.path.join(args.out, 'latent_visualization')
            os.makedirs(latent_recon_dir, exist_ok=True)
            
            # Analyze disentanglement quality
            print("  🔬 Analyzing disentanglement quality...")
            disentanglement_dir = os.path.join(args.out, 'disentanglement_analysis')
            os.makedirs(disentanglement_dir, exist_ok=True)
            
            # Standard analysis (new predictors)
            print("    📊 Running standard disentanglement analysis (new predictors)...")
            disentanglement_results = analyze_disentanglement(analysis_model, val_loader_analysis, args.device, save_path=disentanglement_dir)
            
            # Built-in predictor analysis
            print("    📊 Running built-in predictor analysis...")
            from core.WILD.disentanglement_analysis import analyze_disentanglement_builtin_predictors
            builtin_results = analyze_disentanglement_builtin_predictors(analysis_model, val_loader_analysis, args.device, save_path=disentanglement_dir)
            
            latent_analysis_tasks = [
                ("Latent analysis (without components)", lambda: generate_images_latent(
                    analysis_model, args.device, 'id_val', latent_recon_dir, (val_x, val_y, val_metadata), 'without', args)),
                ("Latent analysis (individual components)", lambda: generate_images_latent(
                    analysis_model, args.device, 'id_val', latent_recon_dir, (val_x, val_y, val_metadata), 'only', args))
            ]
            
            for desc, task in tqdm(latent_analysis_tasks, desc="Latent Analysis", unit="analysis"):
                task()
    
    # =============================================================================
    # 6. SUMMARY AND COMPARISON
    # =============================================================================
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)
    
    print("Models trained and saved:")
    for model_name, model in trained_models.items():
        print(f"  ✅ {model_name.upper()}: {model.__class__.__name__}")
    
    print(f"\nVisualization files created:")
    for model_name in args.models:
        if model_name == 'nvae' and 'nvae_model' in locals():
            print(f"  📈 NVAE latent spaces: {os.path.join(latent_space_dir, 'nvae_latent_spaces.png')}")
        elif model_name == 'diva' and 'diva_model' in locals():
            print(f"  📈 DIVA latent spaces: {os.path.join(latent_space_dir, 'diva_latent_spaces.png')}")
        elif model_name == 'dann' and 'dann_model' in locals():
            print(f"  📈 DANN latent spaces: {os.path.join(latent_space_dir, 'dann_latent_spaces.png')}")
        elif model_name == 'irm' and 'irm_model' in locals():
            print(f"  📈 IRM latent spaces: {os.path.join(latent_space_dir, 'irm_latent_spaces.png')}")
    
    print(f"\nModel files saved:")
    # Model file paths would be printed here if metrics are available
    
    # Training metrics summary
    print(f"\nTraining metrics:")
    if not args.skip_training:
        metrics_summary = {}
        if 'nvae' in args.models and 'nvae_metrics' in locals():
            metrics_summary['NVAE'] = nvae_metrics
        if 'diva' in args.models and 'diva_metrics' in locals():
            metrics_summary['DIVA'] = diva_metrics
        if 'dann' in args.models and 'dann_metrics' in locals():
            metrics_summary['DANN'] = dann_metrics
        if 'irm' in args.models and 'irm_metrics' in locals():
            metrics_summary['IRM'] = irm_metrics
        
        for model_name, metrics in metrics_summary.items():
            print(f"  {model_name}:")
            print(f"    Best epoch: {metrics.get('best_model_epoch', 'N/A')}")
            print(f"    Best val loss: {float(metrics.get('best_val_loss', metrics.get('best_val_accuracy', 'N/A'))):.4f}")
            print(f"    Epochs trained: {metrics.get('epochs_trained', metrics.get('total_epochs_trained', 'N/A'))}")
    else:
        print("  Training was skipped - no metrics available")
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 All results saved to: {args.out}")
    print(f"💾 Dataset cached at: {args.data_dir}")
    
    # Store trained models for potential future use
    print(f"\nTrained models available in memory: {list(trained_models.keys())}")
    print("=" * 50)




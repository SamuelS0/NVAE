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
from core.WILD.disentanglement_visualization import (
    visualize_disentanglement,
    visualize_latent_interpolation,
    visualize_factor_traversal
)
from core.comparison.train import train_nvae, train_diva, train_dann, train_irm, train_staged_nvae, train_dann_augmented
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

import glob
from core.comparison.dann import DANN
from core.comparison.irm import IRM
from core.CRMNIST.dann_model import AugmentedDANN


def load_model_checkpoint(models_dir, model_name, spec_data, args):
    """
    Load a pre-trained model checkpoint for WILD experiments.

    Args:
        models_dir: Directory containing model checkpoints
        model_name: Name of the model ('nvae', 'diva', 'dann', 'dann_augmented', 'irm')
        spec_data: Dataset specification data
        args: Command line arguments

    Returns:
        Loaded model and metrics (if available)
    """
    # Look for model files with different naming patterns
    model_patterns = [
        f"{model_name}_model_epoch_*.pt",
        f"{model_name}_model.pt",
        f"{model_name}.pt"
    ]

    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(os.path.join(models_dir, pattern))
        if matches:
            # Get the most recent model file
            model_path = max(matches, key=os.path.getctime)
            break

    if model_path is None:
        print(f"❌ No {model_name.upper()} model checkpoint found in {models_dir}")
        print(f"   Looking for patterns: {model_patterns}")
        return None, None

    print(f"📁 Loading {model_name.upper()} model from: {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=args.device)

        # Initialize model based on type
        if model_name == 'nvae':
            model = VAE(
                class_map=None,
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zay_dim=args.zay_dim,
                za_dim=args.za_dim,
                y_dim=spec_data['num_y_classes'],
                a_dim=spec_data['num_r_classes'],
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                beta_3=args.beta_3,
                beta_4=args.beta_4,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
                recon_weight=args.recon_weight,
                device=args.device,
                resolution=args.resolution,
                model='vae'
            )

        elif model_name == 'diva':
            model = VAE(
                class_map=None,
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zay_dim=args.zay_dim,
                za_dim=args.za_dim,
                y_dim=spec_data['num_y_classes'],
                a_dim=spec_data['num_r_classes'],
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                beta_3=args.beta_3,
                beta_4=args.beta_4,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
                recon_weight=args.recon_weight,
                device=args.device,
                resolution=args.resolution,
                model='diva'
            )

        elif model_name == 'dann':
            z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim
            model = DANN(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'wild')

        elif model_name == 'dann_augmented':
            # Redistribute dimensions for AugmentedDANN to match total of other models
            total_dim = args.zy_dim + args.zx_dim + args.zay_dim + args.za_dim
            zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)
            zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)
            zdy_aug = total_dim // 3

            model = AugmentedDANN(
                class_map=None,  # WILD doesn't use class_map
                zy_dim=zy_aug,
                zd_dim=zd_aug,
                zdy_dim=zdy_aug,
                y_dim=spec_data['num_y_classes'],
                d_dim=spec_data['num_r_classes'],
                lambda_reversal=getattr(args, 'lambda_reversal', 1.0),
                sparsity_weight=getattr(args, 'sparsity_weight', 0.01),
                alpha_y=args.alpha_1,
                alpha_d=args.alpha_2,
                beta_adv=getattr(args, 'beta_adv', 0.1),
                image_size=96  # WILD uses 96x96 images
            )

        elif model_name == 'irm':
            z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim
            model = IRM(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'wild',
                       penalty_weight=1e4, penalty_anneal_iters=500)

        # Load state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            metrics = checkpoint.get('training_metrics', {})
            print(f"✅ Loaded {model_name.upper()} model with metrics: {metrics}")
        else:
            # Assume it's just the state dict
            model.load_state_dict(checkpoint)
            metrics = {}
            print(f"✅ Loaded {model_name.upper()} model")

        # Move to device
        model = model.to(args.device)
        model.eval()

        return model, metrics

    except Exception as e:
        print(f"❌ Error loading {model_name.upper()} model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


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
    parser.add_argument('--alpha_1', type=float, default=150.0, help='y label loss multiplier')
    parser.add_argument('--alpha_2', type=float, default=40.0, help='domain label loss multiplier')
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
    parser.add_argument('--models', type=str, nargs='+', default=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                   choices=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
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
    parser.add_argument('--use_zay_annealing', action='store_true', default=True, help='Enable zay capacity annealing in staged training (default: True)')
    parser.add_argument('--no_zay_annealing', dest='use_zay_annealing', action='store_false', help='Disable zay capacity annealing in staged training')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')

    # L1 sparsity penalty arguments
    parser.add_argument('--l1_lambda_zy', type=float, default=10.0,
                       help='L1 penalty weight for zy latent (default: 10.0)')
    parser.add_argument('--l1_lambda_zx', type=float, default=10.0,
                       help='L1 penalty weight for zx latent (default: 10.0)')
    parser.add_argument('--l1_lambda_zay', type=float, default=100.0,
                       help='L1 penalty weight for zay latent (default: 100.0)')
    parser.add_argument('--l1_lambda_za', type=float, default=10.0,
                       help='L1 penalty weight for za latent (default: 10.0)')

    # AugmentedDANN-specific parameters
    parser.add_argument('--lambda_reversal', type=float, default=1.0,
                       help='Lambda parameter for gradient reversal in AugmentedDANN (default: 1.0)')
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                       help='Weight for sparsity penalty on zdy in AugmentedDANN (default: 0.01)')
    parser.add_argument('--beta_adv', type=float, default=0.1,
                       help='Weight for adversarial loss in AugmentedDANN (default: 0.1)')

    # OOD Domain Generalization arguments
    parser.add_argument('--no_ood', action='store_true', default=False,
                       help='Disable OOD mode and include all hospitals in training (default: OOD enabled, last hospital withheld)')
    parser.add_argument('--ood_hospital_idx', type=int, default=None,
                       help='Specific hospital index to withhold for OOD testing (0-4). Default: 4 (last hospital) when OOD enabled')

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
                model=args.model,
                l1_lambda_zy=args.l1_lambda_zy,
                l1_lambda_zx=args.l1_lambda_zx,
                l1_lambda_zay=args.l1_lambda_zay,
                l1_lambda_za=args.l1_lambda_za)

    return model



if __name__ == "__main__":
    print("🔬 WILD VAE Training Pipeline")
    print("=" * 50)
    
    # Parse arguments
    args = get_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Validate resolution argument
    if args.resolution == 'high':
        print("❌ ERROR: High resolution (448x448) is not supported for WILD dataset.")
        print("   The Camelyon17-WILDS dataset provides 96x96 patches natively.")
        print("   Upscaling would not add meaningful information and would only slow training.")
        print("   Please use --resolution low (or omit the flag, as 'low' is the default).")
        exit(1)

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

    # =============================================================================
    # OOD DOMAIN GENERALIZATION SETUP
    # =============================================================================
    # Determine which hospital to withhold for OOD testing
    ood_hospital = None
    if not args.no_ood:
        # Default: withhold last hospital (hospital 4)
        ood_hospital = args.ood_hospital_idx if args.ood_hospital_idx is not None else 4
        print(f"\n{'='*80}")
        print(f"🎯 OOD MODE ENABLED: Withholding hospital {ood_hospital} for OOD testing")
        print(f"   Training on hospitals: {[i for i in range(5) if i != ood_hospital]}")
        print(f"   OOD test hospital: {ood_hospital}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"📊 STANDARD MODE: Using all 5 hospitals for training")
        print(f"{'='*80}\n")

    # Prepare data and directories
    train_loader, val_loader, test_loader = prepare_data(
        dataset, args,
        exclude_hospitals=[ood_hospital] if ood_hospital is not None else None
    )

    # Create OOD test loader if in OOD mode
    if ood_hospital is not None:
        from torchvision import transforms
        from torch.utils.data import Subset, DataLoader
        from core.WILD.utils_wild import get_eval_loader, custom_collate_fn

        # Get test data and filter to only OOD hospital
        transform = transforms.Compose([transforms.ToTensor()])
        test_data_full = dataset.get_subset("test", transform=transform)

        # Create OOD test set (only withheld hospital)
        ood_test_indices = [i for i in test_data_full.indices
                           if test_data_full.dataset.metadata_array[i, 0] == ood_hospital]
        ood_test_data = Subset(test_data_full.dataset, ood_test_indices)
        ood_test_loader = DataLoader(ood_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

        # Create ID test set (exclude withheld hospital) - rename test_loader to id_test_loader
        id_test_indices = [i for i in test_data_full.indices
                          if test_data_full.dataset.metadata_array[i, 0] != ood_hospital]
        id_test_data = Subset(test_data_full.dataset, id_test_indices)
        id_test_loader = DataLoader(id_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

        # For backward compatibility, keep test_loader pointing to id_test_loader
        test_loader = id_test_loader

        print(f"\n📊 Test set distribution:")
        print(f"   ID Test: {len(id_test_data)} samples (hospitals: {np.unique([test_data_full.dataset.metadata_array[i, 0] for i in id_test_indices])})")
        print(f"   OOD Test: {len(ood_test_data)} samples (hospital: {ood_hospital})")
    else:
        ood_test_loader = None
        id_test_loader = test_loader

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
        if 'nvae' in trained_models:
            nvae_model = trained_models['nvae']
            print("🎨 Generating NVAE latent space visualization...")
            nvae_latent_path = os.path.join(latent_space_dir, 'nvae_latent_spaces.png')
            nvae_model.visualize_latent_spaces(test_loader, args.device, save_path=nvae_latent_path)

            # Evaluate latent expressiveness
            print("🧪 Evaluating NVAE latent expressiveness...")
            nvae_expressiveness_dir = os.path.join(args.out, 'nvae_expressiveness')
            os.makedirs(nvae_expressiveness_dir, exist_ok=True)
            from core.WILD.latent_expressiveness import evaluate_latent_expressiveness
            nvae_expressiveness = evaluate_latent_expressiveness(
                nvae_model, train_loader, val_loader, test_loader, args.device, nvae_expressiveness_dir
            )

            # Disentanglement visualizations for NVAE
            print("🎨 Generating NVAE disentanglement visualizations...")
            nvae_disentangle_dir = os.path.join(args.out, 'nvae_disentanglement')
            os.makedirs(nvae_disentangle_dir, exist_ok=True)

            try:
                # 1. Main disentanglement visualization
                disentangle_path = os.path.join(nvae_disentangle_dir, 'disentanglement')
                visualize_disentanglement(
                    nvae_model, val_loader, args.device,
                    save_path=disentangle_path,
                    num_variations=7,
                    num_examples=3
                )
                print(f"   ✅ Disentanglement visualization saved")

                # 2. Latent interpolation
                interp_path = os.path.join(nvae_disentangle_dir, 'interpolation')
                visualize_latent_interpolation(
                    nvae_model, val_loader, args.device,
                    save_path=interp_path,
                    num_steps=7
                )
                print(f"   ✅ Interpolation visualization saved")

                # 3. Factor traversal
                traversal_path = os.path.join(nvae_disentangle_dir, 'traversal')
                visualize_factor_traversal(
                    nvae_model, args.device,
                    save_path=traversal_path,
                    num_steps=7
                )
                print(f"   ✅ Factor traversal visualization saved")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not generate some disentanglement visualizations: {str(e)}")

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
        if 'diva' in trained_models:
            diva_model = trained_models['diva']
            print("🎨 Generating DIVA latent space visualization...")
            diva_latent_path = os.path.join(latent_space_dir, 'diva_latent_spaces.png')
            diva_model.visualize_latent_spaces_diva(test_loader, args.device, save_path=diva_latent_path)

            # Evaluate latent expressiveness
            print("🧪 Evaluating DIVA latent expressiveness...")
            diva_expressiveness_dir = os.path.join(args.out, 'diva_expressiveness')
            os.makedirs(diva_expressiveness_dir, exist_ok=True)
            from core.WILD.latent_expressiveness import evaluate_latent_expressiveness
            diva_expressiveness = evaluate_latent_expressiveness(
                diva_model, train_loader, val_loader, test_loader, args.device, diva_expressiveness_dir
            )

            # Disentanglement visualizations for DIVA
            print("🎨 Generating DIVA disentanglement visualizations...")
            diva_disentangle_dir = os.path.join(args.out, 'diva_disentanglement')
            os.makedirs(diva_disentangle_dir, exist_ok=True)

            try:
                # 1. Main disentanglement visualization
                disentangle_path = os.path.join(diva_disentangle_dir, 'disentanglement')
                visualize_disentanglement(
                    diva_model, val_loader, args.device,
                    save_path=disentangle_path,
                    num_variations=7,
                    num_examples=3
                )
                print(f"   ✅ Disentanglement visualization saved")

                # 2. Latent interpolation
                interp_path = os.path.join(diva_disentangle_dir, 'interpolation')
                visualize_latent_interpolation(
                    diva_model, val_loader, args.device,
                    save_path=interp_path,
                    num_steps=7
                )
                print(f"   ✅ Interpolation visualization saved")

                # 3. Factor traversal
                traversal_path = os.path.join(diva_disentangle_dir, 'traversal')
                visualize_factor_traversal(
                    diva_model, args.device,
                    save_path=traversal_path,
                    num_steps=7
                )
                print(f"   ✅ Factor traversal visualization saved")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not generate some disentanglement visualizations: {str(e)}")

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
        if 'dann' in trained_models:
            dann_model = trained_models['dann']
            print("🎨 Generating DANN latent space visualization...")
            dann_latent_path = os.path.join(latent_space_dir, 'dann_latent_spaces.png')
            dann_model.visualize_latent_space(test_loader, args.device, save_path=dann_latent_path)

    # =============================================================================
    # 3.5. TRAIN AND TEST AUGMENTED DANN MODEL
    # =============================================================================
    if 'dann_augmented' in args.models:
        print("\n" + "="*60)
        print("🔥 TRAINING AUGMENTED DANN MODEL")
        print("="*60)

        dann_aug_params = {
            **model_params,
            'lambda_reversal': getattr(args, 'lambda_reversal', 1.0),
            'sparsity_weight': getattr(args, 'sparsity_weight', 0.01),
            'beta_adv': getattr(args, 'beta_adv', 0.1)
        }

        with open(os.path.join(args.out, 'dann_augmented_model_params.json'), 'w') as f:
            json.dump(dann_aug_params, f)

        if not args.skip_training:
            # Train AugmentedDANN using the new training function
            dann_aug_model, dann_aug_metrics = train_dann_augmented(
                args, spec_data, train_loader, val_loader, dataset='wild'
            )
            trained_models['dann_augmented'] = dann_aug_model

            # Save AugmentedDANN model
            dann_aug_model_path = os.path.join(models_dir, f"dann_augmented_model_epoch_{dann_aug_metrics['best_model_epoch']}.pt")
            torch.save(dann_aug_model.state_dict(), dann_aug_model_path)
            print(f"AugmentedDANN model saved to: {dann_aug_model_path}")
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained AugmentedDANN model for visualization...")
            dann_aug_model, dann_aug_metrics = load_model_checkpoint(models_dir, 'dann_augmented', spec_data, args)
            if dann_aug_model is not None:
                trained_models['dann_augmented'] = dann_aug_model
            else:
                print("⚠️  Skipping AugmentedDANN visualization - no pre-trained model found")

        # Visualize AugmentedDANN latent spaces
        if 'dann_augmented' in trained_models:
            dann_aug_model = trained_models['dann_augmented']
            print("🎨 Generating AugmentedDANN latent space visualization...")
            dann_aug_latent_path = os.path.join(latent_space_dir, 'dann_augmented_latent_spaces.png')
            visualize_latent_spaces(dann_aug_model, test_loader, args.device, type='dann_augmented', save_path=dann_aug_latent_path)

            # Evaluate latent expressiveness
            print("🧪 Evaluating AugmentedDANN latent expressiveness...")
            dann_aug_expressiveness_dir = os.path.join(args.out, 'dann_augmented_expressiveness')
            os.makedirs(dann_aug_expressiveness_dir, exist_ok=True)
            from core.WILD.latent_expressiveness import evaluate_latent_expressiveness
            dann_aug_expressiveness = evaluate_latent_expressiveness(
                dann_aug_model, train_loader, val_loader, test_loader, args.device, dann_aug_expressiveness_dir
            )

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
        if 'irm' in trained_models:
            irm_model = trained_models['irm']
            print("🎨 Generating IRM latent space visualization...")
            irm_latent_path = os.path.join(latent_space_dir, 'irm_latent_spaces.png')
            # Always use generic visualization for WILD data (handles metadata format correctly)
            # IRM.visualize_latent_space() expects CRMNIST format (x,y,c,r) not WILD (x,y,metadata)
            from core.utils import balanced_sample_for_visualization
            features_dict, labels_dict, stats = balanced_sample_for_visualization(
                model=irm_model, dataloader=test_loader, device=args.device,
                model_type="irm", max_samples=3000, dataset_type="wild"
            )
            print(f"IRM features extracted: {features_dict['features'].shape}")
    
    # =============================================================================
    # 5. POST-TRAINING ANALYSIS (for NVAE/DIVA models only)
    # =============================================================================
    if ('nvae' in args.models and 'nvae' in trained_models) or ('diva' in args.models and 'diva' in trained_models):
        print("\n" + "="*60)
        print("🔬 RUNNING POST-TRAINING ANALYSIS")
        print("="*60)

        # Choose the model for analysis (prefer NVAE if available)
        analysis_model = trained_models.get('nvae') or trained_models.get('diva')
        if analysis_model:
            analysis_model.eval()
            
            # Prepare validation data for latent analysis
            print("  📊 Preparing validation data for latent analysis...")
            transform = transforms.Compose([transforms.ToTensor()])
            final_val_data = dataset.get_subset(args.val_type, transform=transform)
            val_loader_analysis = get_eval_loader("standard", final_val_data, batch_size=10)
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
    for model_name in trained_models.keys():
        latent_file = f"{model_name}_latent_spaces.png"
        print(f"  📈 {model_name.upper()} latent spaces: {os.path.join(latent_space_dir, latent_file)}")
    
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
        if 'dann_augmented' in args.models and 'dann_aug_metrics' in locals():
            metrics_summary['DANN_AUGMENTED'] = dann_aug_metrics
        if 'irm' in args.models and 'irm_metrics' in locals():
            metrics_summary['IRM'] = irm_metrics
        
        for model_name, metrics in metrics_summary.items():
            print(f"  {model_name}:")
            print(f"    Best epoch: {metrics.get('best_model_epoch', 'N/A')}")
            print(f"    Best val loss: {float(metrics.get('best_val_loss', metrics.get('best_val_accuracy', 'N/A'))):.4f}")
            print(f"    Epochs trained: {metrics.get('epochs_trained', metrics.get('total_epochs_trained', 'N/A'))}")
    else:
        print("  Training was skipped - no metrics available")
    
    # =============================================================================
    # COMPREHENSIVE EXPRESSIVENESS ANALYSIS
    # =============================================================================
    if not args.skip_training:
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EXPRESSIVENESS ANALYSIS")
        print("="*60)

        from core.WILD.compare_all_expressiveness import main as compare_expressiveness
        compare_expressiveness(args.out)

    # =============================================================================
    # 7. INFORMATION-THEORETIC EVALUATION
    # =============================================================================
    if not args.skip_training and len(trained_models) > 0:
        print("\n" + "="*60)
        print("🧮 INFORMATION-THEORETIC EVALUATION")
        print("Testing adherence to Minimal Information Partition theorem")
        print("="*60)

        try:
            from core.information_theoretic_evaluation import (
                MinimalInformationPartitionEvaluator,
                evaluate_model
            )
            from core.visualization.plot_information_theoretic import visualize_all

            # Create output directory for IT analysis
            it_output_dir = os.path.join(args.out, 'information_theoretic_analysis')
            os.makedirs(it_output_dir, exist_ok=True)

            # Evaluate each trained model
            it_results = {}

            # Only evaluate models with explicit latent decomposition
            # NVAE and DIVA use qz() with VAE-style latent spaces
            # AugmentedDANN uses extract_features() with 3-component latent spaces
            # Note: baseline DANN (key='dann') is excluded; only AugmentedDANN (key='dann_augmented') is supported
            LATENT_DECOMPOSITION_MODELS = ['nvae', 'diva', 'dann_augmented']

            for model_name, model in trained_models.items():
                # Filter to only models with latent decomposition
                if model_name not in LATENT_DECOMPOSITION_MODELS:
                    print(f"\n⚠️  Skipping {model_name} - IT evaluation requires latent decomposition")
                    print(f"    Information-theoretic evaluation only supports models with explicit")
                    print(f"    partitioned latent spaces.")
                    print(f"    Compatible models: {', '.join(LATENT_DECOMPOSITION_MODELS)}")
                    continue

                print(f"\n{'='*60}")
                print(f"Evaluating {model_name.upper()} on information-theoretic metrics")
                print(f"{'='*60}")

                try:
                    # Run IT evaluation with 200 batches (~12.8k samples with batch_size=64)
                    # and 100 bootstrap iterations for confidence intervals
                    results = evaluate_model(
                        model=model,
                        dataloader=val_loader,
                        device=args.device,
                        max_batches=200,
                        n_bootstrap=100
                    )

                    it_results[model_name.upper()] = results

                    # Save individual model results
                    model_it_dir = os.path.join(it_output_dir, model_name)
                    os.makedirs(model_it_dir, exist_ok=True)

                    evaluator = MinimalInformationPartitionEvaluator()
                    evaluator.save_results(
                        results,
                        os.path.join(model_it_dir, 'it_results.json')
                    )

                except Exception as e:
                    print(f"⚠️  Failed to evaluate {model_name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Compare models if we have results
            if len(it_results) > 1:
                print("\n" + "="*60)
                print("📊 COMPARING MODELS ON INFORMATION-THEORETIC METRICS")
                print("="*60)

                evaluator = MinimalInformationPartitionEvaluator()
                comparison = evaluator.compare_models(it_results)

                # Save comparison results
                evaluator.save_results(
                    comparison,
                    os.path.join(it_output_dir, 'model_comparison.json')
                )

                # Generate all visualizations
                visualize_all(comparison, it_output_dir)

                # Print final rankings
                print("\n" + "="*60)
                print("🏆 FINAL RANKINGS - Minimal Information Partition Adherence")
                print("="*60)

                sorted_models = sorted(
                    comparison['model_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for rank, (model, score) in enumerate(sorted_models, 1):
                    print(f"{rank}. {model:<20} Partition Quality: {score:.4f}")

                print("\n✅ Information-theoretic evaluation complete!")
                print(f"   Results saved to: {it_output_dir}")

            else:
                print("\n⚠️  Not enough models for IT comparison")

        except ImportError as e:
            print(f"\n⚠️  Skipping information-theoretic evaluation:")
            print(f"   {e}")
            print(f"   Install npeet: pip install git+https://github.com/gregversteeg/NPEET.git")
        except Exception as e:
            print(f"\n⚠️  Information-theoretic evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # =============================================================================
    # OOD DOMAIN GENERALIZATION EVALUATION
    # =============================================================================
    if ood_test_loader is not None:
        print("\n" + "="*80)
        print("🎯 OOD DOMAIN GENERALIZATION EVALUATION")
        print("="*80)
        print(f"\nEvaluating domain generalization for withheld hospital {ood_hospital}...")

        from core.test import test_with_ood
        import json

        # Create OOD evaluation directory
        ood_eval_dir = os.path.join(args.out, 'ood_evaluation')
        os.makedirs(ood_eval_dir, exist_ok=True)

        # Create OOD visualization directory
        ood_viz_dir = os.path.join(args.out, 'latent_space_ood')
        os.makedirs(ood_viz_dir, exist_ok=True)

        # Dictionary to store all OOD results
        ood_results_summary = {}

        for model_name, model in trained_models.items():
            print(f"\n{'='*80}")
            print(f"📊 {model_name.upper()} - Domain Generalization Results")
            print(f"{'='*80}")

            # Determine model type for test_with_ood
            if 'dann' in model_name.lower() and 'augmented' not in model_name.lower():
                model_type = 'dann'
            elif 'augmented' in model_name.lower():
                model_type = 'dann_augmented'
            elif 'irm' in model_name.lower():
                model_type = 'irm'
            else:
                model_type = model_name.lower()

            # Evaluate on ID and OOD test sets
            ood_results = test_with_ood(
                model,
                id_test_loader,
                ood_test_loader,
                dataset_type='wild',
                device=args.device,
                model_type=model_type
            )

            # Store results
            ood_results_summary[model_name] = ood_results

            # Save individual model results
            results_file = os.path.join(ood_eval_dir, f'{model_name}_ood_results.json')
            with open(results_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = {}
                for key, value in ood_results.items():
                    if isinstance(value, dict):
                        json_results[key] = {k: float(v) for k, v in value.items()}
                    elif value is not None:
                        json_results[key] = float(value)
                    else:
                        json_results[key] = None
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")

            # Generate OOD latent space visualizations (separate from ID visualizations)
            if model_type in ['nvae', 'diva'] and not args.skip_training:
                print(f"\n📊 Generating OOD latent space visualization for {model_name}...")
                ood_viz_path = os.path.join(ood_viz_dir, f'{model_name}_latent_spaces_ood.png')
                try:
                    from core.utils import visualize_latent_spaces
                    visualize_latent_spaces(
                        model=model,
                        dataloader=ood_test_loader,
                        device=args.device,
                        type='wild',
                        save_path=ood_viz_path,
                        max_samples=750
                    )
                    print(f"   ✅ OOD visualization saved to: {ood_viz_path}")
                except Exception as e:
                    print(f"   ⚠️  OOD visualization failed: {e}")

        # Print summary comparison
        print("\n" + "="*80)
        print("📊 OOD GENERALIZATION SUMMARY - ALL MODELS")
        print("="*80)
        print(f"\n{'Model':<20} {'ID Acc':<12} {'OOD Acc':<12} {'Gap':<12} {'ID Loss':<12} {'OOD Loss':<12}")
        print("-" * 80)

        for model_name, results in ood_results_summary.items():
            id_acc = results['id_metrics'].get('y_accuracy', 0.0)
            ood_acc = results['ood_metrics'].get('y_accuracy', 0.0) if results['ood_metrics'] else 0.0
            gap = results['generalization_gap'].get('y_accuracy', 0.0) if results['generalization_gap'] else 0.0
            id_loss = results['id_loss']
            ood_loss = results['ood_loss'] if results['ood_loss'] else 0.0

            print(f"{model_name:<20} {id_acc:<12.4f} {ood_acc:<12.4f} {gap:<12.4f} {id_loss:<12.4f} {ood_loss:<12.4f}")

        # Save summary results
        summary_file = os.path.join(ood_eval_dir, 'ood_summary.json')
        with open(summary_file, 'w') as f:
            summary_data = {}
            for model_name, results in ood_results_summary.items():
                summary_data[model_name] = {
                    'id_accuracy': float(results['id_metrics'].get('y_accuracy', 0.0)),
                    'ood_accuracy': float(results['ood_metrics'].get('y_accuracy', 0.0)) if results['ood_metrics'] else 0.0,
                    'accuracy_gap': float(results['generalization_gap'].get('y_accuracy', 0.0)) if results['generalization_gap'] else 0.0,
                    'id_loss': float(results['id_loss']),
                    'ood_loss': float(results['ood_loss']) if results['ood_loss'] else 0.0
                }
            json.dump(summary_data, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_file}")

        print("\n" + "="*80)
        print("✅ OOD DOMAIN GENERALIZATION EVALUATION COMPLETE!")
        print(f"   Results directory: {ood_eval_dir}")
        print(f"   OOD visualizations: {ood_viz_dir}")
        print("="*80)

    print(f"\n🎉 All experiments completed!")
    print(f"📁 All results saved to: {args.out}")
    print(f"💾 Dataset cached at: {args.data_dir}")

    # Store trained models for potential future use
    print(f"\nTrained models available in memory: {list(trained_models.keys())}")
    print("=" * 50)




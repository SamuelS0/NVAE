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
import glob
import core.CRMNIST.utils_crmnist
from core.CRMNIST.data_generation import generate_crmnist_dataset
from core.CRMNIST.model import VAE
from core.comparison.train import train_nvae, train_diva, train_dann, train_irm
from core.CRMNIST.latent_expressiveness import evaluate_latent_expressiveness
from core.CRMNIST.disentanglement_visualization import (
    visualize_disentanglement,
    visualize_latent_interpolation,
    visualize_factor_traversal
)
from core.comparison.dann import DANN
from core.comparison.irm import IRM
from core.CRMNIST.dann_model import AugmentedDANN
from core.CRMNIST.dann_trainer import DANNTrainer
from core.utils import visualize_latent_spaces, get_parser
"""
CRMNIST model training script.

This script trains various models (NVAE, DIVA, DANN) on the CRMNIST dataset,
a custom version of MNIST with color and rotation transformations.

Run with:
python -m core.CRMNIST.run_crmnist --out results/ --config conf/crmnist.json --model_type nvae
python -m core.CRMNIST.run_crmnist --out results/ --config conf/crmnist.json --model_type dann
"""

def load_model_checkpoint(models_dir, model_name, spec_data, args):
    """
    Load a pre-trained model checkpoint.
    
    Args:
        models_dir: Directory containing model checkpoints
        model_name: Name of the model ('nvae', 'diva', 'dann', 'irm')
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
                class_map=spec_data['class_map'],
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
                diva=False,
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zay=getattr(args, 'l1_lambda_zay', 0.0),
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0)
            )
            
        elif model_name == 'diva':
            model = VAE(
                class_map=spec_data['class_map'],
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
                diva=True,
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zay=getattr(args, 'l1_lambda_zay', 0.0),
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0)
            )
            
        elif model_name == 'dann':
            z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim
            model = DANN(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'crmnist')

        elif model_name == 'dann_augmented':
            # Redistribute dimensions for AugmentedDANN to match total of other models
            total_dim = args.zy_dim + args.zx_dim + args.zay_dim + args.za_dim
            zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)
            zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)
            zdy_aug = total_dim // 3

            model = AugmentedDANN(
                class_map=spec_data['class_map'],
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
                image_size=28
            )

        elif model_name == 'irm':
            z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim
            model = IRM(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'crmnist',
                       penalty_weight=args.irm_penalty_weight, penalty_anneal_iters=args.irm_anneal_iters)
        
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
        return None, None



if __name__ == "__main__":
    parser = get_parser('CRMNIST')
    parser.add_argument('--intensity', '-i', type=float, default=1.5)
    parser.add_argument('--intensity_decay', '-d', type=float, default=1.0)
    parser.add_argument('--config', type=str, default = 'conf/crmnist.json')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--zy_dim', type=int, default=32)
    parser.add_argument('--zx_dim', type=int, default=32)
    parser.add_argument('--zay_dim', type=int, default=32)
    parser.add_argument('--za_dim', type=int, default=32)
    parser.add_argument('--beta_1', type=float, default=1.0)
    parser.add_argument('--beta_2', type=float, default=1.0)
    parser.add_argument('--beta_3', type=float, default=1.0)
    parser.add_argument('--beta_4', type=float, default=1.0)
    parser.add_argument('--alpha_1', type=float, default=100.0)
    parser.add_argument('--alpha_2', type=float, default=100.0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--dataset', type=str, default='crmnist')
    parser.add_argument('--use_cache', action='store_true', default=True, 
                       help='Use cached datasets if available (default: True)')
    parser.add_argument('--no_cache', action='store_true', default=False,
                       help='Disable dataset caching and force regeneration')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--beta_annealing', action='store_true', help='Enable beta annealing for KL divergence')
    parser.add_argument('--beta_scale', type=float, default=1.0)

    # L1 sparsity penalty arguments
    parser.add_argument('--l1_lambda_zy', type=float, default=0.0,
                       help='L1 penalty weight for zy latent (default: 0.0)')
    parser.add_argument('--l1_lambda_zx', type=float, default=0.01,
                       help='L1 penalty weight for zx latent (default: 0.01)')
    parser.add_argument('--l1_lambda_zay', type=float, default=0.01,
                       help='L1 penalty weight for zay latent (default: 0.01)')
    parser.add_argument('--l1_lambda_za', type=float, default=0.0,
                       help='L1 penalty weight for za latent (default: 0.0)')

    # Model selection arguments
    parser.add_argument('--models', type=str, nargs='+', default=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                       choices=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                       help='Which models to train and test (default: all 5 models)')
    parser.add_argument('--skip_training', action='store_true', default=False,
                       help='Skip training and only do visualization (requires pre-trained models)')
    parser.add_argument('--setting', type=str, default='standard',
                       help='Experimental setting name for organizing outputs (default: standard)')

    # AugmentedDANN-specific parameters
    parser.add_argument('--lambda_reversal', type=float, default=1.0,
                       help='Lambda parameter for gradient reversal in AugmentedDANN (default: 1.0)')
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                       help='Weight for sparsity penalty on zdy in AugmentedDANN (default: 0.01)')
    parser.add_argument('--beta_adv', type=float, default=0.1,
                       help='Weight for adversarial loss in AugmentedDANN (default: 0.1)')

    # IRM-specific parameters
    parser.add_argument('--irm_penalty_weight', type=float, default=1e4,
                       help='Weight for IRM invariance penalty (default: 1e4)')
    parser.add_argument('--irm_anneal_iters', type=int, default=500,
                       help='Number of iterations before applying IRM penalty (default: 500)')

    args = parser.parse_args()
    
    # Set cache behavior based on arguments
    use_cache = args.use_cache and not args.no_cache
    
    # Set up CUDA if available
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.cuda:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Run experiment
    #model = run_experiment(args)



    #run comparison experiments
    print("\n🎯 Running comparison experiments...")
    
    os.makedirs(args.out, exist_ok=True)
    reconstructions_dir = os.path.join(args.out, 'reconstructions')
    os.makedirs(reconstructions_dir, exist_ok=True)
    models_dir = os.path.join(args.out, 'models')
    os.makedirs(models_dir, exist_ok=True)
    domain_samples_dir = os.path.join(args.out, 'domain_samples')
    os.makedirs(domain_samples_dir, exist_ok=True)
    latent_space_dir = os.path.join(args.out, 'latent_space')
    os.makedirs(latent_space_dir, exist_ok=True)
    
    # Log some information
    print(f"Starting CRMNIST model comparison...")
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
    spec_data['y_c'], subsets = core.CRMNIST.utils_crmnist.choose_label_subset(spec_data)
    # Update domain_data with subsets
    for i, subset in subsets.items():
        if i in domain_data:
            domain_data[i]['subset'] = subset
            
    # Generate dataset (with caching)
    print("Loading/generating datasets for comparison experiments...")
    train_dataset = generate_crmnist_dataset(spec_data, train=True,
                                            transform_intensity=args.intensity,
                                            transform_decay=args.intensity_decay,
                                            use_cache=use_cache)
    test_dataset = generate_crmnist_dataset(spec_data, train=False,
                                           transform_intensity=args.intensity,
                                           transform_decay=args.intensity_decay,
                                           use_cache=use_cache)
    
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
            nvae_model, nvae_metrics = train_nvae(args, spec_data, train_loader, val_loader, dataset='crmnist')
            trained_models['nvae'] = nvae_model
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained NVAE model for visualization...")
            nvae_model, nvae_metrics = load_model_checkpoint(models_dir, 'nvae', spec_data, args)
            if nvae_model is not None:
                trained_models['nvae'] = nvae_model
            else:
                print("⚠️  Skipping NVAE visualization - no pre-trained model found")

        # Visualize NVAE latent spaces
        if 'nvae' in trained_models:
            nvae_model = trained_models['nvae']
            print("🎨 Generating NVAE latent space visualization...")
            nvae_latent_path = os.path.join(latent_space_dir, 'nvae_latent_spaces.png')
            visualize_latent_spaces(nvae_model, val_loader, args.device, type='crmnist', save_path=nvae_latent_path)
            
            # Evaluate latent expressiveness for NVAE
            print("🧪 Evaluating NVAE latent expressiveness...")
            nvae_expressiveness_dir = os.path.join(args.out, 'nvae_expressiveness')
            os.makedirs(nvae_expressiveness_dir, exist_ok=True)
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
            diva_model, diva_metrics = train_diva(args, spec_data, train_loader, val_loader, dataset='crmnist')
            trained_models['diva'] = diva_model
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained DIVA model for visualization...")
            diva_model, diva_metrics = load_model_checkpoint(models_dir, 'diva', spec_data, args)
            if diva_model is not None:
                trained_models['diva'] = diva_model
            else:
                print("⚠️  Skipping DIVA visualization - no pre-trained model found")

        # Visualize DIVA latent spaces
        if 'diva' in trained_models:
            diva_model = trained_models['diva']
            print("🎨 Generating DIVA latent space visualization...")
            diva_latent_path = os.path.join(latent_space_dir, 'diva_latent_spaces.png')
            visualize_latent_spaces(diva_model, val_loader, args.device, type='crmnist', save_path=diva_latent_path)
            
            # Evaluate latent expressiveness for DIVA
            print("🧪 Evaluating DIVA latent expressiveness...")
            diva_expressiveness_dir = os.path.join(args.out, 'diva_expressiveness')
            os.makedirs(diva_expressiveness_dir, exist_ok=True)
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
            dann_model, dann_metrics = train_dann(args, spec_data, train_loader, val_loader, dataset='crmnist')
            trained_models['dann'] = dann_model
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained DANN model for visualization...")
            dann_model, dann_metrics = load_model_checkpoint(models_dir, 'dann', spec_data, args)
            if dann_model is not None:
                trained_models['dann'] = dann_model
            else:
                print("⚠️  Skipping DANN visualization - no pre-trained model found")

        # Visualize DANN latent space
        if 'dann' in trained_models:
            dann_model = trained_models['dann']
            print("🎨 Generating DANN latent space visualization...")
            dann_latent_path = os.path.join(latent_space_dir, 'dann_latent_spaces.png')
            dann_model.visualize_latent_space(val_loader, args.device, save_path=dann_latent_path)

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
            # Redistribute dimensions for AugmentedDANN to match total of other models
            # AugmentedDANN uses 3 subspaces (zy, zd, zdy) while others use 4
            # To maintain fair comparison, redistribute total dimension across 3 subspaces
            total_dim = args.zy_dim + args.zx_dim + args.zay_dim + args.za_dim  # Total: 128
            zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)  # 43
            zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)   # 43
            zdy_aug = total_dim // 3  # 42

            print(f"📏 AugmentedDANN dimension redistribution: zy={zy_aug}, zd={zd_aug}, zdy={zdy_aug} (total={zy_aug+zd_aug+zdy_aug})")

            # Create AugmentedDANN model
            dann_aug_model = AugmentedDANN(
                class_map=spec_data['class_map'],
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
                image_size=28
            ).to(args.device)

            # Create optimizer
            optimizer = optim.Adam(dann_aug_model.parameters(), lr=args.learning_rate)

            # Train using DANNTrainer
            trainer = DANNTrainer(dann_aug_model, optimizer, args.device, args, patience=5)
            trainer.train(train_loader, val_loader, args.epochs)

            # Get metrics
            dann_aug_metrics = {
                'best_model_state': trainer.best_model_state,
                'best_val_loss': trainer.best_val_loss,
                'epochs_trained': trainer.epochs_trained,
                'best_model_epoch': trainer.best_epoch
            }

            # Load best model
            dann_aug_model.load_state_dict(trainer.best_model_state)
            trained_models['dann_augmented'] = dann_aug_model
        else:
            # Load pre-trained model
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
            visualize_latent_spaces(dann_aug_model, val_loader, args.device, type='dann_augmented', save_path=dann_aug_latent_path)

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
            irm_model, irm_metrics = train_irm(args, spec_data, train_loader, val_loader, dataset='crmnist')
            trained_models['irm'] = irm_model
        else:
            # Load pre-trained model for visualization
            print("Loading pre-trained IRM model for visualization...")
            irm_model, irm_metrics = load_model_checkpoint(models_dir, 'irm', spec_data, args)
            if irm_model is not None:
                trained_models['irm'] = irm_model
            else:
                print("⚠️  Skipping IRM visualization - no pre-trained model found")

        # Visualize IRM latent space
        if 'irm' in trained_models:
            irm_model = trained_models['irm']
            print("🎨 Generating IRM latent space visualization...")
            irm_latent_path = os.path.join(latent_space_dir, 'irm_latent_spaces.png')
            if hasattr(irm_model, 'visualize_latent_space'):
                irm_model.visualize_latent_space(val_loader, args.device, save_path=irm_latent_path)
            else:
                # Use the generic visualization function
                from core.utils import balanced_sample_for_visualization
                features_dict, labels_dict, stats = balanced_sample_for_visualization(
                    model=irm_model, dataloader=val_loader, device=args.device, 
                    model_type="irm", max_samples=3000
                )
                print(f"IRM features extracted: {features_dict['features'].shape}")
    
    # =============================================================================
    # 5. SUMMARY AND COMPARISON
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
    
    print(f"\nModel files saved to: {models_dir}")

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
            print(f"    Best val accuracy: {metrics.get('best_val_accuracy', 0.0):.4f}")
            print(f"    Epochs trained: {metrics.get('epochs_trained', 'N/A')}")
    else:
        print("  Training was skipped - no metrics available")
    
    print(f"\n🎉 All experiments completed!")
    print(f"Results saved to: {args.out}")
    
    # Store trained models for potential future use
    print(f"\nTrained models available in memory: {list(trained_models.keys())}")
    
    # =============================================================================
    # 6. COMPREHENSIVE EXPRESSIVENESS COMPARISON
    # =============================================================================
    if not args.skip_training:
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EXPRESSIVENESS ANALYSIS")
        print("="*60)

        # Import and run comprehensive comparison
        from core.CRMNIST.compare_all_expressiveness import main as compare_expressiveness
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
                    # Run IT evaluation with optimized parameters for faster computation
                    # 50 batches (~3.2k samples) and 10 bootstrap iterations
                    # This provides 40x speedup while maintaining statistical validity
                    results = evaluate_model(
                        model=model,
                        dataloader=val_loader,
                        device=args.device,
                        max_batches=50,      # Reduced from 200 (4x faster)
                        n_bootstrap=10       # Reduced from 100 (10x faster)
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

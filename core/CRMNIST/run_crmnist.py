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
from core.comparison.dann import DANN
from core.comparison.irm import IRM
from core.CRMNIST.dann_model import AugmentedDANN
from core.CRMNIST.dann_trainer import DANNTrainer
from core.utils import visualize_latent_spaces
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
            model = AugmentedDANN(
                class_map=spec_data['class_map'],
                zy_dim=args.zy_dim,
                zd_dim=args.za_dim,  # Use za_dim for domain-specific features
                zdy_dim=args.zay_dim,  # Use zay_dim for domain-class interaction
                y_dim=spec_data['num_y_classes'],
                d_dim=spec_data['num_r_classes'],
                lambda_reversal=getattr(args, 'lambda_reversal', 1.0),
                sparsity_weight=getattr(args, 'sparsity_weight', 0.01),
                alpha_y=args.alpha_1,
                alpha_d=args.alpha_2,
                beta_adv=getattr(args, 'beta_adv', 0.1)
            )

        elif model_name == 'irm':
            z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim
            model = IRM(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'crmnist', 
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
        return None, None



if __name__ == "__main__":
    parser = core.utils.get_parser('CRMNIST')
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
                       help='L1 penalty weight for zy latent (default: 0.0 = disabled)')
    parser.add_argument('--l1_lambda_zx', type=float, default=0.0,
                       help='L1 penalty weight for zx latent (default: 0.0 = disabled)')
    parser.add_argument('--l1_lambda_zay', type=float, default=0.0,
                       help='L1 penalty weight for zay latent (default: 0.0 = disabled)')
    parser.add_argument('--l1_lambda_za', type=float, default=0.0,
                       help='L1 penalty weight for za latent (default: 0.0 = disabled)')

    # Model selection arguments
    parser.add_argument('--models', type=str, nargs='+', default=['nvae', 'diva', 'dann', 'irm'],
                       choices=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                       help='Which models to train and test (default: all)')
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
            
            # Save NVAE model
            nvae_model_path = os.path.join(models_dir, f"nvae_model_epoch_{nvae_metrics['best_model_epoch']}.pt")
            torch.save(nvae_model.state_dict(), nvae_model_path)
            print(f"NVAE model saved to: {nvae_model_path}")
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
            
            # Save DIVA model
            diva_model_path = os.path.join(models_dir, f"diva_model_epoch_{diva_metrics['best_model_epoch']}.pt")
            torch.save(diva_model.state_dict(), diva_model_path)
            print(f"DIVA model saved to: {diva_model_path}")
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
            dann_model, dann_metrics = train_dann(args, spec_data, train_loader, test_loader, dataset='crmnist')
            trained_models['dann'] = dann_model
            
            # Save DANN model
            dann_model_path = os.path.join(models_dir, f"dann_model_epoch_{dann_metrics['best_model_epoch']}.pt")
            torch.save(dann_model.state_dict(), dann_model_path)
            print(f"DANN model saved to: {dann_model_path}")
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
            # Create AugmentedDANN model
            dann_aug_model = AugmentedDANN(
                class_map=spec_data['class_map'],
                zy_dim=args.zy_dim,
                zd_dim=args.za_dim,
                zdy_dim=args.zay_dim,
                y_dim=spec_data['num_y_classes'],
                d_dim=spec_data['num_r_classes'],
                lambda_reversal=getattr(args, 'lambda_reversal', 1.0),
                sparsity_weight=getattr(args, 'sparsity_weight', 0.01),
                alpha_y=args.alpha_1,
                alpha_d=args.alpha_2,
                beta_adv=getattr(args, 'beta_adv', 0.1)
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

            # Save model
            dann_aug_model_path = os.path.join(models_dir, f"dann_augmented_model_epoch_{dann_aug_metrics['best_model_epoch']}.pt")
            torch.save(dann_aug_model.state_dict(), dann_aug_model_path)
            print(f"AugmentedDANN model saved to: {dann_aug_model_path}")
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
            irm_model, irm_metrics = train_irm(args, spec_data, train_loader, test_loader, dataset='crmnist')
            trained_models['irm'] = irm_model
            
            # Save IRM model
            irm_model_path = os.path.join(models_dir, f"irm_model_epoch_{irm_metrics['best_model_epoch']}.pt")
            torch.save(irm_model.state_dict(), irm_model_path)
            print(f"IRM model saved to: {irm_model_path}")
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
            print(f"    Best val accuracy: {float(metrics.get('best_val_accuracy', 'N/A')):.4f}")
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

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
                zdy_dim=args.zdy_dim,
                zd_dim=args.zd_dim,
                y_dim=spec_data['num_y_classes'],
                a_dim=spec_data['num_r_classes'],
                beta_zy=args.beta_zy,
                beta_zx=args.beta_zx,
                beta_zdy=args.beta_zdy,
                beta_zd=args.beta_zd,
                alpha_y=args.alpha_y,
                alpha_d=args.alpha_d,
                diva=False,
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zdy=getattr(args, 'l1_lambda_zdy', 0.0),
                l1_lambda_zd=getattr(args, 'l1_lambda_zd', 0.0)
            )
            
        elif model_name == 'diva':
            model = VAE(
                class_map=spec_data['class_map'],
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zdy_dim=args.zdy_dim,
                zd_dim=args.zd_dim,
                y_dim=spec_data['num_y_classes'],
                a_dim=spec_data['num_r_classes'],
                beta_zy=args.beta_zy,
                beta_zx=args.beta_zx,
                beta_zdy=args.beta_zdy,
                beta_zd=args.beta_zd,
                alpha_y=args.alpha_y,
                alpha_d=args.alpha_d,
                diva=True,
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zdy=getattr(args, 'l1_lambda_zdy', 0.0),
                l1_lambda_zd=getattr(args, 'l1_lambda_zd', 0.0)
            )
            
        elif model_name == 'dann':
            # Use z_dim=16 to match NVAE/DIVA classification space (z_y + z_dy = 8 + 8 = 16)
            z_dim = 16
            model = DANN(
                z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], 'crmnist',
                domain_weight=getattr(args, 'dann_domain_weight', 1.0),
                lambda_gamma=getattr(args, 'dann_lambda_gamma', 10.0)
            )

        elif model_name == 'dann_augmented':
            # Redistribute dimensions for AugmentedDANN to match total of other models
            total_dim = args.zy_dim + args.zx_dim + args.zdy_dim + args.zd_dim
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
                sparsity_weight_zdy=getattr(args, 'sparsity_weight_zdy', 2.0),
                sparsity_weight_zy_zd=getattr(args, 'sparsity_weight_zy_zd', 0.5),
                alpha_y=args.alpha_y,
                alpha_d=args.alpha_d,
                beta_adv=getattr(args, 'beta_adv', 0.2),
                image_size=28,
                use_conditional_adversarial=getattr(args, 'use_conditional_adversarial', True),
                lambda_schedule_gamma=getattr(args, 'lambda_schedule_gamma', 5.0)
            )

        elif model_name == 'irm':
            # Use z_dim=16 to match NVAE/DIVA classification space (z_y + z_dy = 8 + 8 = 16)
            z_dim = 16
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
    parser.add_argument('--rotation_step', type=int, default=15,
                       help='Rotation step between domains in degrees (default: 15). Domains will rotate by 0, step, 2*step, ...')
    parser.add_argument('--config', type=str, default = 'conf/crmnist.json')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--zy_dim', type=int, default=8)
    parser.add_argument('--zx_dim', type=int, default=8)
    parser.add_argument('--zdy_dim', type=int, default=8)
    parser.add_argument('--zd_dim', type=int, default=8)
    # KL weights - DEFAULT: low (1.0) based on grid search showing kl-low best for VAEs
    parser.add_argument('--beta_zy', type=float, default=1.0)
    parser.add_argument('--beta_zx', type=float, default=1.0)
    parser.add_argument('--beta_zdy', type=float, default=1.0)
    parser.add_argument('--beta_zd', type=float, default=1.0)
    # Classifier weights - DEFAULT: high (150.0) based on grid search showing classifier-high best
    parser.add_argument('--alpha_y', type=float, default=150.0)
    parser.add_argument('--alpha_d', type=float, default=150.0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--dataset', type=str, default='crmnist')
    parser.add_argument('--use_cache', action='store_true', default=True, 
                       help='Use cached datasets if available (default: True)')
    parser.add_argument('--no_cache', action='store_true', default=False,
                       help='Disable dataset caching and force regeneration')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--beta_annealing', action='store_true', help='Enable beta annealing for KL divergence')
    parser.add_argument('--beta_scale', type=float, default=1.0)

    # L1 sparsity penalty arguments for VAE models (NVAE/DIVA)
    # Based on 20-epoch IT analysis: light L1 on z_dy reduces I(z_y;z_dy) entanglement
    # NVAE showed I(z_y;z_dy)=1.98 with zero L1 - z_dy was duplicating class info
    parser.add_argument('--l1_lambda_zy', type=float, default=0.0,
                       help='L1 penalty weight for zy latent (default: 0.0)')
    parser.add_argument('--l1_lambda_zx', type=float, default=0.0,
                       help='L1 penalty weight for zx latent (default: 0.0)')
    parser.add_argument('--l1_lambda_zdy', type=float, default=5.0,
                       help='L1 penalty weight for zdy latent (default: 5.0, reduces z_y/z_dy entanglement)')
    parser.add_argument('--l1_lambda_zd', type=float, default=0.0,
                       help='L1 penalty weight for zd latent (default: 0.0)')

    # Model selection arguments
    parser.add_argument('--models', type=str, nargs='+', default=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                       choices=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
                       help='Which models to train and test (default: all 5 models)')
    parser.add_argument('--skip_training', action='store_true', default=False,
                       help='Skip training and only do visualization (requires pre-trained models)')
    parser.add_argument('--setting', type=str, default='standard',
                       help='Experimental setting name for organizing outputs (default: standard)')

    # AugmentedDANN-specific parameters
    # DEFAULT values based on 20-epoch IT analysis: balanced sparsity to prevent dimension collapse
    # The domain_focus preset (zd=8.0, others=0) caused z_d to collapse to 1 dimension!
    # New balanced preset: moderate z_d sparsity with light constraints on z_y and z_dy
    parser.add_argument('--lambda_reversal', type=float, default=1.0,
                       help='Lambda parameter for gradient reversal in AugmentedDANN (default: 1.0)')
    parser.add_argument('--sparsity_weight', type=float, default=1.5,
                       help='Target weight for sparsity penalty on zdy in AugmentedDANN (default: 1.5, balanced preset)')
    parser.add_argument('--sparsity_weight_other', type=float, default=0.5,
                       help='Target weight for sparsity penalty on zy in AugmentedDANN (default: 0.5, balanced preset)')
    parser.add_argument('--sparsity_weight_zd', type=float, default=2.5,
                       help='Target weight for sparsity penalty on zd in AugmentedDANN (default: 2.5, balanced preset)')
    parser.add_argument('--beta_adv', type=float, default=1.0,
                       help='Weight for adversarial loss in AugmentedDANN (default: 1.0, high preset)')
    parser.add_argument('--use_conditional_adversarial', action='store_true', default=True,
                       help='Use conditional adversarial for I(Z_Y;D|Y)=0 and I(Z_D;Y|D)=0 (default: True)')
    parser.add_argument('--no_conditional_adversarial', dest='use_conditional_adversarial',
                       action='store_false',
                       help='Disable conditional adversarial (use unconditional)')
    parser.add_argument('--lambda_schedule_gamma', type=float, default=10.0,
                       help='Controls adversarial ramp-up speed: 10=fast (high preset), 5=moderate, 2=slow (default: 10.0)')

    # IRM-specific parameters
    # Note: With environment-averaged loss (fixed), penalty_weight=5 is appropriate
    # Higher values (100+) can cause model collapse to uniform predictions
    parser.add_argument('--irm_penalty_weight', type=float, default=5.0,
                       help='Weight for IRM invariance penalty (default: 5.0)')
    parser.add_argument('--irm_anneal_iters', type=int, default=1000,
                       help='Number of iterations before applying IRM penalty (default: 1000)')

    # OOD Domain Generalization arguments
    parser.add_argument('--no_ood', action='store_true', default=False,
                       help='Disable OOD mode and include all domains in training (default: OOD enabled, last domain withheld)')
    parser.add_argument('--ood_domain_idx', type=int, default=None,
                       help='Specific domain index to withhold for OOD testing (0-5). Default: 5 (last domain) when OOD enabled')

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

    # Override rotation values based on rotation_step argument
    # This allows configuring rotations like 0, 10, 20, 30, 40, 50 (step=10) instead of 0, 15, 30, 45, 60, 75 (step=15)
    print(f"Rotation step: {args.rotation_step}°")
    for domain_idx in domain_data:
        old_rotation = domain_data[domain_idx].get('rotation', domain_idx * 15)
        new_rotation = domain_idx * args.rotation_step
        domain_data[domain_idx]['rotation'] = new_rotation
        if old_rotation != new_rotation:
            print(f"  Domain {domain_idx}: rotation {old_rotation}° -> {new_rotation}°")
    
    # Choose labels subset if not already chosen
    # Use a fixed seed for reproducibility (can be overridden via args.label_seed if added)
    label_seed = getattr(args, 'label_seed', 42)
    spec_data['y_c'], subsets = core.CRMNIST.utils_crmnist.choose_label_subset(spec_data, seed=label_seed)
    # Update domain_data with subsets
    for i, subset in subsets.items():
        if i in domain_data:
            domain_data[i]['subset'] = subset
            
    # =============================================================================
    # OOD DOMAIN GENERALIZATION SETUP
    # =============================================================================
    # Determine which domain to withhold for OOD testing
    ood_domain = None
    if not args.no_ood:
        # Default: withhold last domain (domain 5)
        ood_domain = args.ood_domain_idx if args.ood_domain_idx is not None else 5
        print(f"\n{'='*80}")
        print(f"🎯 OOD MODE ENABLED: Withholding domain {ood_domain} for OOD testing")
        print(f"   Training on domains: {[i for i in range(6) if i != ood_domain]}")
        print(f"   OOD test domain: {ood_domain}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"📊 STANDARD MODE: Using all 6 domains for training")
        print(f"{'='*80}\n")

    # Generate dataset (with caching)
    print("Loading/generating datasets for comparison experiments...")

    if ood_domain is not None:
        # OOD mode: Generate ID training data (exclude OOD domain)
        # Generate train dataset using base_split='train' to prevent data leakage
        # This ensures validation images are completely different base MNIST images
        train_dataset = generate_crmnist_dataset(
            spec_data, train=True,
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache,
            exclude_domains=[ood_domain],
            base_split='train',  # Use only 80% of base MNIST images for training
            base_split_ratio=0.8,
            base_split_seed=42
        )

        # Generate validation dataset from DIFFERENT base images (no leakage!)
        val_dataset = generate_crmnist_dataset(
            spec_data, train=True,  # Use MNIST training set, but different base images
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache,
            exclude_domains=[ood_domain],
            base_split='val',  # Use remaining 20% of base MNIST images for validation
            base_split_ratio=0.8,
            base_split_seed=42
        )

        # Generate full test dataset
        full_test_dataset = generate_crmnist_dataset(
            spec_data, train=False,
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache
        )

        # Import data utilities for filtering
        from core.data_utils import create_ood_split

        # Split test dataset into ID and OOD portions
        id_test_dataset, ood_test_dataset = create_ood_split(
            full_test_dataset,
            ood_domain,
            dataset_type='crmnist'
        )

        print(f"\n📊 Dataset sizes (OOD mode, NO DATA LEAKAGE):")
        print(f"   Training (ID only): {len(train_dataset)} samples (from unique base images)")
        print(f"   Validation (ID only): {len(val_dataset)} samples (from DIFFERENT base images)")
        print(f"   ID Test: {len(id_test_dataset)} samples")
        print(f"   OOD Test: {len(ood_test_dataset)} samples")
    else:
        # Standard mode: Use all domains
        # Generate train dataset using base_split='train' to prevent data leakage
        train_dataset = generate_crmnist_dataset(
            spec_data, train=True,
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache,
            base_split='train',  # Use only 80% of base MNIST images for training
            base_split_ratio=0.8,
            base_split_seed=42
        )

        # Generate validation dataset from DIFFERENT base images (no leakage!)
        val_dataset = generate_crmnist_dataset(
            spec_data, train=True,  # Use MNIST training set, but different base images
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache,
            base_split='val',  # Use remaining 20% of base MNIST images for validation
            base_split_ratio=0.8,
            base_split_seed=42
        )

        test_dataset = generate_crmnist_dataset(
            spec_data, train=False,
            transform_intensity=args.intensity,
            transform_decay=args.intensity_decay,
            use_cache=use_cache
        )
        id_test_dataset = test_dataset
        ood_test_dataset = None

        print(f"\n📊 Dataset sizes (NO DATA LEAKAGE):")
        print(f"   Training: {len(train_dataset)} samples (from unique base images)")
        print(f"   Validation: {len(val_dataset)} samples (from DIFFERENT base images)")
        print(f"   Test: {len(test_dataset)} samples")

    # NOTE: We no longer use random_split() here because that causes data leakage!
    # Instead, train_dataset and val_dataset are generated from completely different
    # base MNIST images using the base_split parameter above.

    # Create data loaders (train_dataset and val_dataset are already properly separated)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=args.batch_size, shuffle=False) if ood_test_dataset else None

    # For backward compatibility, keep test_loader pointing to id_test_loader
    test_loader = id_test_loader

    # Verify OOD domain is excluded from ID test data (in OOD mode)
    if ood_domain is not None:
        id_test_domains = set()
        for batch in id_test_loader:
            r_labels = batch[3]  # rotation/domain labels (one-hot)
            domains = r_labels.argmax(dim=1).tolist()
            id_test_domains.update(domains)
        assert ood_domain not in id_test_domains, f"OOD domain {ood_domain} found in ID test data!"
        print(f"   Verified: ID test data contains only domains {sorted(id_test_domains)} (OOD domain {ood_domain} excluded)")

    # Get dataset dimensions
    num_y_classes = spec_data['num_y_classes']
    num_r_classes = spec_data['num_r_classes']

    print(f"Dataset dimensions: y_dim={num_y_classes}, r_dim=num_domains={num_r_classes}")
    
    # Dictionary to store all trained models
    trained_models = {}

    # Base training configuration (shared across models)
    base_training_config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'dataset': args.dataset,
        'use_cache': use_cache,
        'seed': None,  # No seed argument defined
    }

    # VAE-specific base parameters (for NVAE/DIVA)
    vae_base_params = {
        'zy_dim': args.zy_dim,
        'zx_dim': args.zx_dim,
        'zdy_dim': args.zdy_dim,
        'zd_dim': args.zd_dim,
        'beta_zy': args.beta_zy,
        'beta_zx': args.beta_zx,
        'beta_zdy': args.beta_zdy,
        'beta_zd': args.beta_zd,
        'alpha_y': args.alpha_y,
        'alpha_d': args.alpha_d,
        'l1_lambda_zy': args.l1_lambda_zy,
        'l1_lambda_zx': args.l1_lambda_zx,
        'l1_lambda_zdy': args.l1_lambda_zdy,
        'l1_lambda_zd': args.l1_lambda_zd,
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

        nvae_params = {
            **vae_base_params,
            **base_training_config,
            'diva': False,
            'model_type': 'NVAE'
        }
        with open(os.path.join(args.out, 'nvae_model_params.json'), 'w') as f:
            json.dump(nvae_params, f, indent=2)
        
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
                    num_examples=3,
                    rotation_step=args.rotation_step
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

        diva_params = {
            **vae_base_params,
            **base_training_config,
            'diva': True,
            'model_type': 'DIVA'
        }
        with open(os.path.join(args.out, 'diva_model_params.json'), 'w') as f:
            json.dump(diva_params, f, indent=2)
        
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
                    num_examples=3,
                    rotation_step=args.rotation_step
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

        dann_params = {
            'z_dim': 16,  # Fixed z_dim=16 to match NVAE/DIVA classification space (z_y + z_dy)
            'num_y_classes': num_y_classes,
            'num_r_classes': num_r_classes,
            **base_training_config,
            'model_type': 'DANN'
        }
        with open(os.path.join(args.out, 'dann_model_params.json'), 'w') as f:
            json.dump(dann_params, f, indent=2)
        
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

        # Calculate AugmentedDANN dimensions first (needed for params)
        total_dim = args.zy_dim + args.zx_dim + args.zdy_dim + args.zd_dim  # Total: 32 (default)
        zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)  # 11 for 32 total
        zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)   # 11 for 32 total
        zdy_aug = total_dim // 3  # 10 for 32 total
        zx_aug = 0  # AugmentedDANN doesn't use zx explicitly

        print(f"📏 AugmentedDANN dimension redistribution: zy={zy_aug}, zd={zd_aug}, zdy={zdy_aug} (total={zy_aug+zd_aug+zdy_aug})")

        dann_aug_params = {
            # Architecture - use actual dimensions used by AugmentedDANN
            'zy_dim': zy_aug,
            'zd_dim': zd_aug,
            'zdy_dim': zdy_aug,
            'zx_dim': zx_aug,
            'num_y_classes': num_y_classes,
            'num_r_classes': num_r_classes,

            # AugmentedDANN-specific hyperparameters (CRITICAL!)
            'alpha_y': 1.0,  # Class prediction weight (from model default)
            'alpha_d': 1.0,  # Domain prediction weight (from model default)
            'beta_adv': getattr(args, 'beta_adv', 0.15),
            'lambda_reversal': getattr(args, 'lambda_reversal', 1.0),
            'sparsity_weight': getattr(args, 'sparsity_weight', 2.0),

            # Training config
            **base_training_config,
            'model_type': 'AugmentedDANN'
        }

        with open(os.path.join(args.out, 'dann_augmented_model_params.json'), 'w') as f:
            json.dump(dann_aug_params, f, indent=2)

        if not args.skip_training:

            # Create AugmentedDANN model
            # Default parameters based on 20-epoch IT analysis: balanced sparsity preset
            # Prevents dimension collapse seen with domain_focus (zd=8.0)
            dann_aug_model = AugmentedDANN(
                class_map=spec_data['class_map'],
                zy_dim=zy_aug,
                zd_dim=zd_aug,
                zdy_dim=zdy_aug,
                y_dim=spec_data['num_y_classes'],
                d_dim=spec_data['num_r_classes'],
                lambda_reversal=getattr(args, 'lambda_reversal', 1.0),
                sparsity_weight_zdy=getattr(args, 'sparsity_weight', 1.5),  # balanced: 1.5
                sparsity_weight_zy=getattr(args, 'sparsity_weight_other', 0.5),  # balanced: 0.5
                sparsity_weight_zd=getattr(args, 'sparsity_weight_zd', 2.5),  # balanced: 2.5
                alpha_y=args.alpha_y,
                alpha_d=args.alpha_d,
                beta_adv=getattr(args, 'beta_adv', 1.0),  # high: 1.0
                image_size=28,
                use_conditional_adversarial=getattr(args, 'use_conditional_adversarial', True),
                lambda_schedule_gamma=getattr(args, 'lambda_schedule_gamma', 10.0)  # high: 10.0
            ).to(args.device)

            # Create optimizer and LR scheduler
            optimizer = optim.Adam(dann_aug_model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 10% decay every 5 epochs

            # Train using DANNTrainer
            trainer = DANNTrainer(dann_aug_model, optimizer, args.device, args, patience=5, scheduler=scheduler)
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

            # Evaluate latent expressiveness for AugmentedDANN
            print("🧪 Evaluating AugmentedDANN latent expressiveness...")
            dann_aug_expressiveness_dir = os.path.join(args.out, 'dann_augmented_expressiveness')
            os.makedirs(dann_aug_expressiveness_dir, exist_ok=True)
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

        irm_params = {
            'z_dim': 16,  # Fixed z_dim=16 to match NVAE/DIVA classification space (z_y + z_dy)
            'num_y_classes': num_y_classes,
            'num_r_classes': num_r_classes,

            # IRM-specific hyperparameters (tuned for environment-averaged loss)
            'penalty_weight': getattr(args, 'irm_penalty_weight', 5.0),
            'penalty_anneal_iters': getattr(args, 'irm_anneal_iters', 1000),

            # Training config
            **base_training_config,
            'model_type': 'IRM'
        }
        with open(os.path.join(args.out, 'irm_model_params.json'), 'w') as f:
            json.dump(irm_params, f, indent=2)
        
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
                MinimalInformationPartitionEvaluator
            )
            from core.visualization.plot_information_theoretic import visualize_all

            # Create output directory for IT analysis
            it_output_dir = os.path.join(args.out, 'information_theoretic_analysis')
            os.makedirs(it_output_dir, exist_ok=True)

            # CRITICAL FIX: Create a shuffled dataloader for IT evaluation
            # The standard val_loader has shuffle=False, and the dataset is ordered by domain.
            # This causes balanced_sample_for_visualization to only see domain 0 in its
            # detection phase (first 20 batches), making IT metrics meaningless since
            # I(z;D|Y) requires samples from multiple domains to measure domain information.
            from torch.utils.data import DataLoader as ITDataLoader
            it_eval_loader = ITDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            print(f"📊 Created shuffled dataloader for IT evaluation (ensures domain diversity)")

            # Evaluate each trained model
            it_results = {}

            # Model categories for IT evaluation
            # Decomposed models: have explicit latent partitions (z_y, z_d, z_dy, z_x)
            DECOMPOSED_MODELS = ['nvae', 'diva', 'dann_augmented']
            # Monolithic models: single feature representation (evaluated with different metrics)
            MONOLITHIC_MODELS = ['dann', 'irm']
            # All IT-compatible models
            IT_COMPATIBLE_MODELS = DECOMPOSED_MODELS + MONOLITHIC_MODELS

            # Import monolithic feature extraction
            from core.information_theoretic_evaluation import extract_monolithic_features

            for model_name, model in trained_models.items():
                # Filter to IT-compatible models
                if model_name not in IT_COMPATIBLE_MODELS:
                    print(f"\n⚠️  Skipping {model_name} - IT evaluation not supported")
                    print(f"    Compatible models: {', '.join(IT_COMPATIBLE_MODELS)}")
                    continue

                print(f"\n{'='*60}")
                print(f"Evaluating {model_name.upper()} on information-theoretic metrics")
                print(f"{'='*60}")

                try:
                    import numpy as np

                    # Create evaluator (shared between decomposed and monolithic)
                    evaluator = MinimalInformationPartitionEvaluator(
                        n_neighbors=10,
                        n_bootstrap=10,
                        max_dims=50,
                        pca_variance=0.95
                    )

                    if model_name in DECOMPOSED_MODELS:
                        # =====================================================
                        # DECOMPOSED MODELS (NVAE, DIVA, AugmentedDANN)
                        # =====================================================
                        from core.utils import balanced_sample_for_visualization

                        # Map model names to model types for feature extraction
                        model_type_map = {
                            'nvae': 'nvae',
                            'diva': 'diva',
                            'dann_augmented': 'dann_augmented'
                        }
                        model_type = model_type_map.get(model_name, 'nvae')

                        print(f"  Using balanced sampling for domain-diverse IT evaluation...")
                        features_dict, labels_dict, sampling_stats = balanced_sample_for_visualization(
                            model=model,
                            dataloader=it_eval_loader,
                            device=args.device,
                            model_type=model_type,
                            max_samples=10000,
                            target_samples_per_combination=100,
                            dataset_type="crmnist",
                            use_color_in_combination=False
                        )

                        # Extract latents and convert to numpy
                        z_y = features_dict['zy'].numpy() if features_dict['zy'] is not None else None
                        z_d = features_dict['zd'].numpy() if features_dict['zd'] is not None else None
                        z_dy = features_dict['zdy'].numpy() if features_dict.get('zdy') is not None else None
                        z_x = features_dict['zx'].numpy() if features_dict['zx'] is not None else None

                        # Extract labels - convert one-hot to indices if needed
                        y_labels = labels_dict['y']
                        if len(y_labels.shape) > 1 and y_labels.shape[1] > 1:
                            y_labels = y_labels.argmax(dim=1)
                        y_labels = y_labels.numpy()

                        d_labels = labels_dict['r']  # rotation is domain
                        if len(d_labels.shape) > 1 and d_labels.shape[1] > 1:
                            d_labels = d_labels.argmax(dim=1)
                        d_labels = d_labels.numpy()

                        print(f"  Collected {len(y_labels)} balanced samples across {len(np.unique(d_labels))} domains")
                        print(f"  Domain distribution: {dict(zip(*np.unique(d_labels, return_counts=True)))}")

                        # Run partition evaluation
                        results = evaluator.evaluate_latent_partition(
                            z_y=z_y,
                            z_d=z_d,
                            z_dy=z_dy,
                            z_x=z_x,
                            y_labels=y_labels,
                            d_labels=d_labels,
                            compute_bootstrap=True
                        )

                        # Add unified metrics for cross-model comparison
                        # Z_total = z_y + z_dy (only Y-predictive latents per user choice)
                        z_total_components = [z_y]
                        if z_dy is not None and np.var(z_dy) > 1e-10:
                            z_total_components.append(z_dy)
                        z_total = np.concatenate(z_total_components, axis=1)

                        domain_leakage = evaluator.conditional_mi(z_total, d_labels, y_labels, apply_pca=True)
                        results['unified_metrics'] = {
                            'domain_leakage': domain_leakage,
                            'domain_invariance_score': 1.0 / (1.0 + domain_leakage),
                            'total_class_info': evaluator.mutual_information(z_total, y_labels, apply_pca=True),
                            'class_info_conditional': evaluator.conditional_mi(z_total, y_labels, d_labels, apply_pca=True),
                        }

                    else:
                        # =====================================================
                        # MONOLITHIC MODELS (DANN, IRM)
                        # =====================================================
                        print(f"  Extracting monolithic features for {model_name}...")

                        # Extract features using the monolithic extractor
                        Z, y_labels, d_labels = extract_monolithic_features(
                            model=model,
                            dataloader=it_eval_loader,
                            device=args.device,
                            max_batches=200
                        )

                        print(f"  Collected {len(y_labels)} samples across {len(np.unique(d_labels))} domains")
                        print(f"  Domain distribution: {dict(zip(*np.unique(d_labels, return_counts=True)))}")

                        # Run monolithic evaluation
                        results = evaluator.evaluate_monolithic_representation(
                            Z=Z,
                            y_labels=y_labels,
                            d_labels=d_labels,
                            compute_bootstrap=False
                        )

                    it_results[model_name.upper()] = results

                    # Save individual model results
                    model_it_dir = os.path.join(it_output_dir, model_name)
                    os.makedirs(model_it_dir, exist_ok=True)

                    # Reuse evaluator from above for saving
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

                # Separate decomposed and monolithic models for appropriate comparison
                decomposed_results = {k: v for k, v in it_results.items()
                                      if v.get('model_type') != 'monolithic'}
                monolithic_results = {k: v for k, v in it_results.items()
                                      if v.get('model_type') == 'monolithic'}

                # Standard partition comparison for decomposed models only
                if len(decomposed_results) > 1:
                    comparison = evaluator.compare_models(decomposed_results)
                    evaluator.save_results(
                        comparison,
                        os.path.join(it_output_dir, 'decomposed_model_comparison.json')
                    )
                    # Generate visualizations for decomposed models
                    visualize_all(comparison, it_output_dir)

                # Unified comparison across ALL models (decomposed + monolithic)
                # This uses domain_invariance_score which is comparable across all model types
                unified_comparison = evaluator.compare_unified_metrics(it_results)
                evaluator.save_results(
                    unified_comparison,
                    os.path.join(it_output_dir, 'unified_model_comparison.json')
                )

                # Print unified rankings (all models)
                print("\n" + "="*60)
                print("🏆 UNIFIED RANKINGS - Domain Invariance (All Models)")
                print("="*60)
                print("  (Higher = better domain invariance, lower I(Z;D|Y))")

                sorted_models = sorted(
                    unified_comparison['unified_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for rank, (model, score) in enumerate(sorted_models, 1):
                    model_type = "decomposed" if model in [k.upper() for k in DECOMPOSED_MODELS] else "monolithic"
                    print(f"{rank}. {model:<20} Domain Invariance: {score:.4f}  ({model_type})")

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
        print(f"\nEvaluating domain generalization for withheld domain {ood_domain}...")

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
                dataset_type='crmnist',
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
            if model_type in ['nvae', 'diva', 'dann_augmented'] and not args.skip_training:
                print(f"\n📊 Generating OOD latent space visualization for {model_name}...")
                ood_viz_path = os.path.join(ood_viz_dir, f'{model_name}_latent_spaces_ood.png')
                try:
                    from core.utils import visualize_latent_spaces
                    visualize_latent_spaces(
                        model=model,
                        dataloader=ood_test_loader,
                        device=args.device,
                        type=model_type,  # Use correct model type for feature extraction
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

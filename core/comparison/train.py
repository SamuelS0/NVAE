import core.CRMNIST.model as crmnist_model
import core.WILD.model_wild as wild_model
from core.train import train
import torch.optim as optim
import os
import torch
from core.comparison.dann import DANN
from core.comparison.dann_trainer import DANNTrainer
from core.comparison.irm import IRM
from core.comparison.irm_trainer import IRMTrainer
import numpy as np
from core.CRMNIST.trainer import CRMNISTTrainer
import json
from core.WILD.trainer import WILDTrainer
from core.WILD.staged_trainer import StagedWILDTrainer   

"""
    Train NVAE and comparison models
    Each function returns the trained model and the training metrics
"""

def train_nvae(args, spec_data, train_loader, test_loader, dataset):
    print("Training NVAE...")
    if dataset == 'crmnist':
        nvae = crmnist_model.VAE(class_map=spec_data['class_map'],
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
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        
        model_params = {
            'class_map': spec_data['class_map'],
            'zy_dim': args.zy_dim,
            'zx_dim': args.zx_dim,
            'zay_dim': args.zay_dim,
            'za_dim': args.za_dim,
            'y_dim': spec_data['num_y_classes'],
            'a_dim': spec_data['num_r_classes'],
            'beta_1': args.beta_1,
            'beta_2': args.beta_2,
            'beta_3': args.beta_3,
            'beta_4': args.beta_4,
            'alpha_1': args.alpha_1,
            'alpha_2': args.alpha_2,
            'diva': False
        }
        trainer_class = CRMNISTTrainer
    # TODO: add wild nvae
    elif dataset == 'wild':
        nvae = wild_model.VAE(class_map=None,
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zay_dim=args.zay_dim,
                za_dim=args.za_dim,
                y_dim=args.num_y_classes,
                a_dim=args.num_r_classes,
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                beta_3=args.beta_3,
                beta_4=args.beta_4,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
                recon_weight=args.recon_weight,
                device=args.device,
                resolution=args.resolution,
                model='vae',
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zay=getattr(args, 'l1_lambda_zay', 0.0),
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        
        model_params = {
            'recon_weight': args.recon_weight,
            'zy_dim': args.zy_dim,
            'zx_dim': args.zx_dim,
            'zay_dim': args.zay_dim,
            'za_dim': args.za_dim,
            'y_dim': args.num_y_classes,
            'a_dim': args.num_r_classes,
            'beta_1': args.beta_1,
            'beta_2': args.beta_2,
            'beta_3': args.beta_3,
            'beta_4': args.beta_4,
            'alpha_1': args.alpha_1,
            'alpha_2': args.alpha_2,
            'beta_scale': args.beta_scale,
            'diva': False
        }
        trainer_class = WILDTrainer
    # Move model to device
    with open(os.path.join(args.out, 'nvae_model_params_trainer.json'), 'w') as f:
        json.dump(model_params, f)
    nvae = nvae.to(args.device)
    
    optimizer = optim.Adam(nvae.parameters(), lr=args.learning_rate)
    patience = args.patience
    
    # Define model parameters for saving
    
    
    training_metrics = train(args, nvae, optimizer, train_loader, test_loader, args.device, patience, trainer_class=trainer_class)

    if training_metrics['best_model_state'] is not None:
        nvae.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return nvae, training_metrics

def train_staged_nvae(args, spec_data, train_loader, test_loader, dataset):
    """Train NVAE with staged training for better disentanglement."""
    print("Training NVAE with staged training...")

    if dataset == 'crmnist':
        nvae = crmnist_model.VAE(class_map=spec_data['class_map'],
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
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        
        model_params = {
            'class_map': spec_data['class_map'],
            'zy_dim': args.zy_dim,
            'zx_dim': args.zx_dim,
            'zay_dim': args.zay_dim,
            'za_dim': args.za_dim,
            'y_dim': spec_data['num_y_classes'],
            'a_dim': spec_data['num_r_classes'],
            'beta_1': args.beta_1,
            'beta_2': args.beta_2,
            'beta_3': args.beta_3,
            'beta_4': args.beta_4,
            'alpha_1': args.alpha_1,
            'alpha_2': args.alpha_2,
            'diva': False,
            'staged_training': True
        }
        trainer_class = CRMNISTTrainer  # Note: Would need staged version for CRMNIST

    elif dataset == 'wild':
        nvae = wild_model.VAE(class_map=None,
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zay_dim=args.zay_dim,
                za_dim=args.za_dim,
                y_dim=args.num_y_classes,
                a_dim=args.num_r_classes,
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                beta_3=args.beta_3,
                beta_4=args.beta_4,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
                recon_weight=args.recon_weight,
                device=args.device,
                resolution=args.resolution,
                model='vae',
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zay=getattr(args, 'l1_lambda_zay', 0.0),
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        
        model_params = {
            'recon_weight': args.recon_weight,
            'zy_dim': args.zy_dim,
            'zx_dim': args.zx_dim,
            'zay_dim': args.zay_dim,
            'za_dim': args.za_dim,
            'y_dim': args.num_y_classes,
            'a_dim': args.num_r_classes,
            'beta_1': args.beta_1,
            'beta_2': args.beta_2,
            'beta_3': args.beta_3,
            'beta_4': args.beta_4,
            'alpha_1': args.alpha_1,
            'alpha_2': args.alpha_2,
            'beta_scale': args.beta_scale,
            'diva': False,
            'staged_training': True
        }
        trainer_class = StagedWILDTrainer
        
    # Move model to device
    with open(os.path.join(args.out, 'nvae_model_params_staged_trainer.json'), 'w') as f:
        json.dump(model_params, f)
    nvae = nvae.to(args.device)
    
    optimizer = optim.Adam(nvae.parameters(), lr=args.learning_rate)
    patience = args.patience
    training_metrics = train(args, nvae, optimizer, train_loader, test_loader, args.device, patience, trainer_class=trainer_class)
    

    if training_metrics['best_model_state'] is not None:
        nvae.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best staged model for final evaluation")

    return nvae, training_metrics

def train_diva(args, spec_data, train_loader, test_loader, dataset):
    print("Training DIVA...")
    if dataset == 'crmnist':
        diva = crmnist_model.VAE(class_map=spec_data['class_map'],
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
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        
        model_params = {
        'class_map': spec_data['class_map'],
        'zy_dim': args.zy_dim,
        'zx_dim': args.zx_dim,
        'zay_dim': args.zay_dim,
        'za_dim': args.za_dim,
        'y_dim': spec_data['num_y_classes'],
        'a_dim': spec_data['num_r_classes'],
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'beta_3': args.beta_3,
        'beta_4': args.beta_4,
        'alpha_1': args.alpha_1,
        'alpha_2': args.alpha_2,
        'diva': True
    }
        trainer_class = CRMNISTTrainer

    elif dataset == 'wild':
        diva = wild_model.VAE(class_map=None,
                zy_dim=args.zy_dim,
                zx_dim=args.zx_dim,
                zay_dim=args.zay_dim,
                za_dim=args.za_dim,
                y_dim=args.num_y_classes,
                a_dim=args.num_r_classes,
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                beta_3=args.beta_3,
                beta_4=args.beta_4,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
                recon_weight=args.recon_weight,
                device=args.device,
                resolution=args.resolution,
                model='diva',
                l1_lambda_zy=getattr(args, 'l1_lambda_zy', 0.0),
                l1_lambda_zx=getattr(args, 'l1_lambda_zx', 0.0),
                l1_lambda_zay=getattr(args, 'l1_lambda_zay', 0.0),
                l1_lambda_za=getattr(args, 'l1_lambda_za', 0.0))
        model_params = {
        'recon_weight': args.recon_weight,
        'zy_dim': args.zy_dim,
        'zx_dim': args.zx_dim,
        'zay_dim': args.zay_dim,
        'za_dim': args.za_dim,
        'y_dim': args.num_y_classes,
        'a_dim': args.num_r_classes,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'beta_3': args.beta_3,
        'beta_4': args.beta_4,
        'alpha_1': args.alpha_1,
        'alpha_2': args.alpha_2,
        'beta_scale': args.beta_scale,
        'diva': True
        }
        trainer_class = WILDTrainer
    with open(os.path.join(args.out, 'diva_model_params_trainer.json'), 'w') as f:
        json.dump(model_params, f)
    # Move model to device
    diva = diva.to(args.device)
    
    optimizer = optim.Adam(diva.parameters(), lr=args.learning_rate)
    patience = args.patience
    
    # Define model parameters for saving
    
    
    training_metrics = train(args, diva, optimizer, train_loader, test_loader, args.device, patience, trainer_class=trainer_class)

    if training_metrics['best_model_state'] is not None:
        diva.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return diva, training_metrics

def train_dann(args, spec_data, train_loader, val_loader, dataset):
    print("Training DANN...")

    # latent dimension is the sum of all split latent dimensions
    z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim

    dann = DANN(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], dataset)
    dann = dann.to(args.device)
    optimizer = optim.Adam(dann.parameters(), lr=args.learning_rate)
    patience = args.patience

    # Define model parameters for saving
    model_params = {
        'z_dim': z_dim,
        'num_y_classes': spec_data['num_y_classes'],
        'num_r_classes': spec_data['num_r_classes']
    }
    with open(os.path.join(args.out, 'dann_model_params_trainer.json'), 'w') as f:
        json.dump(model_params, f)

    training_metrics = train(args, dann, optimizer, train_loader, val_loader, args.device, patience, trainer_class=DANNTrainer)

    if training_metrics['best_model_state'] is not None:
        dann.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return dann, training_metrics

def train_irm(args, spec_data, train_loader, val_loader, dataset, seed=None):
    """
    Train IRM model
    """
    print("Training IRM...")

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # latent dimension is the sum of all split latent dimensions
    z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim

    # Use command-line arguments if available, otherwise use defaults
    penalty_weight = getattr(args, 'irm_penalty_weight', 1e4)
    anneal_iters = getattr(args, 'irm_anneal_iters', 500)

    irm = IRM(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], dataset,
             penalty_weight=penalty_weight, penalty_anneal_iters=anneal_iters)
    irm = irm.to(args.device)

    optimizer = optim.Adam(irm.parameters(), lr=args.learning_rate)
    patience = args.patience

    # Define model parameters for saving
    model_params = {
        'z_dim': z_dim,
        'penalty_weight': penalty_weight,
        'penalty_anneal_iters': anneal_iters
    }
    with open(os.path.join(args.out, 'irm_model_params_trainer.json'), 'w') as f:
        json.dump(model_params, f)

    # Simple training loop for IRM
    print("Training IRM...")

    training_metrics = train(args, irm, optimizer, train_loader, val_loader, args.device, patience, trainer_class=IRMTrainer)

    if training_metrics['best_model_state'] is not None:
        irm.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")


    return irm, training_metrics


def train_dann_augmented(args, spec_data, train_loader, val_loader, dataset, seed=None):
    """
    Train Augmented DANN model with 3-way latent decomposition (zy, zd, zdy).
    Works for both CRMNIST (28x28, 128 dims) and WILD (96x96, 512 dims).
    """
    print("Training Augmented DANN...")

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Import AugmentedDANN and DANNTrainer
    from core.CRMNIST.dann_model import AugmentedDANN
    from core.CRMNIST.dann_trainer import DANNTrainer

    # Redistribute dimensions for fair comparison
    # AugmentedDANN uses 3 subspaces (zy, zd, zdy) while NVAE/DIVA use 4
    # To maintain fair comparison, redistribute total dimension across 3 subspaces
    total_dim = args.zy_dim + args.zx_dim + args.zay_dim + args.za_dim
    zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)
    zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)
    zdy_aug = total_dim // 3

    print(f"📏 AugmentedDANN dimension redistribution for {dataset.upper()}:")
    print(f"   Total dimensions: {total_dim}")
    print(f"   zy (class): {zy_aug}, zd (domain): {zd_aug}, zdy (interaction): {zdy_aug}")
    print(f"   Redistributed total: {zy_aug + zd_aug + zdy_aug}")

    # Determine image size based on dataset
    image_size = 28 if dataset == 'crmnist' else 96

    # Create AugmentedDANN model
    dann_aug_model = AugmentedDANN(
        class_map=spec_data.get('class_map', None),  # WILD uses None
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
        image_size=image_size  # Adaptive: 28 for CRMNIST, 96 for WILD
    ).to(args.device)

    # Create optimizer
    optimizer = optim.Adam(dann_aug_model.parameters(), lr=args.learning_rate)

    # Train using DANNTrainer (already dataset-agnostic)
    trainer = DANNTrainer(
        dann_aug_model,
        optimizer,
        args.device,
        args,
        patience=getattr(args, 'patience', 5)
    )
    trainer.train(train_loader, val_loader, args.epochs)

    # Get metrics in standard format
    training_metrics = {
        'best_model_state': trainer.best_model_state,
        'best_val_loss': trainer.best_val_loss,
        'best_val_accuracy': trainer.best_val_accuracy if hasattr(trainer, 'best_val_accuracy') else 0.0,
        'epochs_trained': trainer.epochs_trained,
        'best_model_epoch': trainer.best_epoch
    }

    # Load best model
    if training_metrics['best_model_state'] is not None:
        dann_aug_model.load_state_dict(training_metrics['best_model_state'])
        print("✅ Loaded best AugmentedDANN model for final evaluation")

    return dann_aug_model, training_metrics
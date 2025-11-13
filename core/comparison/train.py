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
    with open(os.path.join(args.out, 'model_params.json'), 'w') as f:
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
    with open(os.path.join(args.out, 'model_params_staged.json'), 'w') as f:
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
    with open(os.path.join(args.out, 'model_params.json'), 'w') as f:
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
    with open(os.path.join(args.out, 'model_params.json'), 'w') as f:
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

    irm = IRM(z_dim, spec_data['num_y_classes'], spec_data['num_r_classes'], dataset, penalty_weight=1e4, penalty_anneal_iters=500)
    irm = irm.to(args.device)

    optimizer = optim.Adam(irm.parameters(), lr=args.learning_rate)
    patience = args.patience

    # Define model parameters for saving
    model_params = {
        'z_dim': z_dim,
        'penalty_weight': 1e4,
        'penalty_anneal_iters': 500
    }
    with open(os.path.join(args.out, 'model_params.json'), 'w') as f:
        json.dump(model_params, f)

    # Simple training loop for IRM
    print("Training IRM...")

    training_metrics = train(args, irm, optimizer, train_loader, val_loader, args.device, patience, trainer_class=IRMTrainer)

    if training_metrics['best_model_state'] is not None:
        irm.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")


    return irm, training_metrics
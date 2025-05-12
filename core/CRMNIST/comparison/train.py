from core.CRMNIST.model import VAE
from core.train import train
import torch.optim as optim
import os
import torch
from core.CRMNIST.comparison.dann import DANN
from core.CRMNIST.comparison.dann_trainer import DANNTrainer

"""
    Train NVAE and comparison models
    Each function returns the trained model and the training metrics
"""

def train_nvae(args, spec_data, train_loader, test_loader, models_dir):
    print("Training NVAE...")
    nvae = VAE(class_map=spec_data['class_map'],
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
               diva=False)
    
    # Move model to device
    nvae = nvae.to(args.device)
    
    optimizer = optim.Adam(nvae.parameters(), lr=args.learning_rate)
    patience = 5
    training_metrics = train(args, nvae, optimizer, train_loader, test_loader, args.device, patience)

    if training_metrics['best_model_state'] is not None:
        nvae.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Save both model parameters and state dict
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
        'diva': False
    }
    
    torch.save({
        'params': model_params,
        'state_dict': nvae.state_dict()
    }, os.path.join(models_dir, "nvae_checkpoint.pt"))

    return nvae, training_metrics

def train_diva(args, spec_data, train_loader, test_loader, models_dir):
    print("Training DIVA...")
    print(f"args zy_dim: {args.zy_dim}")
    diva = VAE(class_map=spec_data['class_map'],
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
               diva=True)
    
    # Move model to device
    diva = diva.to(args.device)
    
    optimizer = optim.Adam(diva.parameters(), lr=args.learning_rate)
    patience = 5
    training_metrics = train(args, diva, optimizer, train_loader, test_loader, args.device, patience)

    if training_metrics['best_model_state'] is not None:
        diva.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Save both model parameters and state dict
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
        'diva': True
    }
    
    torch.save({
        'params': model_params,
        'state_dict': diva.state_dict()
    }, os.path.join(models_dir, "diva_checkpoint.pt"))

    return diva, training_metrics

def train_dann(args, spec_data, train_loader, test_loader, models_dir):
    print("Training DANN...")

    # latent dimension is the sum of all split latent dimensions
    z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim

    dann = DANN(spec_data, z_dim)
    dann = dann.to(args.device)
    optimizer = optim.Adam(dann.parameters(), lr=args.learning_rate)
    patience = 5
    training_metrics = train(args, dann, optimizer, train_loader, test_loader, args.device, patience, DANNTrainer)

    if training_metrics['best_model_state'] is not None:
        dann.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    # Save both model parameters and state dict
    model_params = {
        'spec_data': spec_data,
        'z_dim': z_dim
    }
    
    torch.save({
        'params': model_params,
        'state_dict': dann.state_dict()
    }, os.path.join(models_dir, "dann_checkpoint.pt"))

    return dann, training_metrics
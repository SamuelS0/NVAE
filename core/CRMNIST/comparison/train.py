from core.CRMNIST.model import VAE
from core.train import train
import torch.optim as optim
import os
import torch
from core.CRMNIST.comparison.dann import DANN
from core.CRMNIST.comparison.dann_trainer import DANNTrainer
from core.CRMNIST.comparison.irm import IRM
import numpy as np

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
               alpha_1=args.alpha_1,
               alpha_2=args.alpha_2,
               diva=False)
    
    # Move model to device
    nvae = nvae.to(args.device)
    
    optimizer = optim.Adam(nvae.parameters(), lr=args.learning_rate)
    patience = 5
    
    # Define model parameters for saving
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
    
    training_metrics = train(args, nvae, optimizer, train_loader, test_loader, args.device, patience, model_params=model_params)

    if training_metrics['best_model_state'] is not None:
        nvae.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return nvae, training_metrics

def train_diva(args, spec_data, train_loader, test_loader, models_dir):
    print("Training DIVA...")
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
               alpha_1=args.alpha_1,
               alpha_2=args.alpha_2,
               diva=True)
    
    # Move model to device
    diva = diva.to(args.device)
    
    optimizer = optim.Adam(diva.parameters(), lr=args.learning_rate)
    patience = 5
    
    # Define model parameters for saving
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
    
    training_metrics = train(args, diva, optimizer, train_loader, test_loader, args.device, patience, model_params=model_params)

    if training_metrics['best_model_state'] is not None:
        diva.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return diva, training_metrics

def train_dann(args, spec_data, train_loader, test_loader, models_dir):
    print("Training DANN...")

    # latent dimension is the sum of all split latent dimensions
    z_dim = args.zy_dim + args.za_dim + args.zx_dim + args.zay_dim

    dann = DANN(spec_data, z_dim)
    dann = dann.to(args.device)
    optimizer = optim.Adam(dann.parameters(), lr=args.learning_rate)
    patience = 5
    
    # Define model parameters for saving
    model_params = {
        'spec_data': spec_data,
        'z_dim': z_dim
    }
    
    training_metrics = train(args, dann, optimizer, train_loader, test_loader, args.device, patience, DANNTrainer, model_params=model_params)

    if training_metrics['best_model_state'] is not None:
        dann.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")

    return dann, training_metrics

def train_irm(args, spec_data, train_loader, test_loader, models_dir, seed=None):
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
    
    irm = IRM(spec_data, z_dim, penalty_weight=1e4, penalty_anneal_iters=500)
    irm = irm.to(args.device)
    
    optimizer = optim.Adam(irm.parameters(), lr=args.learning_rate)
    patience = 5
    
    # Define model parameters for saving
    model_params = {
        'spec_data': spec_data,
        'z_dim': z_dim,
        'penalty_weight': 1e4,
        'penalty_anneal_iters': 500
    }
    
    # Simple training loop for IRM
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        irm.train()
        train_loss = 0
        for x, y, c, r in train_loader:
            x, y, r = x.to(args.device), y.to(args.device), r.to(args.device)
            
            optimizer.zero_grad()
            irm_loss, class_loss, penalty = irm.loss_function(x, y, r)
            irm_loss.backward()
            optimizer.step()
            train_loss += class_loss.item()
        
        # Validation
        irm.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y, c, r in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits, _ = irm.forward(x)
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = irm.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_model_state is not None:
        irm.load_state_dict(best_model_state)
    
    training_metrics = {
        'best_model_epoch': epoch + 1,
        'best_validation_loss': 1.0 - best_val_acc,  # Convert accuracy to loss-like metric
        'best_batch_metrics': {'y_accuracy': best_val_acc},
        'model_params': model_params
    }
    
    return irm, training_metrics
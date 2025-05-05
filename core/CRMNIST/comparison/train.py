from core.CRMNIST.model import VAE
from core.train import train
import torch.optim as optim
import os
import torch

"""
    Train NVAE and comparison models
    Each function returns the trained model and the training metrics
"""

def train_nvae(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir):
    print("Training NVAE...")
    nvae = VAE(class_map=class_map,
               zy_dim=args.zy_dim,
               zx_dim=args.zx_dim,
               zay_dim=args.zay_dim,
               za_dim=args.za_dim,
               y_dim=num_y_classes,
               a_dim=num_r_classes,
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
    
    final_model_path = os.path.join(models_dir, f"nvae_model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(nvae.state_dict(), final_model_path)

    return nvae, training_metrics

def train_diva(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir):
    print("Training DIVA...")
    print(f"args zy_dim: {args.zy_dim}")
    diva = VAE(class_map=class_map,
               zy_dim=args.zy_dim,
               zx_dim=args.zx_dim,
               zay_dim=args.zay_dim,
               za_dim=args.za_dim,
               y_dim=num_y_classes,
               a_dim=num_r_classes,
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
    
    final_model_path = os.path.join(models_dir, f"diva_model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(diva.state_dict(), final_model_path)

    return diva, training_metrics
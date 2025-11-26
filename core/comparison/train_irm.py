import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from core.CRMNIST.comparison.irm import IRM
from core.CRMNIST.comparison.irm_trainer import IRMTrainer

def train_irm_model():
    """Train IRM model on Colored MNIST"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data setup - you'll need to import your data generation function
    # This assumes you have a get_data_loaders function in your data_generation module
    print("Loading Colored MNIST data...")
    try:
        from core.CRMNIST.data_generation import get_data_loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=128,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
    except ImportError:
        print("Error: Could not import data_generation module.")
        print("Please make sure your data loading function is available.")
        print("You may need to adjust the import path based on your project structure.")
        return None, None
    
    # Model specifications
    spec_data = {
        'num_y_classes': 10,  # 10 digits
        'num_r_classes': 6,   # 6 rotation domains
    }
    
    # Model setup
    z_dim = 32  # Same total dimension as your VAE (8+8+8+8=32)
    penalty_weight = 1e4  # IRM penalty weight
    penalty_anneal_iters = 500  # Start applying penalty after 500 iterations
    
    print(f"Creating IRM model with z_dim={z_dim}, penalty_weight={penalty_weight}")
    model = IRM(
        spec_data=spec_data,
        z_dim=z_dim,
        penalty_weight=penalty_weight,
        penalty_anneal_iters=penalty_anneal_iters
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer setup
    save_dir = './checkpoints/irm'
    trainer = IRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        save_dir=save_dir
    )
    
    # Training
    num_epochs = 100
    print(f"Starting training for {num_epochs} epochs...")
    
    best_val_acc, final_test_acc = trainer.train(
        num_epochs=num_epochs,
        save_every=10,
        visualize_every=20
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    
    return model, trainer

def compare_with_vae():
    """
    Example function showing how to compare IRM with your VAE model
    """
    print("\n" + "="*50)
    print("COMPARISON BETWEEN VAE AND IRM")
    print("="*50)
    
    # This is a template for comparison - you would need to load your trained VAE model
    print("To compare with your VAE model:")
    print("1. Load your trained VAE model")
    print("2. Extract features using the same test data")
    print("3. Compare latent space visualizations")
    print("4. Compare classification accuracies")
    print("5. Analyze which approach better separates task-relevant vs spurious features")
    
    # Example comparison code (you would need to adapt this):
    """
    # Load your trained VAE
    vae_model = VAE(...)  # Your VAE model
    vae_model.load_state_dict(torch.load('path_to_vae_checkpoint.pth'))
    
    # Load trained IRM
    irm_model = IRM(...)
    irm_model.load_state_dict(torch.load('checkpoints/irm/best_irm_model.pth'))
    
    # Compare on test data
    test_loader = ...  # Your test data loader
    
    # VAE classification accuracy
    vae_acc = vae_model.get_accuracy(test_loader, device)
    
    # IRM classification accuracy  
    irm_acc = irm_model.get_accuracy(test_loader, device)
    
    print(f"VAE Test Accuracy: {vae_acc:.4f}")
    print(f"IRM Test Accuracy: {irm_acc:.4f}")
    
    # Visualize latent spaces
    vae_model.visualize_latent_spaces(test_loader, device, 'vae_latent.png')
    irm_model.visualize_latent_space(test_loader, device, 'irm_latent.png')
    """

if __name__ == "__main__":
    # Train IRM model
    model, trainer = train_irm_model()
    
    if model is not None:
        # Show comparison template
        compare_with_vae()
        
        print(f"\nIRM model training completed!")
        print(f"Model saved in: ./checkpoints/irm/")
        print(f"Visualizations saved in: ./checkpoints/irm/")
        print(f"\nTo use the trained model:")
        print(f"1. Load the model: model.load_state_dict(torch.load('checkpoints/irm/best_irm_model.pth'))")
        print(f"2. Compare with your VAE model using the same encoder architecture")
        print(f"3. Analyze the differences in learned representations") 
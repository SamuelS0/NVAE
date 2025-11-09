"""
Example usage of the Augmented DANN model for CRMNIST dataset.

This script demonstrates:
1. How to train an Augmented DANN model
2. How to evaluate disentanglement quality
3. How to visualize latent spaces
4. How to load and use a trained model

Run with:
python -m core.CRMNIST.example_dann_usage
"""

import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Some visualization features will be disabled.")
    plt = None

# Import DANN components
try:
    from core.CRMNIST.dann_model import AugmentedDANN
    from core.CRMNIST.dann_trainer import DANNTrainer
    from core.CRMNIST.data_generation import generate_crmnist_dataset
    from core.utils import visualize_latent_spaces
    import core.CRMNIST.utils_crmnist as utils_crmnist
except ImportError as e:
    raise ImportError(f"Required DANN modules not found: {e}. Please ensure all core modules are properly installed.")


def load_config():
    """Load CRMNIST configuration"""
    config_path = '/Users/samuelspeas/repos/NVAE/conf/crmnist.json'
    
    with open(config_path, 'r') as file:
        spec_data = json.load(file)
    
    # Prepare domain data
    domain_data = {int(key): value for key, value in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data
    
    # Choose labels subset
    spec_data['y_c'], subsets = utils_crmnist.choose_label_subset(spec_data)
    for i, subset in subsets.items():
        if i in domain_data:
            domain_data[i]['subset'] = subset
    
    return spec_data


def create_datasets(spec_data, batch_size=64):
    """Create train, validation, and test datasets"""
    print("Creating datasets...")
    
    # Generate datasets
    train_dataset = generate_crmnist_dataset(spec_data, train=True)
    test_dataset = generate_crmnist_dataset(spec_data, train=False)
    
    # Create validation split
    train_size = len(train_dataset)
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train size: {len(train_subset)}")
    print(f"Validation size: {len(val_subset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_dann_model(spec_data, device='cpu'):
    """Create and initialize DANN model"""
    print("Creating DANN model...")
    
    class_map = spec_data['class_map']
    num_y_classes = spec_data['num_y_classes']
    num_r_classes = spec_data['num_r_classes']
    
    model = AugmentedDANN(
        class_map=class_map,
        zy_dim=12,           # Class-specific features
        zd_dim=12,           # Domain-specific features  
        zdy_dim=12,          # Domain-class interaction
        y_dim=num_y_classes, # Number of digit classes (10)
        d_dim=num_r_classes, # Number of rotation domains (6)
        lambda_reversal=1.0,  # Initial gradient reversal strength
        sparsity_weight=0.01, # L1 penalty on interaction features
        alpha_y=1.0,         # Weight for class classification
        alpha_d=1.0,         # Weight for domain classification
        beta_adv=1.0         # Weight for adversarial losses
    )
    
    model = model.to(device)
    
    print(f"Model created with:")
    print(f"  Class dimension (zy): {model.zy_dim}")
    print(f"  Domain dimension (zd): {model.zd_dim}")
    print(f"  Interaction dimension (zdy): {model.zdy_dim}")
    print(f"  Total latent dimension: {model.z_total_dim}")
    
    return model


def train_dann_model(model, train_loader, val_loader, device='cpu', epochs=20):
    """Train the DANN model"""
    print(f"Training DANN model for {epochs} epochs...")
    
    # Create mock args object for trainer
    class Args:
        def __init__(self):
            self.out = './dann_results'
            self.setting = 'dann_example'  # Required by DANNTrainer
            self.dataset = 'crmnist'
            self.zy_dim = 12
            self.zx_dim = 12
            self.zay_dim = 12
            self.za_dim = 12
            self.beta_1 = 1.0
            self.beta_2 = 1.0
            self.beta_3 = 1.0
            self.beta_4 = 1.0
            self.alpha_1 = 1.0
            self.alpha_2 = 1.0
            self.epochs = epochs
            self.batch_size = 64
            self.learning_rate = 1e-3
    
    args = Args()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = DANNTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        args=args,
        patience=5,
        lambda_schedule=True  # Enable gradient reversal scheduling
    )
    
    # Train the model
    trainer.train(train_loader, val_loader, epochs)
    
    print(f"Training completed! Best epoch: {trainer.best_epoch}")
    print(f"Best validation score: {trainer.best_val_accuracy:.4f}")
    
    return trainer


def evaluate_model(trainer, test_loader):
    """Evaluate the trained DANN model"""
    print("\nEvaluating model on test set...")
    
    # Evaluate disentanglement
    results = trainer.evaluate_disentanglement(test_loader)
    
    print("\nDisentanglement Quality Assessment:")
    print("-" * 40)
    print(f"Class Accuracy: {results['class_accuracy']:.4f}")
    print(f"Domain Accuracy: {results['domain_accuracy']:.4f}")
    print(f"Domain Adversarial Accuracy: {results['domain_adversarial_accuracy']:.4f} (should be low)")
    print(f"Class Adversarial Accuracy: {results['class_adversarial_accuracy']:.4f} (should be low)")
    print(f"Overall Disentanglement Score: {results['disentanglement_score']:.4f} (higher is better)")
    
    # Interpretation guide
    print("\nInterpretation Guide:")
    print("- High class/domain accuracy: Model performs well on main tasks")
    print("- Low adversarial accuracies: Good disentanglement (zy doesn't encode domain info, zd doesn't encode class info)")
    print("- High disentanglement score: Overall good separation of factors")
    
    return results


def visualize_latent_spaces_dann(model, test_loader, device, save_path='./dann_latent_visualization.png'):
    """Visualize DANN latent spaces using t-SNE"""
    print(f"\nVisualizing latent spaces...")
    
    # Use the utility function with DANN type
    visualize_latent_spaces(
        model=model,
        dataloader=test_loader, 
        device=device,
        type="dann",
        save_path=save_path,
        max_samples=2000
    )
    
    print(f"Latent space visualization saved to {save_path}")


def demonstrate_feature_extraction(model, test_loader, device):
    """Demonstrate how to extract and analyze features"""
    print("\nDemonstrating feature extraction...")
    
    model.eval()
    
    # Get a batch of test data
    x, y, c, r = next(iter(test_loader))
    x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
    
    with torch.no_grad():
        # Extract latent features
        zy, zd, zdy = model.extract_features(x)
        
        # Get predictions
        outputs = model.dann_forward(x)
        
        print(f"Batch size: {x.shape[0]}")
        print(f"Feature dimensions:")
        print(f"  zy (class-specific): {zy.shape}")
        print(f"  zd (domain-specific): {zd.shape}")  
        print(f"  zdy (interaction): {zdy.shape}")
        
        # Show prediction accuracies for this batch
        y_pred = outputs['y_pred_main'].argmax(dim=1)
        d_pred = outputs['d_pred_main'].argmax(dim=1)
        
        if len(y.shape) > 1:
            y_true = y.argmax(dim=1)
        else:
            y_true = y
            
        if len(r.shape) > 1:
            r_true = r.argmax(dim=1)
        else:
            r_true = r
        
        class_acc = (y_pred == y_true).float().mean().item()
        domain_acc = (d_pred == r_true).float().mean().item()
        
        print(f"\nBatch predictions:")
        print(f"  Class accuracy: {class_acc:.4f}")
        print(f"  Domain accuracy: {domain_acc:.4f}")
        
        # Show some examples
        print(f"\nFirst 5 samples:")
        for i in range(min(5, x.shape[0])):
            print(f"  Sample {i}: True class={y_true[i].item()}, Pred class={y_pred[i].item()}, "
                  f"True domain={r_true[i].item()}, Pred domain={d_pred[i].item()}")


def save_model_demo(model, save_path='./dann_model.pt'):
    """Demonstrate how to save and load a trained model"""
    print(f"\nSaving model to {save_path}...")
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'zy_dim': model.zy_dim,
            'zd_dim': model.zd_dim,
            'zdy_dim': model.zdy_dim,
            'y_dim': model.y_dim,
            'd_dim': model.d_dim,
            'sparsity_weight': model.sparsity_weight,
            'alpha_y': model.alpha_y,
            'alpha_d': model.alpha_d,
            'beta_adv': model.beta_adv
        }
    }, save_path)
    
    print("Model saved successfully!")


def load_model_demo(save_path='./dann_model.pt', spec_data=None, device='cpu'):
    """Demonstrate how to load a trained model"""
    print(f"\nLoading model from {save_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(save_path, map_location=device)
    config = checkpoint['model_config']
    
    # Recreate model
    model = AugmentedDANN(
        class_map=spec_data['class_map'],
        zy_dim=config['zy_dim'],
        zd_dim=config['zd_dim'],
        zdy_dim=config['zdy_dim'],
        y_dim=config['y_dim'],
        d_dim=config['d_dim'],
        sparsity_weight=config['sparsity_weight'],
        alpha_y=config['alpha_y'],
        alpha_d=config['alpha_d'],
        beta_adv=config['beta_adv']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def main():
    """Main example function"""
    print("=" * 60)
    print("Augmented DANN for CRMNIST - Complete Example")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('./dann_results', exist_ok=True)
    
    try:
        # 1. Load configuration and create datasets
        spec_data = load_config()
        train_loader, val_loader, test_loader = create_datasets(spec_data, batch_size=64)
        
        # 2. Create and train model
        model = create_dann_model(spec_data, device)
        trainer = train_dann_model(model, train_loader, val_loader, device, epochs=5)  # Short training for demo
        
        # 3. Evaluate model
        results = evaluate_model(trainer, test_loader)
        
        # 4. Visualize latent spaces
        visualize_latent_spaces_dann(model, test_loader, device, './dann_results/latent_spaces.png')
        
        # 5. Demonstrate feature extraction
        demonstrate_feature_extraction(model, test_loader, device)
        
        # 6. Save and load model demo
        save_model_demo(model, './dann_results/trained_model.pt')
        loaded_model = load_model_demo('./dann_results/trained_model.pt', spec_data, device)
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("Check './dann_results/' for saved outputs.")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found. {e}")
        print("Please ensure the CRMNIST configuration file exists.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
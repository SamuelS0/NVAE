#!/usr/bin/env python3
"""
Standalone script to test latent variable expressiveness on a trained VAE model.

Usage:
    python test_expressiveness.py --model_path model.pt --config conf/crmnist.json --out results/
"""

import torch
import torch.optim as optim
import json
import os
import argparse
from torch.utils.data import DataLoader

# Import necessary modules
import core.CRMNIST.utils_crmnist
from core.CRMNIST.data_generation import generate_crmnist_dataset
from core.CRMNIST.model import VAE
from core.CRMNIST.latent_expressiveness import evaluate_latent_expressiveness

def load_model(model_path, config_path, model_params_path, device):
    """Load a trained VAE model from checkpoint."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        spec_data = json.load(f)
    
    # Prepare domain data
    domain_data = {int(key): value for key, value in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data
    class_map = spec_data['class_map']
    
    # Choose labels subset
    spec_data['y_c'], subsets = core.CRMNIST.utils_crmnist.choose_label_subset(spec_data)
    for i, subset in subsets.items():
        if i in domain_data:
            domain_data[i]['subset'] = subset
    
    # Get dataset dimensions
    num_y_classes = spec_data['num_y_classes']
    num_r_classes = spec_data['num_r_classes']
    
    # Initialize model with default parameters (you may want to load these from a config)
    # Load model parameters from config
    #go back one directory and find model_params.json
    print(f"Loading model parameters from {model_params_path}")
    with open(model_params_path, 'r') as f:
        model_params = json.load(f)
    
    model = VAE(
        class_map=class_map,
        zy_dim=model_params['zy_dim'],
        zx_dim=model_params['zx_dim'], 
        zay_dim=model_params['zay_dim'],
        za_dim=model_params['za_dim'],
        y_dim=num_y_classes,
        a_dim=num_r_classes,
        beta_1=model_params['beta_1'],
        beta_2=model_params['beta_2'],
        beta_3=model_params['beta_3'],
        beta_4=model_params['beta_4'],
        diva=model_params.get('diva', False)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, spec_data

def main():
    parser = argparse.ArgumentParser(description='Test latent variable expressiveness')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration JSON file')
    parser.add_argument('--model_params_path', '-p',type=str, required=True,
                       help='Path to model parameters JSON file')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for data loading')
    parser.add_argument('--intensity', type=float, default=1.5,
                       help='Transform intensity for dataset generation')
    parser.add_argument('--intensity_decay', type=float, default=1.0,
                       help='Transform intensity decay for dataset generation')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--diva', action='store_true', default=False,
                       help='Model is a DIVA variant (no zay component)')
    
    args = parser.parse_args()
    
    # Set up device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    print(f"🔧 Using device: {device}")
    print(f"📁 Model path: {args.model_path}")
    print(f"⚙️  Config path: {args.config}")
    print(f"📂 Output directory: {args.out}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load model
    print("\n🔄 Loading trained model...")
    model, spec_data = load_model(args.model_path, args.config, args.model_params_path, device)
    
    # Generate datasets
    print("\n📊 Loading datasets...")
    train_dataset = generate_crmnist_dataset(
        spec_data, train=True,
        transform_intensity=args.intensity,
        transform_decay=args.intensity_decay,
        use_cache=True
    )
    test_dataset = generate_crmnist_dataset(
        spec_data, train=False,
        transform_intensity=args.intensity,
        transform_decay=args.intensity_decay,
        use_cache=True
    )
    
    # Create validation split
    train_size = len(train_dataset)
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"📈 Train samples: {len(train_subset)}")
    print(f"📈 Validation samples: {len(val_subset)}")
    print(f"📈 Test samples: {len(test_dataset)}")
    # Run expressiveness evaluation
    print("\n🧪 Evaluating latent variable expressiveness...")
    results = evaluate_latent_expressiveness(
        model, train_loader, val_loader, test_loader, device, args.out
    )
    
    print("\n✅ Expressiveness evaluation completed!")
    print(f"📊 Results saved to: {args.out}")
    
    # Print summary
    print("\n📋 SUMMARY:")
    print("="*50)
    
    if 'domain_za_alone' in results and 'domain_za_zay' in results:
        domain_improvement_val = results['domain_za_zay']['val_acc'] - results['domain_za_alone']['val_acc']
        domain_improvement_test = results['domain_za_zay']['test_acc'] - results['domain_za_alone']['test_acc']
        print(f"Domain classification improvement (Val):  +{domain_improvement_val:.4f} ({domain_improvement_val*100:.2f}%)")
        print(f"Domain classification improvement (Test): +{domain_improvement_test:.4f} ({domain_improvement_test*100:.2f}%)")
    
    if 'label_zy_alone' in results and 'label_zy_zay' in results:
        label_improvement_val = results['label_zy_zay']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zay']['test_acc'] - results['label_zy_alone']['test_acc']
        print(f"Label classification improvement (Val):   +{label_improvement_val:.4f} ({label_improvement_val*100:.2f}%)")
        print(f"Label classification improvement (Test):  +{label_improvement_test:.4f} ({label_improvement_test*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    main() 
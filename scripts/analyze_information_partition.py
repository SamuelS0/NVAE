#!/usr/bin/env python3
"""
Standalone script for information-theoretic evaluation of trained models.

This script loads pre-trained models and evaluates their adherence to the
Minimal Information Partition theorem using information-theoretic metrics.

Usage:
    python scripts/analyze_information_partition.py \\
        --dataset crmnist \\
        --model_dir results/crmnist_20epochs \\
        --output_dir results/crmnist_20epochs/information_theoretic_analysis \\
        --max_batches 200 \\
        --bootstrap 100 \\
        --cuda

Arguments:
    --dataset: Dataset name (crmnist or wild)
    --model_dir: Directory containing trained model checkpoints
    --output_dir: Directory to save analysis results
    --max_batches: Number of batches to use for evaluation (controls sample size)
    --bootstrap: Number of bootstrap iterations for confidence intervals
    --cuda: Use GPU if available
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.information_theoretic_evaluation import (
    MinimalInformationPartitionEvaluator,
    extract_latents_from_model,
    evaluate_model
)
from core.visualization.plot_information_theoretic import visualize_all
from core.CRMNIST.model import VAE as CRMNIST_VAE
from core.WILD.model import VAE as WILD_VAE
from core.comparison.dann import DANN
from core.comparison.augmented_dann import AugmentedDANN
from core.comparison.irm import IRM


def load_dataset(dataset_name, batch_size=64):
    """Load the appropriate dataset"""
    if dataset_name.lower() == 'crmnist':
        from core.CRMNIST.data import get_dataloaders
        _, val_loader, _ = get_dataloaders(batch_size=batch_size)
        return val_loader
    elif dataset_name.lower() == 'wild':
        from core.WILD.data import get_dataloaders
        _, val_loader, _ = get_dataloaders(batch_size=batch_size)
        return val_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_model(model_path, model_type, dataset_name, device):
    """Load a trained model from checkpoint"""

    # Determine model class
    if dataset_name.lower() == 'crmnist':
        if 'nvae' in model_type.lower():
            model = CRMNIST_VAE(diva=False).to(device)
        elif 'diva' in model_type.lower():
            model = CRMNIST_VAE(diva=True).to(device)
        elif 'augmented_dann' in model_type.lower() or 'augdann' in model_type.lower():
            from core.comparison.models import CNN as ImageCNN
            model = AugmentedDANN(
                feature_extractor=ImageCNN(input_channels=3, num_classes=10),
                num_classes=10,
                num_domains=6,
                augmented=True
            ).to(device)
        elif 'dann' in model_type.lower():
            from core.comparison.models import CNN as ImageCNN
            model = DANN(
                feature_extractor=ImageCNN(input_channels=3, num_classes=10),
                num_classes=10,
                num_domains=6
            ).to(device)
        elif 'irm' in model_type.lower():
            from core.comparison.models import CNN as ImageCNN
            model = IRM(
                feature_extractor=ImageCNN(input_channels=3, num_classes=10),
                num_classes=10
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    elif dataset_name.lower() == 'wild':
        if 'nvae' in model_type.lower():
            model = WILD_VAE(diva=False).to(device)
        elif 'diva' in model_type.lower():
            model = WILD_VAE(diva=True).to(device)
        # TODO: Add DANN/IRM models for WILD if needed
        else:
            raise ValueError(f"Unknown model type for WILD: {model_type}")

    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✓ Loaded {model_type} from {model_path}")

    return model


def find_model_checkpoint(model_dir, model_name):
    """Find the best model checkpoint for a given model name"""
    # Try various naming patterns
    patterns = [
        f"{model_name}.pt",
        f"{model_name}_final.pt",
        f"{model_name}_best.pt",
        f"{model_name}_model.pt",
    ]

    for pattern in patterns:
        path = os.path.join(model_dir, pattern)
        if os.path.exists(path):
            return path

    # Try in subdirectories
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if model_name.lower() in file.lower() and file.endswith('.pt'):
                return os.path.join(root, file)

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Information-Theoretic Evaluation of Model Partitions'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['crmnist', 'wild'],
                       help='Dataset name')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['nvae', 'diva', 'dann', 'augmented_dann', 'irm'],
                       help='Models to evaluate')
    parser.add_argument('--max_batches', type=int, default=200,
                       help='Number of batches for evaluation (~20k samples with batch_size=64)')
    parser.add_argument('--bootstrap', type=int, default=100,
                       help='Number of bootstrap iterations (0 to disable)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for data loading')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    val_loader = load_dataset(args.dataset, batch_size=args.batch_size)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each model
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80 + "\n")

    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*80}\n")

        # Find model checkpoint
        model_path = find_model_checkpoint(args.model_dir, model_name)
        if model_path is None:
            print(f"⚠️  Could not find checkpoint for {model_name}, skipping...")
            continue

        try:
            # Load model
            model = load_model(model_path, model_name, args.dataset, device)

            # Evaluate
            results = evaluate_model(
                model,
                val_loader,
                device,
                max_batches=args.max_batches,
                n_bootstrap=args.bootstrap
            )

            all_results[model_name.upper()] = results

            # Save individual model results
            model_output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            evaluator = MinimalInformationPartitionEvaluator()
            evaluator.save_results(
                results,
                os.path.join(model_output_dir, 'it_results.json')
            )

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare models
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARING MODELS")
        print("="*80 + "\n")

        evaluator = MinimalInformationPartitionEvaluator()
        comparison = evaluator.compare_models(all_results)

        # Save comparison results
        evaluator.save_results(
            comparison,
            os.path.join(args.output_dir, 'model_comparison.json')
        )

        # Generate visualizations
        visualize_all(comparison, args.output_dir)

        # Print summary
        print("\n" + "="*80)
        print("FINAL RANKINGS")
        print("="*80 + "\n")

        sorted_models = sorted(
            comparison['model_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (model, score) in enumerate(sorted_models, 1):
            print(f"{rank}. {model:<20} Partition Quality: {score:.4f}")

        print("\n" + "="*80)
        print(f"✅ Analysis complete! Results saved to: {args.output_dir}")
        print("="*80 + "\n")

    else:
        print("\n⚠️  Not enough models evaluated for comparison")


if __name__ == '__main__':
    main()

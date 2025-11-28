#!/usr/bin/env python
"""
Run Information-Theoretic analysis on existing grid search experiments.

This script loads trained models from a completed grid search and runs
IT analysis on each model that supports it (NVAE, DIVA, AugmentedDANN).
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.information_theoretic_evaluation import (
    MinimalInformationPartitionEvaluator,
    extract_latents_from_model,
)
from core.CRMNIST.grid_search.runner import IT_SUPPORTED_MODELS

# Model type to class mapping
MODEL_CLASSES = {
    'nvae': ('core.CRMNIST.model', 'VAE'),
    'diva': ('core.CRMNIST.model', 'VAE'),
    'dann_augmented': ('core.CRMNIST.dann_model', 'AugmentedDANN'),
}


def load_model(model_type: str, config: dict, device: str):
    """Load a trained model from config."""
    params = config.get('params', {})

    if model_type in ['nvae', 'diva']:
        from core.CRMNIST.model import VAE

        # Load spec_data for model dimensions
        with open('conf/crmnist.json', 'r') as f:
            spec_data = json.load(f)

        n_classes = spec_data.get('n_classes', 10)
        n_domains = len(spec_data.get('domain_data', {}))

        # Create class_map (required by VAE)
        class_map = {i: i for i in range(n_classes)}

        model = VAE(
            class_map=class_map,
            y_dim=n_classes,
            a_dim=n_domains,
            in_channels=3,
            zy_dim=params.get('zy_dim', 8),
            zx_dim=params.get('zx_dim', 8),
            zd_dim=params.get('zd_dim', 8),
            zdy_dim=params.get('zdy_dim', 8),
            diva=params.get('diva', model_type == 'diva'),
            beta_zy=params.get('beta_zy', params.get('beta_1', 1)),
            beta_zx=params.get('beta_zx', params.get('beta_2', 1)),
            beta_zdy=params.get('beta_zdy', params.get('beta_3', 1)),
            beta_zd=params.get('beta_zd', params.get('beta_4', 1)),
            alpha_y=params.get('alpha_y', params.get('alpha_1', 1)),
            alpha_d=params.get('alpha_d', params.get('alpha_2', 1)),
        ).to(device)

    elif model_type == 'dann_augmented':
        from core.CRMNIST.dann_model import AugmentedDANN

        with open('conf/crmnist.json', 'r') as f:
            spec_data = json.load(f)

        n_classes = spec_data.get('n_classes', 10)
        n_domains = len(spec_data.get('domain_data', {}))

        # Create class_map (required by AugmentedDANN)
        class_map = {i: i for i in range(n_classes)}

        # AugmentedDANN redistributes dimensions for fair comparison with NVAE/DIVA
        # Total dim = zy + zx + zdy + zd, redistributed across 3 subspaces
        zy_dim = params.get('zy_dim', 8)
        zx_dim = params.get('zx_dim', 8)
        zdy_dim = params.get('zdy_dim', 8)
        zd_dim = params.get('zd_dim', 8)
        total_dim = zy_dim + zx_dim + zdy_dim + zd_dim
        zy_aug = total_dim // 3 + (1 if total_dim % 3 > 0 else 0)
        zd_aug = total_dim // 3 + (1 if total_dim % 3 > 1 else 0)
        zdy_aug = total_dim // 3

        model = AugmentedDANN(
            class_map=class_map,
            in_channels=3,
            y_dim=n_classes,
            d_dim=n_domains,
            zy_dim=zy_aug,
            zd_dim=zd_aug,
            zdy_dim=zdy_aug,
            alpha_y=params.get('alpha_y', params.get('alpha_1', 1)),
            alpha_d=params.get('alpha_d', params.get('alpha_2', 1)),
            beta_adv=params.get('beta_adv', 0.5),
            sparsity_weight_zdy=params.get('sparsity_weight_zdy', 2.0),
            sparsity_weight_zy_zd=params.get('sparsity_weight_zy_zd', 1.0),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type for IT analysis: {model_type}")

    return model


def load_data(batch_size: int = 64):
    """Load validation data for IT analysis."""
    from core.CRMNIST.data_generation import generate_crmnist_dataset

    with open('conf/crmnist.json', 'r') as f:
        spec_data = json.load(f)

    domain_data = {int(k): v for k, v in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data

    if 'y_c' not in spec_data:
        spec_data['y_c'] = 7

    # Use validation split for IT analysis
    val_dataset = generate_crmnist_dataset(
        spec_data, train=True,
        transform_intensity=1.5,
        transform_decay=1.0,
        use_cache=True,
        exclude_domains=[5],  # OOD domain
        base_split='val',
        base_split_ratio=0.8,
        base_split_seed=42
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return val_loader


def run_it_analysis_on_experiment(
    exp_dir: str,
    model_type: str,
    config: dict,
    val_loader,
    device: str,
    n_bootstrap: int = 0,
    max_batches: int = 100,
    force: bool = False,
):
    """Run IT analysis on a single experiment."""

    # Check if IT analysis already exists
    it_path = os.path.join(exp_dir, 'it_analysis.json')
    if os.path.exists(it_path) and not force:
        print(f"  IT analysis already exists, skipping...")
        return None

    # Check if model checkpoint exists
    model_path = os.path.join(exp_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"  No model checkpoint found, skipping...")
        return None

    # Load model
    model = load_model(model_type, config, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Extract latents
    z_y, z_d, z_dy, z_x, y_labels, d_labels = extract_latents_from_model(
        model, val_loader, device, max_batches=max_batches
    )

    # Create evaluator
    evaluator = MinimalInformationPartitionEvaluator(
        n_neighbors=7,
        n_bootstrap=n_bootstrap,
        max_dims=30,
        pca_variance=0.95
    )

    # Run evaluation (suppress verbose output)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        it_results = evaluator.evaluate_latent_partition(
            z_y, z_d, z_dy, z_x,
            y_labels, d_labels,
            compute_bootstrap=(n_bootstrap > 0)
        )
    finally:
        sys.stdout = old_stdout

    # Save full IT results
    evaluator.save_results(it_results, it_path)

    # Extract summary metrics
    metrics = it_results.get('metrics', {})
    it_summary = {
        'I_zy_Y_given_D': metrics.get('I(z_y;Y|D)', 0.0),
        'I_zy_D_given_Y': metrics.get('I(z_y;D|Y)', 0.0),
        'z_y_specificity': metrics.get('z_y_specificity', 0.0),
        'I_zd_D_given_Y': metrics.get('I(z_d;D|Y)', 0.0),
        'I_zd_Y_given_D': metrics.get('I(z_d;Y|D)', 0.0),
        'z_d_specificity': metrics.get('z_d_specificity', 0.0),
        'I_zdy_Y_D': metrics.get('I(z_dy;Y;D)', 0.0),
        'I_zdy_joint': metrics.get('I(z_dy;Y,D)', 0.0),
        'partition_quality': metrics.get('partition_quality', 0.0),
    }

    # Update metrics.json with IT metrics
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            exp_metrics = json.load(f)
        exp_metrics['it_metrics'] = it_summary
        with open(metrics_path, 'w') as f:
            json.dump(exp_metrics, f, indent=2, default=str)

    return it_summary


def main():
    parser = argparse.ArgumentParser(description='Run IT analysis on grid search experiments')
    parser.add_argument('--grid-dir', type=str, required=True,
                        help='Directory containing grid search results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--n-bootstrap', type=int, default=0,
                        help='Number of bootstrap samples (0=no bootstrap)')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Max batches for IT analysis')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if IT analysis exists')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        choices=['nvae', 'diva', 'dann_augmented'],
                        help='Model types to analyze (default: all supported)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get list of experiments
    experiments_dir = os.path.join(args.grid_dir, 'experiments')
    if not os.path.exists(experiments_dir):
        print(f"No experiments directory found at {experiments_dir}")
        return

    experiments = sorted(os.listdir(experiments_dir))
    print(f"Found {len(experiments)} experiments")

    # Load validation data once
    print("Loading validation data...")
    val_loader = load_data()
    print(f"Loaded {len(val_loader.dataset)} validation samples")

    # Filter by supported models
    supported_models = set(args.models) if args.models else IT_SUPPORTED_MODELS

    # Process each experiment
    success_count = 0
    skip_count = 0
    error_count = 0

    for exp_name in tqdm(experiments, desc="Running IT analysis"):
        exp_dir = os.path.join(experiments_dir, exp_name)
        config_path = os.path.join(exp_dir, 'config.json')

        if not os.path.exists(config_path):
            skip_count += 1
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_type = config.get('model_type', '')

        if model_type not in supported_models:
            skip_count += 1
            continue

        try:
            result = run_it_analysis_on_experiment(
                exp_dir, model_type, config, val_loader, device,
                n_bootstrap=args.n_bootstrap,
                max_batches=args.max_batches,
                force=args.force,
            )
            if result is not None:
                success_count += 1
                tqdm.write(f"  {exp_name}: partition_quality={result['partition_quality']:.4f}")
            else:
                skip_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"  {exp_name}: ERROR - {e}")

    print(f"\nCompleted: {success_count} successful, {skip_count} skipped, {error_count} errors")

    # Re-run results analysis
    print("\nUpdating results summary...")
    from core.CRMNIST.grid_search.results import analyze_results
    analyze_results(args.grid_dir, generate_plots=False)


if __name__ == '__main__':
    main()

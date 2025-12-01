#!/usr/bin/env python
"""
Re-run IT analysis for all completed experiments with the fixed evaluator.

The fix filters out collapsed dimensions before MI estimation, which is essential
for accurate measurements when sparsity regularization is used.
"""

import os
import sys
import json
import glob
import torch
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, '/workspace/NVAE')

from core.information_theoretic_evaluation import (
    MinimalInformationPartitionEvaluator,
    extract_latents_from_model,
)


def load_model_and_data(experiment_dir: str, device: str):
    """Load model and create data loader for an experiment."""

    # Load config
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_type = config['model_type']
    params = config['params']

    # Load the model
    model_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        return None, None, None, f"No model found at {model_path}"

    checkpoint = torch.load(model_path, map_location=device)

    # Infer dimensions from checkpoint (they may differ from config)
    if model_type == 'dann_augmented':
        from core.CRMNIST.dann_model import AugmentedDANN

        # Check if model uses separate encoders (look for encoder_zy vs shared_encoder)
        separate_encoders = 'encoder_zy.9.weight' in checkpoint

        if separate_encoders:
            zy_dim = checkpoint['encoder_zy.9.weight'].shape[0]
            zd_dim = checkpoint['encoder_zd.9.weight'].shape[0]
            zdy_dim = checkpoint['encoder_zdy.9.weight'].shape[0]
        else:
            # Shared encoder - infer from projection layers
            zy_dim = checkpoint['projection_zy.weight'].shape[0]
            zd_dim = checkpoint['projection_zd.weight'].shape[0]
            zdy_dim = checkpoint['projection_zdy.weight'].shape[0]

        # Create class_map for CRMNIST (10 classes)
        class_map = {i: i for i in range(10)}

        model = AugmentedDANN(
            class_map=class_map,
            y_dim=10,
            d_dim=6,
            zy_dim=zy_dim,
            zd_dim=zd_dim,
            zdy_dim=zdy_dim,
            separate_encoders=separate_encoders,
        )
        model.load_state_dict(checkpoint)

    elif model_type in ['nvae', 'diva']:
        from core.CRMNIST.model import VAE

        # Check if model uses separate encoders
        separate_encoders = 'qz.encoder_zy.loc.weight' in checkpoint

        if separate_encoders:
            # Infer dimensions from loc layer output
            zy_dim = checkpoint['qz.encoder_zy.loc.weight'].shape[0]
            zd_dim = checkpoint['qz.encoder_zd.loc.weight'].shape[0]
            zdy_dim = checkpoint.get('qz.encoder_zdy.loc.weight', torch.zeros(8, 1)).shape[0]
            zx_dim = checkpoint.get('qz.encoder_zx.loc.weight', torch.zeros(8, 1)).shape[0]
        else:
            # Shared encoder - use config defaults
            zy_dim = params.get('zy_dim', 8)
            zd_dim = params.get('zd_dim', 8)
            zdy_dim = params.get('zdy_dim', 8)
            zx_dim = params.get('zx_dim', 8)

        # Create class_map for CRMNIST
        class_map = {i: i for i in range(10)}

        model = VAE(
            class_map=class_map,
            diva=(model_type == 'diva'),
            zy_dim=zy_dim,
            zx_dim=zx_dim,
            zdy_dim=zdy_dim,
            zd_dim=zd_dim,
            y_dim=10,
            a_dim=6,
            separate_encoders=separate_encoders,
        )
        model.load_state_dict(checkpoint)
    else:
        return None, None, None, f"Unsupported model type: {model_type}"

    model = model.to(device)
    model.eval()

    # Create data loader (mirroring runner._load_data)
    from core.CRMNIST.data_generation import generate_crmnist_dataset

    spec_path = '/workspace/NVAE/conf/crmnist.json'
    with open(spec_path, 'r') as f:
        spec_data = json.load(f)

    # Prepare domain data (convert string keys to int)
    domain_data = {int(k): v for k, v in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data

    # Add y_c (special digit for unique color) if not present
    if 'y_c' not in spec_data:
        spec_data['y_c'] = 7

    # Update spec with experiment params
    intensity = params.get('intensity', 1.5)
    intensity_decay = params.get('intensity_decay', 1.0)

    # Generate validation dataset
    val_dataset = generate_crmnist_dataset(
        spec_data,
        train=False,
        transform_intensity=intensity,
        transform_decay=intensity_decay,
        use_cache=True,
    )
    # CRITICAL: shuffle=True ensures all domains are sampled when using max_batches
    # Without shuffling, data is ordered by domain and we'd only see domain 0
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=True, num_workers=0
    )

    return model, val_loader, model_type, None


def run_it_analysis(model, data_loader, model_type, device, max_batches=100):
    """Run IT analysis on a model."""

    # Extract latent representations
    z_y, z_d, z_dy, z_x, y_labels, d_labels = extract_latents_from_model(
        model, data_loader, device, max_batches=max_batches
    )

    # Create evaluator with the fixed implementation
    evaluator = MinimalInformationPartitionEvaluator(
        n_neighbors=7,
        n_bootstrap=0,  # Skip bootstrap for speed
        max_dims=30,
        pca_variance=0.95,
        min_variance_threshold=0.01,  # Filter collapsed dimensions
    )

    # Run evaluation
    it_results = evaluator.evaluate_latent_partition(
        z_y, z_d, z_dy, z_x,
        y_labels, d_labels,
        compute_bootstrap=False
    )

    return it_results, evaluator


def main():
    parser = argparse.ArgumentParser(description='Re-run IT analysis with fixed evaluator')
    parser.add_argument('--experiments-dir', type=str,
                        default='/workspace/NVAE/grid_search_results_separate_encoders/experiments',
                        help='Directory containing experiments')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Maximum batches for latent extraction')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process experiments matching this pattern')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find all experiments
    experiment_dirs = sorted(glob.glob(os.path.join(args.experiments_dir, '*/')))

    if args.filter:
        experiment_dirs = [d for d in experiment_dirs if args.filter in d]

    print(f"Found {len(experiment_dirs)} experiments to process")

    # Process each experiment
    success_count = 0
    error_count = 0
    skipped_count = 0

    for exp_dir in tqdm(experiment_dirs, desc="Re-running IT analysis"):
        exp_name = os.path.basename(exp_dir.rstrip('/'))

        try:
            # Check if model exists
            model_path = os.path.join(exp_dir, 'best_model.pt')
            if not os.path.exists(model_path):
                skipped_count += 1
                continue

            # Load model and data
            model, val_loader, model_type, error = load_model_and_data(exp_dir, device)
            if error:
                if args.verbose:
                    print(f"\n  {exp_name}: {error}")
                skipped_count += 1
                continue

            # Suppress output unless verbose
            if not args.verbose:
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    it_results, evaluator = run_it_analysis(
                        model, val_loader, model_type, device, args.max_batches
                    )
            else:
                print(f"\n{'='*60}")
                print(f"Processing: {exp_name}")
                print(f"{'='*60}")
                it_results, evaluator = run_it_analysis(
                    model, val_loader, model_type, device, args.max_batches
                )

            # Save results
            it_path = os.path.join(exp_dir, 'it_analysis.json')
            evaluator.save_results(it_results, it_path)

            # Also update metrics.json with new IT metrics
            metrics_path = os.path.join(exp_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                # Update IT metrics
                it_metrics = it_results.get('metrics', {})
                metrics['it_metrics'] = {
                    'I_zy_Y_given_D': it_metrics.get('I(z_y;Y|D)', 0.0),
                    'I_zy_D_given_Y': it_metrics.get('I(z_y;D|Y)', 0.0),
                    'z_y_specificity': it_metrics.get('z_y_specificity', 0.0),
                    'I_zd_D_given_Y': it_metrics.get('I(z_d;D|Y)', 0.0),
                    'I_zd_Y_given_D': it_metrics.get('I(z_d;Y|D)', 0.0),
                    'z_d_specificity': it_metrics.get('z_d_specificity', 0.0),
                    'I_zdy_Y_D': it_metrics.get('I(z_dy;Y;D)', 0.0),
                    'I_zdy_joint': it_metrics.get('I(z_dy;Y,D)', 0.0),
                    'partition_quality': it_metrics.get('partition_quality', 0.0),
                }

                # Add effective dimensionality info
                eff_dim = it_results.get('effective_dimensionality', {})
                metrics['effective_dimensionality'] = {
                    name: {
                        'active': info['active_dims'] if info else 0,
                        'nominal': info['nominal_dims'] if info else 0,
                        'utilization': info['utilization'] if info else 0.0,
                    }
                    for name, info in eff_dim.items() if info is not None
                }

                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

            success_count += 1

        except Exception as e:
            error_count += 1
            if args.verbose:
                import traceback
                print(f"\n  Error processing {exp_name}: {e}")
                traceback.print_exc()
            else:
                tqdm.write(f"Error: {exp_name}: {str(e)[:50]}")

    print(f"\n{'='*60}")
    print(f"IT Analysis Re-run Complete")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

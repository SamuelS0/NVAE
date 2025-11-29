#!/usr/bin/env python
"""
CRMNIST Hyperparameter Grid Search

Main entry point for running grid search experiments across all models.
Includes Information-Theoretic (IT) analysis of latent space partitioning.

Usage:
    # Run full grid search with IT analysis (default)
    python -m core.CRMNIST.run_grid_search --out results/grid_search/

    # Run quick screening (reduced configs)
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --quick

    # Run specific models only
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --models nvae diva

    # Resume interrupted search
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --resume

    # Run without IT analysis (faster)
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --no-it-analysis

    # Run with IT bootstrap confidence intervals
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --it-bootstrap 50

    # Analyze existing results only
    python -m core.CRMNIST.run_grid_search --out results/grid_search/ --analyze-only

    # List available configurations
    python -m core.CRMNIST.run_grid_search --list-configs

Note: IT analysis is only computed for models with explicit latent partitioning
(NVAE, DIVA, AugmentedDANN). DANN and IRM use monolithic representations and
do not support IT analysis.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(
        description='CRMNIST Hyperparameter Grid Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--out', '-o',
        type=str,
        default='results/grid_search/',
        help='Output directory for results (default: results/grid_search/)'
    )

    # Mode selection
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick screening with reduced configurations'
    )
    parser.add_argument(
        '--analyze-only', '-a',
        action='store_true',
        help='Only analyze existing results, do not run experiments'
    )
    parser.add_argument(
        '--list-configs', '-l',
        action='store_true',
        help='List all available configurations and exit'
    )

    # Model selection
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        type=str,
        choices=['nvae', 'diva', 'dann', 'dann_augmented', 'irm'],
        help='Model types to include (default: all)'
    )

    # Experiment options
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume interrupted grid search, skip completed experiments'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detects if not specified.'
    )

    # Specific configuration
    parser.add_argument(
        '--config-name', '-c',
        type=str,
        help='Run a specific configuration by name'
    )

    # Preset filters (for targeted search)
    parser.add_argument(
        '--sparsity',
        type=str,
        choices=['none', 'zdy_light', 'low', 'medium', 'medium_high', 'high', 'very_high',
                 'zd_high', 'balanced', 'zdy_focus'],
        help='Filter by sparsity preset (NVAE/DIVA: none/zdy_light/low/medium/medium_high/high/very_high, '
             'AugmentedDANN: also zd_high/balanced/zdy_focus)'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['low', 'medium', 'high'],
        help='Filter by classifier preset'
    )
    parser.add_argument(
        '--kl',
        type=str,
        choices=['low', 'medium_low', 'medium', 'high'],
        help='Filter by KL preset (NVAE/DIVA only)'
    )

    # Visualization options
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress information (default: verbose)'
    )

    # Information-Theoretic analysis options
    parser.add_argument(
        '--no-it-analysis',
        action='store_true',
        help='Skip Information-Theoretic analysis after training'
    )
    parser.add_argument(
        '--it-bootstrap',
        type=int,
        default=0,
        help='Number of bootstrap samples for IT confidence intervals (default: 0, no bootstrap)'
    )
    parser.add_argument(
        '--it-max-batches',
        type=int,
        default=100,
        help='Maximum batches to use for IT analysis (default: 100)'
    )
    parser.add_argument(
        '--separate-encoders',
        action='store_true',
        help='Use separate CNN encoders for each latent space (achieves true I(Z_i; Z_j) = 0)'
    )

    return parser.parse_args()


def list_configurations():
    """Print all available configurations."""
    from core.CRMNIST.grid_search.config import print_config_summary
    print_config_summary()


def filter_configs(configs, args):
    """Filter configurations based on command line arguments."""
    filtered = configs

    if args.sparsity:
        filtered = [c for c in filtered if c.get('preset_names', {}).get('sparsity') == args.sparsity]

    if args.classifier:
        filtered = [c for c in filtered if c.get('preset_names', {}).get('classifier') == args.classifier]

    if args.kl:
        filtered = [c for c in filtered if c.get('preset_names', {}).get('kl') == args.kl]

    return filtered


def main():
    args = parse_args()

    # Handle list configs
    if args.list_configs:
        list_configurations()
        return

    # Handle analyze only
    if args.analyze_only:
        from core.CRMNIST.grid_search.results import analyze_results
        print(f"Analyzing results in {args.out}")
        analyze_results(args.out, generate_plots=not args.no_plots)
        return

    # Handle specific configuration
    if args.config_name:
        from core.CRMNIST.grid_search.runner import run_single_config
        print(f"Running configuration: {args.config_name}")
        result = run_single_config(
            args.config_name,
            args.out,
            models=args.models,
            device=args.device
        )
        if result:
            print(f"Result: {result}")
        return

    # Get configurations
    if args.quick:
        from core.CRMNIST.grid_search.config import get_quick_configs
        configs = get_quick_configs(args.models)
    else:
        from core.CRMNIST.grid_search.config import get_all_configs
        configs = get_all_configs(args.models)

    # Apply filters
    configs = filter_configs(configs, args)

    if not configs:
        print("No configurations match the specified filters!")
        return

    # Apply separate_encoders flag if specified
    if args.separate_encoders:
        print("Using separate encoders for each latent space (I(Z_i; Z_j) = 0 architecture)")
        for config in configs:
            config['params']['separate_encoders'] = True

    print(f"Running {len(configs)} configurations...")

    # Run grid search
    from core.CRMNIST.grid_search.runner import GridSearchRunner

    runner = GridSearchRunner(
        output_dir=args.out,
        device=args.device,
        resume=args.resume,
        verbose=not args.quiet,  # Default to verbose, use --quiet to suppress
        enable_it_analysis=not args.no_it_analysis,
        it_n_bootstrap=args.it_bootstrap,
        it_max_batches=args.it_max_batches,
    )

    results = runner.run_grid_search(configs=configs)

    # Analyze results
    if not args.no_plots:
        from core.CRMNIST.grid_search.results import analyze_results
        print("\nAnalyzing results...")
        analyze_results(args.out, generate_plots=True)


if __name__ == '__main__':
    main()

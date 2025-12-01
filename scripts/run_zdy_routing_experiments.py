#!/usr/bin/env python3
"""
Run 3 targeted MP-DANN experiments to test Z_DY routing hypotheses.

These settings are designed to fix the Z_DY information routing problem identified
in the analysis of the 10-epoch grid search results.

Experiment 1: "Gentle Separation"
    - Equal light sparsity (0.5) on all spaces
    - Ultra-gentle adversarial (beta=0.1, gamma=2.0)
    - Tests: Does minimal adversarial prevent info destruction?

Experiment 2: "Z_DY Preferred"
    - Z_DY has LOWER penalty (0.25) than Z_Y/Z_D (1.5)
    - First preset where sparsity_zdy < sparsity_zy
    - Tests: Does giving Z_DY relative advantage enable proper routing?

Experiment 3: "Class-Focused"
    - Asymmetric classifier (alpha_y=150, alpha_d=25)
    - Light sparsity on class spaces, heavier on domain
    - Tests: Does prioritizing class over domain help OOD?
"""

import sys
import os
sys.path.insert(0, '/workspace/NVAE')

import argparse
import torch
from datetime import datetime
from pathlib import Path

from core.CRMNIST.grid_search.config import (
    FIXED_PARAMS,
    DANN_AUG_SPARSITY_PRESETS,
    DANN_AUG_ADVERSARIAL_PRESETS,
    CLASSIFIER_PRESETS,
    merge_configs
)
from core.CRMNIST.grid_search.runner import GridSearchRunner


# Define the 3 targeted experiments
EXPERIMENTS = [
    {
        'name': 'dann_aug_zdy_routing_gentle_separation',
        'description': 'Gentle Separation: Equal light sparsity, ultra-gentle adversarial',
        'sparsity_preset': 'gentle_equal',
        'adversarial_preset': 'ultra_gentle',
        'classifier_preset': 'high',
    },
    {
        'name': 'dann_aug_zdy_routing_zdy_preferred',
        'description': 'Z_DY Preferred: Z_DY has lower penalty than Z_Y/Z_D',
        'sparsity_preset': 'zdy_preferred',
        'adversarial_preset': 'low',
        'classifier_preset': 'high',
    },
    {
        'name': 'dann_aug_zdy_routing_class_focused',
        'description': 'Class-Focused: Asymmetric classifier (alpha_y >> alpha_d)',
        'sparsity_preset': 'class_focused',
        'adversarial_preset': 'low',
        'classifier_preset': 'class_priority',
    },
]


def build_config(experiment):
    """Build configuration for an experiment."""
    sparsity_params = DANN_AUG_SPARSITY_PRESETS[experiment['sparsity_preset']]
    adversarial_params = DANN_AUG_ADVERSARIAL_PRESETS[experiment['adversarial_preset']]
    classifier_params = CLASSIFIER_PRESETS[experiment['classifier_preset']]

    params = merge_configs(
        FIXED_PARAMS.copy(),
        sparsity_params,
        adversarial_params,
        classifier_params,
    )

    return {
        'name': experiment['name'],
        'model_type': 'dann_augmented',
        'preset_names': {
            'sparsity': experiment['sparsity_preset'],
            'adversarial': experiment['adversarial_preset'],
            'classifier': experiment['classifier_preset'],
        },
        'params': params,
    }


def main():
    parser = argparse.ArgumentParser(description='Run Z_DY routing experiments')
    parser.add_argument('--out', type=str, default='results/zdy_routing_experiments',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--separate-encoders', action='store_true', default=True,
                        help='Use separate encoders (default: True)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Z_DY ROUTING EXPERIMENTS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Separate encoders: {args.separate_encoders}")
    print(f"Epochs: {args.epochs}")
    print()

    # Run each experiment
    results = []
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/3: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"{'='*80}")

        config = build_config(experiment)

        # Override epochs and separate_encoders
        config['params']['epochs'] = args.epochs
        config['params']['separate_encoders'] = args.separate_encoders

        print(f"\nConfiguration:")
        print(f"  Sparsity: zdy={config['params']['sparsity_weight_zdy']}, "
              f"zy={config['params']['sparsity_weight_zy']}, "
              f"zd={config['params']['sparsity_weight_zd']}")
        print(f"  Adversarial: beta_adv={config['params']['beta_adv']}, "
              f"gamma={config['params']['lambda_schedule_gamma']}")
        print(f"  Classifier: alpha_y={config['params']['alpha_y']}, "
              f"alpha_d={config['params']['alpha_d']}")
        print()

        try:
            runner = GridSearchRunner(
                output_dir=str(output_dir),
                device=args.device,
                verbose=True,
                enable_it_analysis=True,
            )
            result = runner.run_experiment(config)
            results.append(result)

            print(f"\nResult for {experiment['name']}:")
            print(f"  Test Accuracy: {result.get('test_accuracy', 'N/A'):.4f}")
            print(f"  OOD Accuracy: {result.get('ood_accuracy', 'N/A'):.4f}")
            print(f"  Partition Quality: {result.get('it_partition_quality', 'N/A'):.4f}")
            print(f"  I(Z_DY; Y,D): {result.get('it_I_zdy_joint', 'N/A'):.4f}")

        except Exception as e:
            print(f"ERROR in experiment {experiment['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': experiment['name'], 'error': str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF Z_DY ROUTING EXPERIMENTS")
    print("=" * 80)
    print(f"{'Experiment':<45} {'Test Acc':>10} {'OOD Acc':>10} {'PQ':>10} {'I(ZDY;Y,D)':>12}")
    print("-" * 87)

    for result in results:
        name = result.get('name', 'Unknown')[:44]
        if 'error' in result:
            print(f"{name:<45} {'ERROR':>10}")
        else:
            test_acc = result.get('test_accuracy', 0)
            ood_acc = result.get('ood_accuracy', 0)
            pq = result.get('it_partition_quality', 0)
            i_zdy = result.get('it_I_zdy_joint', 0)
            print(f"{name:<45} {test_acc:>10.4f} {ood_acc:>10.4f} {pq:>10.4f} {i_zdy:>12.4f}")

    print("\nExperiments complete!")


if __name__ == '__main__':
    main()

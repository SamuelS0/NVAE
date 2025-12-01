#!/usr/bin/env python
"""
Run decorrelation experiments to test if the decorrelation loss can fix the Z_DY routing problem.

The decorrelation loss penalizes the cross-covariance between Z_DY and (Z_Y, Z_D),
forcing Z_DY to capture non-redundant information that is not already in Z_Y or Z_D.

We test 3 different decorrelation strengths:
- Light (beta_decorr=0.1): Gentle nudge towards non-redundancy
- Medium (beta_decorr=0.5): Balanced decorrelation
- Strong (beta_decorr=1.0): Strong enforcement of non-redundancy

Base configuration uses the best-performing setup from previous experiments:
- Sparsity: gentle_equal (0.5, 0.5, 0.5)
- Adversarial: ultra_gentle (beta_adv=0.1, gamma=2.0)
- Classifier: high (alpha_y=100, alpha_d=100)
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.CRMNIST.grid_search.runner import GridSearchRunner
from core.CRMNIST.grid_search.config import (
    FIXED_PARAMS,
    DANN_AUG_SPARSITY_PRESETS,
    DANN_AUG_ADVERSARIAL_PRESETS,
    DANN_AUG_DECORR_PRESETS,
    CLASSIFIER_PRESETS,
    merge_configs
)
import copy


# Define experiments: 3 decorrelation strengths
EXPERIMENTS = [
    {
        'name': 'dann_aug_decorr_light',
        'sparsity_preset': 'gentle_equal',
        'adversarial_preset': 'ultra_gentle',
        'classifier_preset': 'high',
        'decorr_preset': 'light',  # beta_decorr = 0.1
    },
    {
        'name': 'dann_aug_decorr_medium',
        'sparsity_preset': 'gentle_equal',
        'adversarial_preset': 'ultra_gentle',
        'classifier_preset': 'high',
        'decorr_preset': 'medium',  # beta_decorr = 0.5
    },
    {
        'name': 'dann_aug_decorr_strong',
        'sparsity_preset': 'gentle_equal',
        'adversarial_preset': 'ultra_gentle',
        'classifier_preset': 'high',
        'decorr_preset': 'strong',  # beta_decorr = 1.0
    },
]


def create_experiment_config(exp_dict, epochs, separate_encoders=True):
    """Create a full configuration dict for an experiment."""
    # Start with fixed params
    params = copy.deepcopy(FIXED_PARAMS)
    params['epochs'] = epochs
    params['separate_encoders'] = separate_encoders

    # Merge preset parameters
    params.update(DANN_AUG_SPARSITY_PRESETS[exp_dict['sparsity_preset']])
    params.update(DANN_AUG_ADVERSARIAL_PRESETS[exp_dict['adversarial_preset']])
    params.update(CLASSIFIER_PRESETS[exp_dict['classifier_preset']])
    params.update(DANN_AUG_DECORR_PRESETS[exp_dict['decorr_preset']])

    return {
        'name': exp_dict['name'],
        'model_type': 'dann_augmented',
        'preset_names': {
            'sparsity': exp_dict['sparsity_preset'],
            'adversarial': exp_dict['adversarial_preset'],
            'classifier': exp_dict['classifier_preset'],
            'decorrelation': exp_dict['decorr_preset'],
        },
        'params': params,
    }


def main():
    parser = argparse.ArgumentParser(description='Run decorrelation experiments for Z_DY routing')
    parser.add_argument('--out', type=str, default='results/decorrelation_experiments',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--shared-encoders', action='store_true',
                        help='Use shared encoders instead of separate (default: separate)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Create runner
    runner = GridSearchRunner(
        output_dir=args.out,
        device=args.device,
    )

    print("=" * 70)
    print("DECORRELATION EXPERIMENTS FOR Z_DY ROUTING")
    print("=" * 70)
    print(f"\nTesting decorrelation loss to fix Z_DY routing problem")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Separate encoders: {not args.shared_encoders}")
    print(f"Output: {args.out}")
    print(f"\nRunning {len(EXPERIMENTS)} experiments:")
    for exp in EXPERIMENTS:
        print(f"  - {exp['name']}: decorr={exp['decorr_preset']}")
    print()

    # Run experiments
    results = []
    for i, exp_dict in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}: {exp_dict['name']}")
        print(f"{'='*70}")

        config = create_experiment_config(
            exp_dict,
            epochs=args.epochs,
            separate_encoders=not args.shared_encoders
        )

        print(f"\nConfiguration:")
        print(f"  Sparsity: {exp_dict['sparsity_preset']}")
        print(f"  Adversarial: {exp_dict['adversarial_preset']}")
        print(f"  Classifier: {exp_dict['classifier_preset']}")
        print(f"  Decorrelation: {exp_dict['decorr_preset']} (beta_decorr={config['params'].get('beta_decorr', 0.0)})")

        try:
            result = runner.run_experiment(config)
            results.append({
                'experiment': exp_dict['name'],
                'config': config,
                'result': result,
                'status': 'success',
            })

            # Print summary
            if 'eval_metrics' in result:
                print(f"\nResults:")
                print(f"  Test Accuracy: {result['eval_metrics'].get('test_accuracy', 'N/A'):.4f}")
                print(f"  ID Accuracy:   {result['eval_metrics'].get('id_accuracy', 'N/A'):.4f}")
                print(f"  OOD Accuracy:  {result['eval_metrics'].get('ood_accuracy', 'N/A'):.4f}")
                print(f"  Gen Gap:       {result['eval_metrics'].get('gen_gap', 'N/A'):.4f}")

            if 'it_metrics' in result:
                print(f"\nIT Metrics:")
                print(f"  I(Z_Y;Y|D):    {result['it_metrics'].get('I_zy_Y_given_D', 'N/A'):.4f}")
                print(f"  I(Z_D;D|Y):    {result['it_metrics'].get('I_zd_D_given_Y', 'N/A'):.4f}")
                print(f"  I(Z_DY;Y;D):   {result['it_metrics'].get('I_zdy_Y_D', 'N/A'):.4f}")
                print(f"  I(Z_DY;Y,D):   {result['it_metrics'].get('I_zdy_joint', 'N/A'):.4f}")
                print(f"  Partition Q:   {result['it_metrics'].get('partition_quality', 'N/A'):.4f}")

        except Exception as e:
            print(f"\nERROR: Experiment failed with: {e}")
            results.append({
                'experiment': exp_dict['name'],
                'config': config,
                'error': str(e),
                'status': 'failed',
            })

    # Save summary
    summary_path = os.path.join(args.out, 'experiment_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': args.device,
        'epochs': args.epochs,
        'separate_encoders': not args.shared_encoders,
        'experiments': [
            {
                'name': r['experiment'],
                'status': r['status'],
                'test_accuracy': r.get('result', {}).get('eval_metrics', {}).get('test_accuracy') if r['status'] == 'success' else None,
                'ood_accuracy': r.get('result', {}).get('eval_metrics', {}).get('ood_accuracy') if r['status'] == 'success' else None,
                'I_zdy_Y_D': r.get('result', {}).get('it_metrics', {}).get('I_zdy_Y_D') if r['status'] == 'success' else None,
                'I_zdy_joint': r.get('result', {}).get('it_metrics', {}).get('I_zdy_joint') if r['status'] == 'success' else None,
                'partition_quality': r.get('result', {}).get('it_metrics', {}).get('partition_quality') if r['status'] == 'success' else None,
            }
            for r in results
        ]
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Experiment':<30} {'Test Acc':>10} {'OOD Acc':>10} {'I(ZDY;Y;D)':>12} {'I(ZDY;Y,D)':>12}")
    print("-" * 74)

    for r in results:
        if r['status'] == 'success':
            test_acc = r['result']['eval_metrics'].get('test_accuracy', 0)
            ood_acc = r['result']['eval_metrics'].get('ood_accuracy', 0)
            i_zdy_yd = r['result']['it_metrics'].get('I_zdy_Y_D', 0)
            i_zdy_joint = r['result']['it_metrics'].get('I_zdy_joint', 0)
            print(f"{r['experiment']:<30} {test_acc:>10.4f} {ood_acc:>10.4f} {i_zdy_yd:>12.4f} {i_zdy_joint:>12.4f}")
        else:
            print(f"{r['experiment']:<30} {'FAILED':>10}")

    print(f"\nSummary saved to: {summary_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()

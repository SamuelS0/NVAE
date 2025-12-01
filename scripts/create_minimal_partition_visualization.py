#!/usr/bin/env python3
"""
Minimal Partition Adherence Visualization

Creates a visualization showing models' adherence to the Minimally Partitioned
Representation definition:

Definition: A representation Z = (Z_Y, Z_D, Z_{YD}, Z_X) is minimally partitioned if:
1. Z_Y captures class-specific: I(Z_Y;D|Y) = 0
2. Z_D captures domain-specific: I(Z_D;Y|D) = 0
3. Z_{YD} captures shared info: predictive of both Y and D
4. Z_X captures residual: I(Z_X;Y,D) = 0

Features:
- Top 3 per model type (9 total) ranked by partition quality
- Same 9 models tracked across all panels
- DANN/IRM baselines shown in domain leakage panel (without top 3 indicators)

Usage:
    python create_minimal_partition_visualization.py [--results_dir PATH] [--output_name NAME]
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path


# Default paths
DEFAULT_RESULTS_DIR = '/workspace/NVAE/results/grid_search_10epochs_separate'
DEFAULT_BASELINE_DIR = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'
DEFAULT_OUTPUT_NAME = 'minimal_partition_adherence.png'

# Model type colors
MODEL_COLORS = {
    'nvae': '#9467BD',           # Purple
    'diva': '#FF7F0E',           # Orange
    'dann_augmented': '#17BECF'  # Cyan
}

# Rank markers and colors
RANK_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
RANK_MARKERS = ['o', 's', '^']  # Circle, Square, Triangle

# Baseline colors
BASELINE_COLORS = {
    'dann': '#555555',  # Dark gray
    'irm': '#888888'    # Medium gray
}


def load_baseline_metrics(baseline_dir: str) -> dict:
    """
    Load DANN/IRM baseline metrics from JSON files.

    Args:
        baseline_dir: Path to information_theoretic_analysis directory

    Returns:
        Dict mapping model name to metrics dict
    """
    baselines = {}
    baseline_path = Path(baseline_dir)

    for model in ['dann', 'irm']:
        results_file = baseline_path / model / 'it_results.json'
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            metrics = data.get('metrics', {})
            baselines[model] = {
                'domain_leakage': metrics.get('I(Z;D|Y)', 0),  # I(Z_Y;D|Y) equivalent
                'class_info': metrics.get('I(Z;Y|D)', 0),
            }
            print(f"  Loaded {model.upper()}: I(Z;D|Y) = {baselines[model]['domain_leakage']:.4f}")

    return baselines


def get_top3_per_model_type(df: pd.DataFrame, metric: str = 'it_partition_quality') -> dict:
    """
    Get top 3 models per model type by specified metric.

    Args:
        df: DataFrame with results
        metric: Column name to rank by

    Returns:
        Dict mapping model_type to DataFrame of top 3
    """
    top_models = {}
    for model_type in ['dann_augmented', 'diva', 'nvae']:
        mt_df = df[df['model_type'] == model_type].copy()
        mt_df = mt_df.dropna(subset=[metric])
        top_models[model_type] = mt_df.nlargest(3, metric)
    return top_models


def create_visualization(
    results_dir: str,
    baseline_dir: str = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    model_types: list = None
):
    """
    Create the minimal partition adherence visualization.

    Args:
        results_dir: Directory containing grid search results
        baseline_dir: Directory containing DANN/IRM baseline results
        output_name: Output filename
        model_types: List of model types to include (default: dann_augmented, diva, nvae)
    """
    if model_types is None:
        model_types = ['dann_augmented', 'nvae', 'diva']

    if baseline_dir is None:
        baseline_dir = DEFAULT_BASELINE_DIR

    results_path = Path(results_dir)
    csv_path = results_path / 'summary' / 'all_results.csv'
    output_path = results_path / 'summary' / output_name

    # Load data
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter to decomposed models
    df = df[df['model_type'].isin(model_types)]
    df = df.dropna(subset=['it_partition_quality'])
    print(f"Loaded {len(df)} experiments")
    print(f"Model counts: {df['model_type'].value_counts().to_dict()}")

    # Get top 3 per model type by partition quality
    print("\nGetting top 3 per model type by partition quality...")
    top3_per_type = get_top3_per_model_type(df, 'it_partition_quality')

    # Collect all top 3 indices for tracking across panels
    top3_indices = {}
    for mt, top_df in top3_per_type.items():
        top3_indices[mt] = top_df.index.tolist()

    # Print top 3 summary
    print("\n" + "="*80)
    print("TOP 3 PER MODEL TYPE (by Partition Quality)")
    print("="*80)
    for mt in model_types:
        print(f"\n{mt.upper()}:")
        for rank, (idx, row) in enumerate(top3_per_type[mt].iterrows(), 1):
            print(f"  #{rank}: {row['name'][:50]}")
            print(f"      PQ={row['it_partition_quality']:.4f}, "
                  f"I(Z_Y;D|Y)={row.get('it_I_zy_D_given_Y', 'N/A')}, "
                  f"Acc={row['test_accuracy']:.4f}")

    # Load baselines
    print("\nLoading DANN/IRM baselines...")
    baselines = load_baseline_metrics(baseline_dir)

    # Panel definitions: (metric_col, title, subtitle, lower_is_better, show_baselines)
    panels = [
        ('it_I_zy_D_given_Y', r'Criterion 1: $I(Z_Y;D|Y) \rightarrow 0$',
         'Domain leakage in class space (lower = better)', True, True),
        ('it_I_zd_Y_given_D', r'Criterion 2: $I(Z_D;Y|D) \rightarrow 0$',
         'Class leakage in domain space (lower = better)', True, False),
        ('it_I_zdy_joint', r'Criterion 3: $Z_{YD}$ captures shared info',
         'Joint information I(Z_YD;Y,D) (higher = better)', False, False),
        ('it_partition_quality', 'Criterion 4: Overall Partition Quality',
         'Combined adherence score (higher = better)', False, False),
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for panel_idx, (metric, title, subtitle, lower_is_better, show_baselines) in enumerate(panels):
        ax = axes[panel_idx]
        pos = 0
        x_ticks = []
        x_labels = []

        for mt in model_types:
            # Skip DIVA for Z_YD panel (criterion 3) since DIVA has no z_dy space
            if metric == 'it_I_zdy_joint' and mt == 'diva':
                continue

            mt_data = df[df['model_type'] == mt][metric].dropna()
            if len(mt_data) == 0:
                continue

            # Calculate quartiles
            q1 = mt_data.quantile(0.25)
            q2 = mt_data.quantile(0.50)
            q3 = mt_data.quantile(0.75)
            min_val = mt_data.min()
            max_val = mt_data.max()

            box_width = 0.6
            whisker_width = 0.3

            # Draw box plot
            rect = plt.Rectangle(
                (pos - box_width/2, q1), box_width, q3-q1,
                fill=True, facecolor=MODEL_COLORS.get(mt, '#888888'),
                edgecolor='black', alpha=0.4, linewidth=1
            )
            ax.add_patch(rect)
            ax.hlines(q2, pos - box_width/2, pos + box_width/2, colors='black', linewidth=2)
            ax.vlines(pos, min_val, q1, colors='black', linewidth=1)
            ax.vlines(pos, q3, max_val, colors='black', linewidth=1)
            ax.hlines(min_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)
            ax.hlines(max_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)

            # Plot top 3 for this model type
            if mt in top3_indices:
                for rank, idx in enumerate(top3_indices[mt]):
                    row = df.loc[idx]
                    value = row[metric]
                    if pd.isna(value):
                        continue
                    jitter = (rank - 1) * 0.15
                    ax.scatter(
                        pos + jitter, value,
                        c=RANK_COLORS[rank], s=200,
                        marker=RANK_MARKERS[rank],
                        edgecolors='black', linewidths=2, zorder=10
                    )
                    ax.annotate(
                        f'#{rank+1}', (pos + jitter, value),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=10, fontweight='bold',
                        color=RANK_COLORS[rank]
                    )

            x_ticks.append(pos)
            display_name = {'dann_augmented': 'MP-DANN', 'nvae': 'MP-VAE', 'diva': 'DIVA'}
            x_labels.append(display_name.get(mt, mt.upper()))
            pos += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add ideal target line for leakage metrics
        if lower_is_better:
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7)

        # Add DANN/IRM baselines to domain leakage panel
        if show_baselines and baselines:
            for bm_name, bm_metrics in baselines.items():
                value = bm_metrics['domain_leakage']
                ax.axhline(
                    y=value, color=BASELINE_COLORS.get(bm_name, 'gray'),
                    linestyle='--', linewidth=2, alpha=0.8, zorder=5
                )
                # Add label on right side
                x_right = ax.get_xlim()[1]
                ax.annotate(
                    f'{bm_name.upper()}: {value:.2f}',
                    xy=(x_right, value), xytext=(5, 0),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=BASELINE_COLORS.get(bm_name, 'gray'),
                    va='center', ha='left'
                )

        # Auto-scale y-axis
        all_vals = df[metric].dropna().tolist()
        if show_baselines and baselines:
            for bm in baselines.values():
                all_vals.append(bm['domain_leakage'])

        if len(all_vals) > 0:
            ymin, ymax = min(all_vals), max(all_vals)
            yrange = ymax - ymin if ymax != ymin else 1
            if lower_is_better:
                ax.set_ylim(-0.05 * yrange, ymax + 0.2 * yrange)
            else:
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.15 * yrange)

    # Create legend with 3 columns - each section vertical within its column
    from matplotlib.patches import FancyBboxPatch

    # Column 1: Top 3 markers (vertical)
    col1_elements = []
    col1_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 per Type:'))
    for rank in range(3):
        col1_elements.append(plt.scatter(
            [], [], c=RANK_COLORS[rank], s=120,
            marker=RANK_MARKERS[rank], edgecolors='black',
            label=f'#{rank+1} Best'
        ))

    # Column 2: Model types (vertical)
    col2_elements = []
    col2_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Model Types:'))
    display_names = {'dann_augmented': 'MP-DANN', 'nvae': 'MP-VAE', 'diva': 'DIVA'}
    for mt in model_types:
        col2_elements.append(mpatches.Patch(
            facecolor=MODEL_COLORS[mt], edgecolor='black', alpha=0.4,
            label=display_names.get(mt, mt.upper())
        ))

    # Column 3: Baselines (vertical)
    col3_elements = []
    if baselines:
        col3_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Baselines:'))
        for bm_name in baselines:
            col3_elements.append(Line2D(
                [0], [0], color=BASELINE_COLORS.get(bm_name, 'gray'),
                linestyle='--', linewidth=2, label=bm_name.upper()
            ))

    # Create three separate column legends - closer horizontally
    leg1 = fig.legend(
        handles=col1_elements, loc='lower left', bbox_to_anchor=(0.32, 0.005),
        ncol=1, fontsize=9, frameon=False, labelspacing=0.3
    )
    leg2 = fig.legend(
        handles=col2_elements, loc='lower center', bbox_to_anchor=(0.5, 0.005),
        ncol=1, fontsize=9, frameon=False, labelspacing=0.3
    )
    if baselines:
        leg3 = fig.legend(
            handles=col3_elements, loc='lower right', bbox_to_anchor=(0.68, 0.005),
            ncol=1, fontsize=9, frameon=False, labelspacing=0.3
        )

    # Add a background box around all legends
    legend_bg = FancyBboxPatch(
        (0.26, 0.002), 0.48, 0.058,
        boxstyle="round,pad=0.01", facecolor='white', edgecolor='gray',
        alpha=0.9, transform=fig.transFigure, zorder=0
    )
    fig.patches.append(legend_bg)

    fig.suptitle(
        'Minimal Partition Adherence: Evaluation by Definition Criteria\n'
        'Box plots show quartile distribution; markers show top 3 per model type by partition quality\n'
        '(10 Epochs, Separate Encoders)',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")

    return df, top3_per_type, baselines


def main():
    parser = argparse.ArgumentParser(
        description='Create Minimal Partition Adherence Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
        help=f'Directory containing grid search results (default: {DEFAULT_RESULTS_DIR})'
    )
    parser.add_argument(
        '--baseline_dir', type=str, default=DEFAULT_BASELINE_DIR,
        help=f'Directory containing DANN/IRM baselines (default: {DEFAULT_BASELINE_DIR})'
    )
    parser.add_argument(
        '--output_name', type=str, default=DEFAULT_OUTPUT_NAME,
        help=f'Output filename (default: {DEFAULT_OUTPUT_NAME})'
    )
    parser.add_argument(
        '--model_types', type=str, nargs='+',
        default=['dann_augmented', 'nvae', 'diva'],
        help='Model types to include (default: dann_augmented nvae diva)'
    )

    args = parser.parse_args()

    create_visualization(
        results_dir=args.results_dir,
        baseline_dir=args.baseline_dir,
        output_name=args.output_name,
        model_types=args.model_types
    )


if __name__ == '__main__':
    main()

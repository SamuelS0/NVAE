#!/usr/bin/env python3
"""
Create visualization showing top 3 models PER MODEL TYPE tracked across all IT metrics.
Each model maintains consistent color across all subplots for easy comparison.
Includes DANN/IRM baseline reference lines for applicable metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import json
import sys
sys.path.insert(0, '/workspace/NVAE')
from core.disentanglement_quality import compute_dqs


# Baseline model colors and styles - gray tones so they don't stand out too much
BASELINE_COLORS = {
    'dann': '#555555',  # Dark gray
    'irm': '#888888',   # Medium gray
}

BASELINE_LABELS = {
    'dann': 'DANN',
    'irm': 'IRM',
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load grid search results and compute DQS."""
    df = pd.read_csv(csv_path)
    # Filter out rows with missing partition quality
    df = df.dropna(subset=['it_partition_quality'])

    # Compute DQS for each model
    dqs_scores = []
    for _, row in df.iterrows():
        metrics = {
            'I(z_y;Y|D)': row.get('it_I_zy_Y_given_D', 0) or 0,
            'I(z_y;D|Y)': row.get('it_I_zy_D_given_Y', 0) or 0,
            'I(z_d;D|Y)': row.get('it_I_zd_D_given_Y', 0) or 0,
            'I(z_d;Y|D)': row.get('it_I_zd_Y_given_D', 0) or 0,
            'I(z_dy;Y;D)': row.get('it_I_zdy_Y_D', 0) or 0,
            'I(z_x;Y,D)': row.get('it_I_zx_YD', 0) or 0,
        }
        result = compute_dqs(metrics)
        dqs_scores.append(result['dqs'])
    df['dqs'] = dqs_scores

    return df


def load_baseline_results(baseline_dir: str = None) -> dict:
    """Load DANN/IRM baseline results for reference lines.

    Returns dict mapping model name to metrics dict.
    """
    if baseline_dir is None:
        baseline_dir = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'

    baselines = {}
    baseline_path = Path(baseline_dir)

    for model in ['dann', 'irm']:
        results_file = baseline_path / model / 'it_results.json'
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            # Map monolithic metrics to decomposed-style names for comparison
            metrics = data.get('metrics', {})
            unified = data.get('unified_metrics', {})

            baselines[model] = {
                # Class capture: I(Z;Y|D) maps to I(z_y;Y|D)
                'class_capture': metrics.get('I(Z;Y|D)', 0),
                # Domain leakage: I(Z;D|Y) maps to I(z_y;D|Y)
                'domain_leakage': metrics.get('I(Z;D|Y)', 0),
                # Domain invariance score
                'domain_invariance': unified.get('domain_invariance_score', 0),
                # Total class info
                'total_class_info': metrics.get('I(Z;Y)', 0),
            }

    return baselines


def get_top_models_per_type(df: pd.DataFrame, n: int = 3) -> dict:
    """Get top N models by DQS for each model type.

    DQS (Disentanglement Quality Score) is the new metric that properly penalizes
    dimension collapse by requiring both z_y and z_d to capture their respective info.
    """
    top_models = {}
    for model_type in df['model_type'].unique():
        mt_df = df[df['model_type'] == model_type]
        top_models[model_type] = mt_df.nlargest(n, 'dqs')
    return top_models


def create_visualization(df: pd.DataFrame, top_models_per_type: dict, output_path: str,
                         baselines: dict = None):
    """Create multi-panel visualization showing top 3 models per type across all IT metrics.

    Args:
        baselines: Optional dict with DANN/IRM baseline metrics for reference lines
    """

    # Define IT metrics to plot with baseline mapping
    # Format: (csv_column, title, subtitle, baseline_key or None)
    metrics = [
        ('it_partition_quality', 'Partition Quality', 'overall IT quality', None),
        ('it_zy_specificity', 'z_y Specificity', 'I(z_y;Y|D) - I(z_y;D|Y)', None),
        ('it_zd_specificity', 'z_d Specificity', 'I(z_d;D|Y) - I(z_d;Y|D)', None),
        ('it_I_zy_Y_given_D', 'I(z_y; Y|D)', 'class info capture', 'class_capture'),
        ('it_I_zd_D_given_Y', 'I(z_d; D|Y)', 'domain info capture', None),
        ('it_I_zdy_Y_D', 'I(z_dy; Y;D)', 'interaction', None),
    ]

    # Model type colors (for box plots background)
    model_type_colors = {
        'nvae': '#9467BD',        # Purple
        'diva': '#FF7F0E',        # Orange
        'dann_augmented': '#17BECF'  # Cyan
    }

    # Colors for top 3 within each model type - distinct markers
    rank_colors = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green for #1, #2, #3
    rank_markers = ['o', 's', '^']  # Circle, Square, Triangle

    model_types = ['dann_augmented', 'diva', 'nvae']

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title, subtitle, baseline_key) in enumerate(metrics):
        ax = axes[idx]

        positions = []
        pos = 0
        x_ticks = []
        x_labels = []

        for mt in model_types:
            mt_data = df[df['model_type'] == mt][metric].dropna()
            if len(mt_data) == 0:
                continue

            # Calculate quartiles
            q1 = mt_data.quantile(0.25)
            q2 = mt_data.quantile(0.50)
            q3 = mt_data.quantile(0.75)
            min_val = mt_data.min()
            max_val = mt_data.max()

            # Draw box plot manually for better control
            box_width = 0.6
            whisker_width = 0.3

            # Box (Q1 to Q3)
            rect = plt.Rectangle((pos - box_width/2, q1), box_width, q3-q1,
                                 fill=True, facecolor=model_type_colors.get(mt, '#888888'),
                                 edgecolor='black', alpha=0.3, linewidth=1)
            ax.add_patch(rect)

            # Median line
            ax.hlines(q2, pos - box_width/2, pos + box_width/2, colors='black', linewidth=2)

            # Whiskers
            ax.vlines(pos, min_val, q1, colors='black', linewidth=1)
            ax.vlines(pos, q3, max_val, colors='black', linewidth=1)
            ax.hlines(min_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)
            ax.hlines(max_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)

            # Plot top 3 models FOR THIS MODEL TYPE
            if mt in top_models_per_type:
                top_mt = top_models_per_type[mt]
                for rank, (_, row) in enumerate(top_mt.iterrows()):
                    value = row[metric]
                    if pd.isna(value):
                        continue

                    # Add jitter to avoid overlap
                    jitter = (rank - 1) * 0.12

                    # Plot marker with rank-specific color and marker
                    ax.scatter(pos + jitter, value,
                              c=rank_colors[rank],
                              s=180,
                              marker=rank_markers[rank],
                              edgecolors='black',
                              linewidths=1.5,
                              zorder=10)

                    # Add rank label
                    ax.annotate(f"#{rank+1}", (pos + jitter, value),
                               textcoords="offset points", xytext=(0, 8),
                               ha='center', fontsize=8, fontweight='bold',
                               color=rank_colors[rank])

            x_ticks.append(pos)
            x_labels.append(mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG'))
            pos += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Set y limits with padding (include baselines in range calculation)
        all_vals = df[metric].dropna().tolist()

        # Add baseline values to range if applicable
        if baselines and baseline_key:
            for bm in baselines.values():
                if baseline_key in bm:
                    all_vals.append(bm[baseline_key])

        if len(all_vals) > 0:
            ymin, ymax = min(all_vals), max(all_vals)
            yrange = ymax - ymin if ymax != ymin else 1
            ax.set_ylim(ymin - 0.1*yrange, ymax + 0.2*yrange)

        # Draw baseline reference lines for applicable metrics
        if baselines and baseline_key:
            x_range = ax.get_xlim()
            for bm_name, bm_metrics in baselines.items():
                if baseline_key in bm_metrics:
                    value = bm_metrics[baseline_key]
                    ax.axhline(y=value, color=BASELINE_COLORS.get(bm_name, 'gray'),
                              linestyle='--', linewidth=2, alpha=0.8, zorder=5)
                    # Add label on the right side
                    ax.annotate(f'{BASELINE_LABELS.get(bm_name, bm_name)}: {value:.2f}',
                               xy=(x_range[1], value), xytext=(5, 0),
                               textcoords='offset points', fontsize=8, fontweight='bold',
                               color=BASELINE_COLORS.get(bm_name, 'gray'),
                               va='center', ha='left')

    # Create legend
    legend_elements = []

    # Rank legend
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 per Model Type:'))
    for rank in range(3):
        legend_elements.append(plt.scatter([], [], c=rank_colors[rank], s=100,
                                           marker=rank_markers[rank], edgecolors='black',
                                           label=f'#{rank+1} Best'))

    # Model type legend
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Model Types:'))
    for mt in model_types:
        legend_elements.append(mpatches.Patch(facecolor=model_type_colors[mt], edgecolor='black', alpha=0.3,
                                              label=mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG')))

    # Baseline legend (if baselines provided)
    if baselines:
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Baselines:'))
        for bm_name in baselines:
            legend_elements.append(Line2D([0], [0], color=BASELINE_COLORS.get(bm_name, 'gray'),
                                         linestyle='--', linewidth=2,
                                         label=BASELINE_LABELS.get(bm_name, bm_name)))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=9, frameon=True, fancybox=True)

    # Title
    baseline_note = " + DANN/IRM baselines" if baselines else ""
    fig.suptitle(f'Top 3 Models Per Type (by DQS) - Tracked Across All IT Metrics{baseline_note}\n'
                 'Box plots show distribution; colored markers indicate top 3 by DQS for each model type',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def print_top_models_summary(top_models_per_type: dict):
    """Print summary of top 3 models per type."""
    print("\n" + "="*80)
    print("TOP 3 MODELS BY DQS - PER MODEL TYPE")
    print("="*80)

    metrics = ['dqs', 'it_partition_quality', 'it_zy_specificity', 'it_zd_specificity',
               'it_I_zy_Y_given_D', 'it_I_zd_D_given_Y', 'it_I_zdy_Y_D']

    for model_type, top_df in top_models_per_type.items():
        print(f"\n{'='*40}")
        print(f"  {model_type.upper()}")
        print(f"{'='*40}")

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"\n  #{rank}: {row['name']}")
            print(f"      Presets: adv={row.get('preset_adversarial', 'N/A')}, "
                  f"cls={row.get('preset_classifier', 'N/A')}, "
                  f"sp={row.get('preset_sparsity', 'N/A')}, "
                  f"kl={row.get('preset_kl', 'N/A')}")
            print(f"      Test Acc: {row['test_accuracy']:.4f}, Gen Gap: {row['gen_gap']:.4f}")
            print(f"      DQS: {row['dqs']:.4f}")
            print(f"      IT Metrics:")
            for m in metrics[1:]:  # Skip DQS (already printed)
                if pd.notna(row.get(m)):
                    print(f"          {m}: {row[m]:.4f}")


def main():
    # Paths
    csv_path = '/workspace/NVAE/grid_search_results_separate_encoders/summary/all_results.csv'
    output_path = '/workspace/NVAE/grid_search_results_separate_encoders/summary/it_top3_models_comparison.png'
    baseline_dir = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'

    # Load data
    print("Loading grid search results...")
    df = load_results(csv_path)
    print(f"Loaded {len(df)} experiments")

    # Load baseline results (DANN/IRM)
    print("\nLoading baseline results (DANN/IRM)...")
    baselines = load_baseline_results(baseline_dir)
    if baselines:
        print(f"Loaded baselines: {list(baselines.keys())}")
        for name, metrics in baselines.items():
            print(f"  {name}: class_capture={metrics['class_capture']:.3f}, "
                  f"domain_leakage={metrics['domain_leakage']:.3f}")
    else:
        print("No baseline results found")

    # Get top 3 models per type
    top_models_per_type = get_top_models_per_type(df, n=3)
    print_top_models_summary(top_models_per_type)

    # Create visualization with baselines
    print(f"\nCreating visualization...")
    create_visualization(df, top_models_per_type, output_path, baselines=baselines)
    print(f"Saved to: {output_path}")

    return top_models_per_type


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create visualization comparing old Partition Quality vs new DQS metric.
Shows how DQS properly penalizes dimension collapse.
Note: DANN/IRM baselines not shown as DQS requires decomposed latent spaces.
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


# Baseline model colors - gray tones so they don't stand out too much
BASELINE_COLORS = {
    'dann': '#555555',  # Dark gray
    'irm': '#888888',   # Medium gray
}

BASELINE_LABELS = {
    'dann': 'DANN',
    'irm': 'IRM',
}


def load_baseline_results(baseline_dir: str = None) -> dict:
    """Load DANN/IRM baseline results for reference.

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

            metrics = data.get('metrics', {})
            unified = data.get('unified_metrics', {})

            baselines[model] = {
                'class_capture': metrics.get('I(Z;Y|D)', 0),
                'domain_leakage': metrics.get('I(Z;D|Y)', 0),
                'domain_invariance': unified.get('domain_invariance_score', 0),
                'total_class_info': metrics.get('I(Z;Y)', 0),
            }

    return baselines


def load_and_compute_dqs(csv_path: str) -> pd.DataFrame:
    """Load grid search results and compute DQS for each model."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['it_partition_quality'])

    # Compute DQS for each model
    dqs_scores = []
    f1_y_scores = []
    f1_d_scores = []
    capture_y_scores = []
    capture_d_scores = []

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
        f1_y_scores.append(result['f1_y'])
        f1_d_scores.append(result['f1_d'])
        capture_y_scores.append(result['capture_y'])
        capture_d_scores.append(result['capture_d'])

    df['dqs'] = dqs_scores
    df['f1_y'] = f1_y_scores
    df['f1_d'] = f1_d_scores
    df['capture_y'] = capture_y_scores
    df['capture_d'] = capture_d_scores

    return df


def get_top_models_by_dqs(df: pd.DataFrame, n: int = 3) -> dict:
    """Get top N models by DQS for each model type.

    These same models will be tracked across ALL IT metric plots for consistency.
    DQS (Disentanglement Quality Score) is the new metric that properly penalizes
    dimension collapse by requiring both z_y and z_d to capture their respective info.
    """
    top_models = {}
    for model_type in df['model_type'].unique():
        mt_df = df[df['model_type'] == model_type]
        if len(mt_df) > 0:
            top_models[model_type] = mt_df.nlargest(n, 'dqs')
    return top_models


def create_comparison_plot(df: pd.DataFrame, output_path: str, top_models_per_type: dict = None):
    """Create DQS metrics visualization with top 3 models tracked consistently.

    Args:
        top_models_per_type: Pre-selected top models by DQS.
                            If None, will be computed internally.
    """

    model_types = ['dann_augmented', 'diva', 'nvae']
    model_type_colors = {
        'nvae': '#9467BD',
        'diva': '#FF7F0E',
        'dann_augmented': '#17BECF'
    }
    rank_colors = ['#E41A1C', '#377EB8', '#4DAF4A']
    rank_markers = ['o', 's', '^']

    # Select top 3 by DQS ONCE - same models tracked across all plots
    if top_models_per_type is None:
        top_models_per_type = get_top_models_by_dqs(df, n=3)

    # Create figure with 2 rows x 3 cols for DQS metrics only
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Core DQS metrics
    row1_metrics = [
        ('dqs', 'DQS (Disentanglement Quality)', 'harmonic mean of F1_y and F1_d'),
        ('f1_y', 'F1_y (Class Latent Quality)', 'harmonic(capture_y, purity_y)'),
        ('f1_d', 'F1_d (Domain Latent Quality)', 'harmonic(capture_d, purity_d)'),
    ]

    # Row 2: Component metrics
    row2_metrics = [
        ('capture_y', 'Capture_y', 'I(z_y;Y|D) / H(Y)'),
        ('capture_d', 'Capture_d', 'I(z_d;D|Y) / H(D)'),
        ('it_I_zdy_Y_D', 'Synergy I(z_dy;Y;D)', 'interaction information'),
    ]

    for row_idx, metrics_list in enumerate([row1_metrics, row2_metrics]):
        for col_idx, (metric, title, subtitle) in enumerate(metrics_list):
            ax = axes[row_idx, col_idx]

            # Use pre-selected top 3 by partition quality (NOT by current metric)

            pos = 0
            x_ticks = []
            x_labels = []

            for mt in model_types:
                mt_data = df[df['model_type'] == mt][metric].dropna()
                if len(mt_data) == 0:
                    continue

                # Calculate quartiles
                q1, q2, q3 = mt_data.quantile([0.25, 0.50, 0.75])
                min_val, max_val = mt_data.min(), mt_data.max()

                # Draw box plot
                box_width = 0.6
                whisker_width = 0.3

                rect = plt.Rectangle((pos - box_width/2, q1), box_width, q3-q1,
                                     fill=True, facecolor=model_type_colors.get(mt, '#888888'),
                                     edgecolor='black', alpha=0.3, linewidth=1)
                ax.add_patch(rect)
                ax.hlines(q2, pos - box_width/2, pos + box_width/2, colors='black', linewidth=2)
                ax.vlines(pos, min_val, q1, colors='black', linewidth=1)
                ax.vlines(pos, q3, max_val, colors='black', linewidth=1)
                ax.hlines(min_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)
                ax.hlines(max_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)

                # Plot top 3 models
                if mt in top_models_per_type:
                    for rank, (_, row) in enumerate(top_models_per_type[mt].iterrows()):
                        value = row[metric]
                        if pd.isna(value):
                            continue
                        jitter = (rank - 1) * 0.12
                        ax.scatter(pos + jitter, value, c=rank_colors[rank], s=180,
                                  marker=rank_markers[rank], edgecolors='black', linewidths=1.5, zorder=10)
                        ax.annotate(f"#{rank+1}", (pos + jitter, value),
                                   textcoords="offset points", xytext=(0, 8),
                                   ha='center', fontsize=8, fontweight='bold', color=rank_colors[rank])

                x_ticks.append(pos)
                x_labels.append(mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG'))
                pos += 1

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
            ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            all_vals = df[metric].dropna()
            if len(all_vals) > 0:
                ymin, ymax = all_vals.min(), all_vals.max()
                yrange = ymax - ymin if ymax != ymin else 1
                ax.set_ylim(ymin - 0.1*yrange, ymax + 0.15*yrange)

    # Add row labels
    axes[0, 0].annotate('DQS SCORES', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                        color='#1f77b4')
    axes[1, 0].annotate('COMPONENTS', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=14, fontweight='bold', rotation=90, va='center', ha='center',
                        color='#1f77b4')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 by DQS:'),
    ]
    for rank in range(3):
        legend_elements.append(plt.scatter([], [], c=rank_colors[rank], s=100,
                                           marker=rank_markers[rank], edgecolors='black',
                                           label=f'#{rank+1} Best'))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    for mt in model_types:
        legend_elements.append(mpatches.Patch(facecolor=model_type_colors[mt], edgecolor='black', alpha=0.3,
                                              label=mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG')))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=9, frameon=True, fancybox=True)

    fig.suptitle('DQS Metrics - Grid Search Results\n'
                 'Top 3 models (by DQS) tracked consistently across all plots',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def print_ranking_comparison(df: pd.DataFrame, baselines: dict = None):
    """Print comparison of rankings by old PQ vs new DQS."""
    print("\n" + "="*90)
    print("RANKING COMPARISON: Old Partition Quality vs New DQS")
    print("="*90)

    for mt in ['dann_augmented', 'diva', 'nvae']:
        mt_df = df[df['model_type'] == mt].copy()
        if len(mt_df) == 0:
            continue

        print(f"\n{'='*40}")
        print(f"  {mt.upper()}")
        print(f"{'='*40}")

        # Top 3 by old PQ
        top_pq = mt_df.nlargest(3, 'it_partition_quality')
        print("\n  TOP 3 by OLD Partition Quality:")
        for rank, (_, row) in enumerate(top_pq.iterrows(), 1):
            print(f"    #{rank}: PQ={row['it_partition_quality']:.3f}, DQS={row['dqs']:.3f}, "
                  f"Cap_y={row['capture_y']:.2f}, Cap_d={row['capture_d']:.2f}")

        # Top 3 by new DQS
        top_dqs = mt_df.nlargest(3, 'dqs')
        print("\n  TOP 3 by NEW DQS:")
        for rank, (_, row) in enumerate(top_dqs.iterrows(), 1):
            print(f"    #{rank}: DQS={row['dqs']:.3f}, PQ={row['it_partition_quality']:.3f}, "
                  f"Cap_y={row['capture_y']:.2f}, Cap_d={row['capture_d']:.2f}")

    # Print baseline comparison (DANN/IRM don't have DQS)
    if baselines:
        print("\n" + "="*90)
        print("MONOLITHIC BASELINES (for reference - no DQS available)")
        print("="*90)
        print("\nNote: DQS requires decomposed latent spaces. DANN/IRM have monolithic Z.")
        print("Compare using domain invariance score instead:")
        for bm_name, bm_metrics in baselines.items():
            print(f"\n  {BASELINE_LABELS.get(bm_name, bm_name)}:")
            print(f"    Domain Invariance: {bm_metrics['domain_invariance']:.3f}")
            print(f"    Class Capture I(Z;Y|D): {bm_metrics['class_capture']:.3f}")
            print(f"    Domain Leakage I(Z;D|Y): {bm_metrics['domain_leakage']:.3f}")


def main():
    csv_path = '/workspace/NVAE/grid_search_results_separate_encoders/summary/all_results.csv'
    output_path = '/workspace/NVAE/grid_search_results_separate_encoders/summary/dqs_vs_pq_comparison.png'
    baseline_dir = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'

    print("Loading grid search results and computing DQS...")
    df = load_and_compute_dqs(csv_path)
    print(f"Processed {len(df)} experiments")

    # Select top 3 by DQS ONCE - these same models tracked across all plots
    print("\nSelecting top 3 models by DQS (tracked across all plots)...")
    top_models_per_type = get_top_models_by_dqs(df, n=3)
    for mt, top_df in top_models_per_type.items():
        print(f"  {mt.upper()}: {list(top_df['name'].values)[:3]}")

    # Load baseline results for reference
    print("\nLoading baseline results (DANN/IRM)...")
    baselines = load_baseline_results(baseline_dir)
    if baselines:
        print(f"Loaded baselines: {list(baselines.keys())}")
    else:
        print("No baseline results found")

    print_ranking_comparison(df, baselines)

    print(f"\nCreating visualization...")
    create_comparison_plot(df, output_path, top_models_per_type)
    print(f"Saved to: {output_path}")
    print("Note: Same top 3 models (by DQS) shown in all plots for consistency")

    # Save updated CSV with DQS scores
    output_csv = '/workspace/NVAE/grid_search_results_separate_encoders/summary/all_results_with_dqs.csv'
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV to: {output_csv}")


if __name__ == '__main__':
    main()

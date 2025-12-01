#!/usr/bin/env python3
"""
Minimal Partition Adherence Score (MPAS) Visualization

Creates a visualization that judges models by adherence to the Minimally Partitioned
Representation definition:

Definition: A representation Z = (Z_Y, Z_D, Z_{YD}, Z_X) is minimally partitioned if:
1. Z_Y captures class-specific information: I(Z_Y;D|Y) = 0
2. Z_D captures domain-specific information: I(Z_D;Y|D) = 0
3. Z_{YD} captures shared information: predictive of both Y and D
4. Z_X captures residual variation: I(Z_X;Y,D) = 0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Constants - entropy in bits (npeet returns MI in bits, i.e., log base 2)
H_Y = 3.322  # log2(10) for 10 classes (bits)
H_D = 2.585  # log2(6) for 6 domains (bits)

# Baseline domain leakage values (from 20-epoch balanced experiments)
# These are I(Z;D|Y) for monolithic models without decomposed spaces
DANN_BASELINE_DOMAIN_LEAKAGE = 0.9578  # I(Z;D|Y) for standard DANN
IRM_BASELINE_DOMAIN_LEAKAGE = 0.9889   # I(Z;D|Y) for IRM

# Paths
CSV_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/all_results.csv'
OUTPUT_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/minimal_partition_adherence.png'


def compute_mpas(row):
    """
    Compute Minimal Partition Adherence Score (MPAS).

    Measures adherence to the Minimally Partitioned Representation definition:
    1. Z_Y captures class-specific information: I(Z_Y;D|Y) = 0
    2. Z_D captures domain-specific information: I(Z_D;Y|D) = 0
    3. Z_YD captures shared information: predictive of both Y and D
    4. Z_X captures residual variation: I(Z_X;Y,D) = 0

    For DIVA (which lacks Z_YD), only criteria 1 and 2 are used.
    For MP-DANN and MP-VAE, all 4 criteria are used.

    Score in [0, 1] where 1 = perfect adherence.
    """
    model_type = row['model_type']

    # Criterion 1: I(Z_Y;D|Y) -> 0 (domain leakage in class space)
    zy_d_given_y = row['it_I_zy_D_given_Y'] or 0
    zy_adherence = 1.0 / (1.0 + zy_d_given_y / H_D)

    # Criterion 2: I(Z_D;Y|D) -> 0 (class leakage in domain space)
    zd_y_given_d = row['it_I_zd_Y_given_D'] or 0
    zd_adherence = 1.0 / (1.0 + zd_y_given_d / H_Y)

    # Criterion 3: Z_YD captures shared info: I(Z_YD;Y,D) should be HIGH
    # Only applicable for models with Z_YD (not DIVA)
    zdy_joint = row['it_I_zdy_joint'] or 0
    max_joint = np.sqrt(H_Y * H_D)
    zdy_score = min(1.0, zdy_joint / max_joint)
    zdy_score = max(0.01, zdy_score)  # Avoid zero in geometric mean

    # Criterion 4: I(Z_X;Y,D) -> 0 (residual independence)
    # Z_X not measured in current experiments, assume perfect adherence
    zx_adherence = 1.0

    # Compute MPAS based on model type
    if model_type == 'diva':
        # DIVA lacks Z_YD space, so only use criteria 1 and 2
        # Geometric mean of 2 applicable criteria
        mpas = (zy_adherence * zd_adherence) ** 0.5
    else:
        # MP-DANN and MP-VAE have Z_YD, use all 4 criteria
        # Geometric mean of all 4 criteria
        mpas = (zy_adherence * zd_adherence * zdy_score * zx_adherence) ** 0.25

    return {
        'mpas': mpas,
        'zy_adherence': zy_adherence,
        'zd_adherence': zd_adherence,
        'zdy_score': zdy_score,
        'zx_adherence': zx_adherence
    }


def load_and_process_data(csv_path):
    """Load data and compute MPAS for all experiments."""
    df = pd.read_csv(csv_path)

    # Filter to decomposed models only
    df = df[df['model_type'].isin(['diva', 'nvae', 'dann_augmented'])]
    df = df.dropna(subset=['it_partition_quality', 'it_I_zy_D_given_Y', 'it_I_zd_Y_given_D',
                           'it_I_zdy_joint', 'it_I_zy_Y_given_D', 'it_I_zd_D_given_Y'])

    # COLLAPSE FILTER: Exclude models with collapsed representation spaces
    # A space is collapsed if it has BOTH low leakage AND low intended MI
    # (Low leakage alone is GOOD if the space still captures intended info)

    LEAKAGE_THRESH = 0.05   # Threshold for "low" leakage
    INTENDED_THRESH = 0.5   # Threshold for "low" intended MI

    # Original collapse criteria (OOD or Z_YD collapse)
    collapsed_ood_zyd = (df['model_type'] == 'dann_augmented') & (
        (df['ood_accuracy'] < 0.50) |  # Very low OOD performance
        (df['it_I_zdy_joint'] < 0.1)   # Completely collapsed Z_YD space
    )
    n_ood_zyd = collapsed_ood_zyd.sum()

    # Collapsed Z_Y space: low I(Z_Y;D|Y) AND low I(Z_Y;Y|D)
    collapsed_zy = (df['model_type'] == 'dann_augmented') & (
        (df['it_I_zy_D_given_Y'] < LEAKAGE_THRESH) &
        (df['it_I_zy_Y_given_D'] < INTENDED_THRESH)
    )
    n_zy = collapsed_zy.sum()

    # Collapsed Z_D space: low I(Z_D;Y|D) AND low I(Z_D;D|Y)
    collapsed_zd = (df['model_type'] == 'dann_augmented') & (
        (df['it_I_zd_Y_given_D'] < LEAKAGE_THRESH) &
        (df['it_I_zd_D_given_Y'] < INTENDED_THRESH)
    )
    n_zd = collapsed_zd.sum()

    # Combined filter
    collapsed = collapsed_ood_zyd | collapsed_zy | collapsed_zd
    n_excluded = collapsed.sum()
    df = df[~collapsed]

    print(f"Collapse filter results:")
    print(f"  - OOD < 50% or I(Z_YD) < 0.1: {n_ood_zyd} models")
    print(f"  - Collapsed Z_Y (low leakage + low intended): {n_zy} models")
    print(f"  - Collapsed Z_D (low leakage + low intended): {n_zd} models")
    print(f"  - Total excluded (unique): {n_excluded} models")

    # Compute MPAS for each row
    mpas_results = df.apply(compute_mpas, axis=1)
    df['mpas'] = [r['mpas'] for r in mpas_results]
    df['zy_adherence'] = [r['zy_adherence'] for r in mpas_results]
    df['zd_adherence'] = [r['zd_adherence'] for r in mpas_results]
    df['zdy_score'] = [r['zdy_score'] for r in mpas_results]

    return df


def create_visualization(df, output_path):
    """Create 2x2 visualization with top 3 per model type tracked across all panels."""

    # Model types and their display names
    model_types = ['dann_augmented', 'nvae', 'diva']
    model_display_names = {
        'dann_augmented': 'MP-DANN',
        'nvae': 'MP-VAE',
        'diva': 'DIVA'
    }

    # Colors for each model type
    model_type_colors = {
        'dann_augmented': '#17BECF',  # Cyan
        'nvae': '#9467BD',            # Purple
        'diva': '#FF7F0E',            # Orange
    }

    # Markers for rank within model type
    rank_markers = ['*', 's', '^']  # Star, Square, Triangle for #1, #2, #3
    rank_sizes = [350, 200, 180]    # Sizes for each rank

    # Get top 3 per model type by MPAS
    top3_per_type = {}
    print("\n" + "="*80)
    print("TOP 3 PER MODEL TYPE BY MPAS")
    print("="*80)

    for mt in model_types:
        mt_data = df[df['model_type'] == mt]
        top3 = mt_data.nlargest(3, 'mpas')
        top3_per_type[mt] = top3.index.tolist()

        print(f"\n{model_display_names[mt]}:")
        for rank, (idx, row) in enumerate(top3.iterrows(), 1):
            print(f"  #{rank}: {row['name']}")
            print(f"      MPAS: {row['mpas']:.4f}, PQ: {row['it_partition_quality']:.4f}, OOD: {row['ood_accuracy']:.2%}")

    # Panel definitions: (metric_col, title, subtitle, lower_is_better)
    panels = [
        ('it_I_zy_D_given_Y', r'Criterion 1: $I(Z_Y;D|Y) \rightarrow 0$', 'Domain leakage in class space (lower = better)', True),
        ('it_I_zd_Y_given_D', r'Criterion 2: $I(Z_D;Y|D) \rightarrow 0$', 'Class leakage in domain space (lower = better)', True),
        ('it_I_zdy_joint', r'Criterion 3: $Z_{YD}$ captures shared info', 'Joint information I(Z_YD;Y,D) (higher = better)', False),
        ('mpas', 'Minimal Partition Adherence Score', 'Geometric mean of applicable criteria (higher = better)', False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for panel_idx, (metric, title, subtitle, lower_is_better) in enumerate(panels):
        ax = axes[panel_idx]
        pos = 0
        x_ticks = []
        x_labels = []

        # For Criterion 3 (Z_YD), skip DIVA since it has no Z_YD space
        panel_model_types = model_types
        if panel_idx == 2:  # Criterion 3: I(Z_YD;Y,D)
            panel_model_types = [mt for mt in model_types if mt != 'diva']

        for mt in panel_model_types:
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
            rect = plt.Rectangle((pos - box_width/2, q1), box_width, q3-q1,
                                 fill=True, facecolor=model_type_colors[mt],
                                 edgecolor='black', alpha=0.4, linewidth=1)
            ax.add_patch(rect)
            ax.hlines(q2, pos - box_width/2, pos + box_width/2, colors='black', linewidth=2)
            ax.vlines(pos, min_val, q1, colors='black', linewidth=1)
            ax.vlines(pos, q3, max_val, colors='black', linewidth=1)
            ax.hlines(min_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)
            ax.hlines(max_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)

            # Plot top 3 for THIS model type
            for rank, top_idx in enumerate(top3_per_type[mt]):
                top_row = df.loc[top_idx]
                value = top_row[metric]
                if pd.isna(value):
                    continue
                # Jitter horizontally to avoid overlap
                jitter = (rank - 1) * 0.12
                ax.scatter(pos + jitter, value,
                          c=model_type_colors[mt],
                          s=rank_sizes[rank],
                          marker=rank_markers[rank],
                          edgecolors='black',
                          linewidths=1.5,
                          zorder=10)
                # Add rank label
                ax.annotate(f'#{rank+1}', (pos + jitter, value),
                           textcoords='offset points', xytext=(0, 12),
                           ha='center', fontsize=9, fontweight='bold',
                           color='black')

            x_ticks.append(pos)
            x_labels.append(model_display_names[mt])
            pos += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add ideal target line for leakage metrics
        if lower_is_better:
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal: 0')

        # Add DANN/IRM baseline lines for Criterion 1 (domain leakage)
        if panel_idx == 0:  # Criterion 1: I(Z_Y;D|Y)
            ax.axhline(y=DANN_BASELINE_DOMAIN_LEAKAGE, color='gray', linestyle=':',
                      linewidth=2.5, alpha=0.8)
            ax.axhline(y=IRM_BASELINE_DOMAIN_LEAKAGE, color='gray', linestyle=':',
                      linewidth=2.5, alpha=0.8)
            # Add labels on the right side
            ax.text(len(model_types) - 0.3, DANN_BASELINE_DOMAIN_LEAKAGE, 'DANN',
                   fontsize=10, fontweight='bold', color='gray', va='center', ha='left')
            ax.text(len(model_types) - 0.3, IRM_BASELINE_DOMAIN_LEAKAGE, 'IRM',
                   fontsize=10, fontweight='bold', color='gray', va='center', ha='left')

        # Auto-scale y-axis
        all_vals = df[metric].dropna()
        if len(all_vals) > 0:
            ymin, ymax = all_vals.min(), all_vals.max()
            # For Panel 1, include DANN/IRM baseline values in scaling
            if panel_idx == 0:
                ymax = max(ymax, IRM_BASELINE_DOMAIN_LEAKAGE, DANN_BASELINE_DOMAIN_LEAKAGE)
            yrange = ymax - ymin if ymax != ymin else 1
            if lower_is_better:
                ax.set_ylim(-0.05 * yrange, ymax + 0.15 * yrange)
            else:
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.15 * yrange)

    # Create legend with 3 columns: Model Types | Top 3 Markers | Baselines
    # Build each column separately, then interleave for proper alignment

    # Column 1: Model Types (3 items)
    col1 = []
    for mt in model_types:
        col1.append(mpatches.Patch(facecolor=model_type_colors[mt], edgecolor='black', alpha=0.4,
                                   label=model_display_names[mt]))

    # Column 2: Top 3 Markers (3 items)
    col2 = []
    for rank in range(3):
        col2.append(plt.scatter([], [], c='gray', s=rank_sizes[rank],
                               marker=rank_markers[rank], edgecolors='black',
                               label=f'#{rank+1} Best'))

    # Column 3: Baselines (3 items - pad with blanks)
    col3 = [
        plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2.5, label='DANN baseline'),
        plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2.5, label='IRM baseline'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal: 0'),
    ]

    # Interleave columns for row-by-row layout with ncol=3
    legend_elements = []
    for i in range(3):
        legend_elements.append(col1[i])
        legend_elements.append(col2[i])
        legend_elements.append(col3[i])

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, fontsize=9, frameon=True, fancybox=True,
               columnspacing=2.0, handletextpad=0.5)

    fig.suptitle('Minimal Partition Adherence: Evaluation by Definition Criteria\n'
                 'Box plots show distribution; markers show top 3 per model type by MPAS\n'
                 '(10 Epochs, Separate Encoders, Excluding Collapsed Models)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    return top3_per_type


def main():
    print("Loading and processing data...")
    df = load_and_process_data(CSV_PATH)
    print(f"Loaded {len(df)} experiments")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")

    print("\nCreating visualization...")
    top3_per_type = create_visualization(df, OUTPUT_PATH)

    return df, top3_per_type


if __name__ == '__main__':
    main()

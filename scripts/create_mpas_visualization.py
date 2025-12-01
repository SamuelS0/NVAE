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

# Constants
H_Y = 2.303  # log(10) for 10 classes
H_D = 1.792  # log(6) for 6 domains

# Paths
CSV_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/all_results.csv'
OUTPUT_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/minimal_partition_adherence.png'


def compute_mpas(row):
    """
    Compute Minimal Partition Adherence Score (MPAS).

    Measures adherence to all 4 criteria of the minimally partitioned representation.
    Score in [0, 1] where 1 = perfect adherence.
    """
    # Criterion 1: I(Z_Y;D|Y) -> 0 (domain leakage in class space)
    zy_leakage = row['it_I_zy_D_given_Y'] / H_D
    zy_adherence = 1 / (1 + zy_leakage)

    # Criterion 2: I(Z_D;Y|D) -> 0 (class leakage in domain space)
    zd_leakage = row['it_I_zd_Y_given_D'] / H_Y
    zd_adherence = 1 / (1 + zd_leakage)

    # Criterion 3: Z_YD captures shared info (normalize by geometric mean)
    zdy_joint = row['it_I_zdy_joint'] / np.sqrt(H_Y * H_D)
    zdy_score = min(max(zdy_joint, 0), 1.0)  # Clamp to [0, 1]

    # Criterion 4: Use partition_quality as proxy for Z_X adherence
    partition_quality = row['it_partition_quality']

    # Final MPAS: Geometric mean of adherence scores
    mpas = (zy_adherence * zd_adherence * max(0.01, zdy_score) * partition_quality) ** 0.25

    return {
        'mpas': mpas,
        'zy_adherence': zy_adherence,
        'zd_adherence': zd_adherence,
        'zdy_score': zdy_score,
        'partition_quality': partition_quality
    }


def load_and_process_data(csv_path):
    """Load data and compute MPAS for all experiments."""
    df = pd.read_csv(csv_path)

    # Filter to decomposed models only
    df = df[df['model_type'].isin(['diva', 'nvae', 'dann_augmented'])]
    df = df.dropna(subset=['it_partition_quality', 'it_I_zy_D_given_Y', 'it_I_zd_Y_given_D', 'it_I_zdy_joint'])

    # Compute MPAS for each row
    mpas_results = df.apply(compute_mpas, axis=1)
    df['mpas'] = [r['mpas'] for r in mpas_results]
    df['zy_adherence'] = [r['zy_adherence'] for r in mpas_results]
    df['zd_adherence'] = [r['zd_adherence'] for r in mpas_results]
    df['zdy_score'] = [r['zdy_score'] for r in mpas_results]

    return df


def create_visualization(df, output_path):
    """Create 2x2 visualization with definition-aligned panels."""

    # Get global top 3 by MPAS
    top3 = df.nlargest(3, 'mpas')
    top3_indices = top3.index.tolist()

    print("\n" + "="*80)
    print("GLOBAL TOP 3 BY MINIMAL PARTITION ADHERENCE SCORE (MPAS)")
    print("="*80)
    for rank, (idx, row) in enumerate(top3.iterrows(), 1):
        print(f"\n#{rank}: {row['name']}")
        print(f"    Model Type: {row['model_type']}")
        print(f"    MPAS: {row['mpas']:.4f}")
        print(f"    Z_Y Adherence (I(Z_Y;D|Y)->0): {row['zy_adherence']:.4f} (leakage={row['it_I_zy_D_given_Y']:.4f})")
        print(f"    Z_D Adherence (I(Z_D;Y|D)->0): {row['zd_adherence']:.4f} (leakage={row['it_I_zd_Y_given_D']:.4f})")
        print(f"    Z_YD Score: {row['zdy_score']:.4f} (joint={row['it_I_zdy_joint']:.4f})")
        print(f"    Partition Quality: {row['it_partition_quality']:.4f}")
        print(f"    Test Accuracy: {row['test_accuracy']:.4f}")

    # Colors and markers
    model_type_colors = {
        'nvae': '#9467BD',        # Purple
        'diva': '#FF7F0E',        # Orange
        'dann_augmented': '#17BECF'  # Cyan
    }
    rank_colors = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
    rank_markers = ['o', 's', '^']  # Circle, Square, Triangle
    model_types = ['dann_augmented', 'diva', 'nvae']

    # Panel definitions: (metric_col, title, subtitle, lower_is_better)
    panels = [
        ('it_I_zy_D_given_Y', r'Criterion 1: $I(Z_Y;D|Y) \rightarrow 0$', 'Domain leakage in class space (lower = better)', True),
        ('it_I_zd_Y_given_D', r'Criterion 2: $I(Z_D;Y|D) \rightarrow 0$', 'Class leakage in domain space (lower = better)', True),
        ('it_I_zdy_joint', r'Criterion 3: $Z_{YD}$ captures shared info', 'Joint information I(Z_YD;Y,D) (higher = better)', False),
        ('mpas', 'Minimal Partition Adherence Score (MPAS)', 'Geometric mean of all criteria adherence (higher = better)', False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (metric, title, subtitle, lower_is_better) in enumerate(panels):
        ax = axes[idx]
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

            box_width = 0.6
            whisker_width = 0.3

            # Draw box plot
            rect = plt.Rectangle((pos - box_width/2, q1), box_width, q3-q1,
                                 fill=True, facecolor=model_type_colors.get(mt, '#888888'),
                                 edgecolor='black', alpha=0.4, linewidth=1)
            ax.add_patch(rect)
            ax.hlines(q2, pos - box_width/2, pos + box_width/2, colors='black', linewidth=2)
            ax.vlines(pos, min_val, q1, colors='black', linewidth=1)
            ax.vlines(pos, q3, max_val, colors='black', linewidth=1)
            ax.hlines(min_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)
            ax.hlines(max_val, pos - whisker_width/2, pos + whisker_width/2, colors='black', linewidth=1)

            # Plot top 3 global models on this model type's box
            for rank, top_idx in enumerate(top3_indices):
                top_row = df.loc[top_idx]
                if top_row['model_type'] == mt:
                    value = top_row[metric]
                    if pd.isna(value):
                        continue
                    jitter = (rank - 1) * 0.15
                    ax.scatter(pos + jitter, value, c=rank_colors[rank], s=200,
                              marker=rank_markers[rank], edgecolors='black', linewidths=2, zorder=10)
                    ax.annotate(f'#{rank+1}', (pos + jitter, value),
                               textcoords='offset points', xytext=(0, 10),
                               ha='center', fontsize=10, fontweight='bold', color=rank_colors[rank])

            x_ticks.append(pos)
            x_labels.append(mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG'))
            pos += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add ideal target line for leakage metrics
        if lower_is_better:
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal: 0')

        # Auto-scale y-axis
        all_vals = df[metric].dropna()
        if len(all_vals) > 0:
            ymin, ymax = all_vals.min(), all_vals.max()
            yrange = ymax - ymin if ymax != ymin else 1
            if lower_is_better:
                ax.set_ylim(-0.05 * yrange, ymax + 0.15 * yrange)
            else:
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.15 * yrange)

    # Legend
    legend_elements = []
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Global Top 3 by MPAS:'))
    for rank in range(3):
        legend_elements.append(plt.scatter([], [], c=rank_colors[rank], s=150,
                                           marker=rank_markers[rank], edgecolors='black',
                                           label=f'#{rank+1} Best'))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Model Types:'))
    for mt in model_types:
        legend_elements.append(mpatches.Patch(facecolor=model_type_colors[mt], edgecolor='black', alpha=0.4,
                                              label=mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG')))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=10, frameon=True, fancybox=True)

    fig.suptitle('Minimal Partition Adherence: Evaluation by Definition Criteria\n'
                 'Box plots show quartile distribution; markers show global top 3 models by MPAS\n'
                 '(10 Epochs, Separate Encoders)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    return top3


def main():
    print("Loading and processing data...")
    df = load_and_process_data(CSV_PATH)
    print(f"Loaded {len(df)} experiments")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")

    print("\nCreating visualization...")
    top3 = create_visualization(df, OUTPUT_PATH)

    return df, top3


if __name__ == '__main__':
    main()

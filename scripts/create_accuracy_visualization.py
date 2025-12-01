#!/usr/bin/env python3
"""
Simple visualization showing test and OOD test accuracy for all models.
Uses grid search 10 epoch separate encoder results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
CSV_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/all_results.csv'
OUTPUT_PATH = '/workspace/NVAE/results/grid_search_10epochs_separate/summary/accuracy_comparison.png'

# Model display names and colors
MODEL_COLORS = {
    'dann_augmented': '#17BECF',  # Cyan
    'nvae': '#9467BD',            # Purple
    'diva': '#FF7F0E',            # Orange
    'dann': '#2CA02C',            # Green
    'irm': '#D62728'              # Red
}

MODEL_DISPLAY_NAMES = {
    'dann_augmented': 'MP-DANN',
    'nvae': 'MP-VAE',
    'diva': 'DIVA',
    'dann': 'DANN',
    'irm': 'IRM'
}

# Rank markers
RANK_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
RANK_MARKERS = ['o', 's', '^']  # Circle, Square, Triangle


def get_top3_per_type(df, metric='test_accuracy'):
    """Get top 3 models per model type by specified metric."""
    top_models = {}
    for model_type in df['model_type'].unique():
        mt_df = df[df['model_type'] == model_type].copy()
        mt_df = mt_df.dropna(subset=[metric])
        top_models[model_type] = mt_df.nlargest(3, metric)
    return top_models


def main():
    # Load data
    print(f"Loading results from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} experiments")
    print(f"Model counts: {df['model_type'].value_counts().to_dict()}")

    # Get top 3 per model type by test accuracy
    top3_per_type = get_top3_per_type(df, 'test_accuracy')

    # Collect top 3 indices
    top3_indices = {}
    for mt, top_df in top3_per_type.items():
        top3_indices[mt] = top_df.index.tolist()

    # Print summary
    print("\n" + "="*80)
    print("TOP 3 PER MODEL TYPE (by Test Accuracy)")
    print("="*80)
    for mt in MODEL_DISPLAY_NAMES.keys():
        if mt in top3_per_type:
            print(f"\n{MODEL_DISPLAY_NAMES[mt]}:")
            for rank, (idx, row) in enumerate(top3_per_type[mt].iterrows(), 1):
                print(f"  #{rank}: Test={row['test_accuracy']:.4f}, OOD={row['ood_accuracy']:.4f}")

    # Panel definitions
    panels = [
        ('test_accuracy', 'Test Accuracy (ID)', 'In-distribution test performance'),
        ('ood_accuracy', 'OOD Accuracy', 'Out-of-distribution test performance'),
    ]

    # Model type order
    model_types = ['dann_augmented', 'nvae', 'diva', 'dann', 'irm']
    model_types = [mt for mt in model_types if mt in df['model_type'].unique()]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, (metric, title, subtitle) in enumerate(panels):
        ax = axes[panel_idx]
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
            x_labels.append(MODEL_DISPLAY_NAMES.get(mt, mt.upper()))
            pos += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Set y-axis limits
        all_vals = df[metric].dropna()
        if len(all_vals) > 0:
            ymin, ymax = all_vals.min(), all_vals.max()
            yrange = ymax - ymin if ymax != ymin else 0.1
            ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.15 * yrange)

    # Create legend
    legend_elements = []

    # Top 3 markers
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 per Type:'))
    for rank in range(3):
        legend_elements.append(plt.scatter(
            [], [], c=RANK_COLORS[rank], s=120,
            marker=RANK_MARKERS[rank], edgecolors='black',
            label=f'#{rank+1} Best'
        ))

    # Model types
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Model Types:'))
    for mt in model_types:
        legend_elements.append(mpatches.Patch(
            facecolor=MODEL_COLORS[mt], edgecolor='black', alpha=0.4,
            label=MODEL_DISPLAY_NAMES.get(mt, mt.upper())
        ))

    fig.legend(
        handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
        ncol=5, fontsize=9, frameon=True, fancybox=True, labelspacing=1.0
    )

    fig.suptitle(
        'Accuracy Comparison: Test (ID) vs OOD Performance\n'
        'Box plots show distribution; markers show top 3 per model type by test accuracy\n'
        '(10 Epochs, Separate Encoders)',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved visualization to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

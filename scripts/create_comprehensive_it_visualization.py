#!/usr/bin/env python3
"""
Comprehensive IT metrics visualization for grid search results.

Creates multi-page visualization covering ALL important IT metrics:
- Page 1: Core Quality Metrics (DQS, PQ, F1_y, F1_d)
- Page 2: Class Information (z_y metrics) - includes DANN/IRM baselines
- Page 3: Domain Information (z_d metrics) - decomposed models only
- Page 4: Interaction & Capture metrics

Same top 3 models (by DQS) tracked consistently across ALL plots.
DANN/IRM baselines shown where metrics are comparable to monolithic Z.
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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Baseline colors and labels - gray tones so they don't stand out too much
BASELINE_COLORS = {
    'dann': '#555555',  # Dark gray
    'irm': '#888888',   # Medium gray
}

BASELINE_LABELS = {
    'dann': 'DANN',
    'irm': 'IRM',
}

# Model type colors
MODEL_TYPE_COLORS = {
    'nvae': '#9467BD',        # Purple
    'diva': '#FF7F0E',        # Orange
    'dann_augmented': '#17BECF'  # Cyan
}

# Rank colors and markers for top 3
RANK_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A']  # Red, Blue, Green
RANK_MARKERS = ['o', 's', '^']  # Circle, Square, Triangle

MODEL_TYPES = ['dann_augmented', 'diva', 'nvae']


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Metrics grouped by page
# Format: (csv_column, title, subtitle, baseline_key or None)
# baseline_key maps to DANN/IRM metrics if comparable

PAGE1_METRICS = [
    ('dqs', 'DQS', 'Disentanglement Quality Score', None),
    ('it_partition_quality', 'Partition Quality', 'Overall IT quality metric', None),
    ('f1_y', 'F1_y', 'Class latent quality (harmonic mean)', None),
    ('f1_d', 'F1_d', 'Domain latent quality (harmonic mean)', None),
]

PAGE2_METRICS = [
    ('it_I_zy_Y_given_D', 'I(z_y; Y|D)', 'Class info capture in z_y', 'I(Z;Y|D)'),
    ('it_I_zy_D_given_Y', 'I(z_y; D|Y)', 'Domain leakage in z_y', 'I(Z;D|Y)'),
    ('capture_y', 'Capture_y', 'Normalized: I(z_y;Y|D) / H(Y)', None),
    ('it_zy_specificity', 'z_y Specificity', 'I(z_y;Y|D) - I(z_y;D|Y)', None),
]

PAGE3_METRICS = [
    ('it_I_zd_D_given_Y', 'I(z_d; D|Y)', 'Domain info capture in z_d', None),
    ('it_I_zd_Y_given_D', 'I(z_d; Y|D)', 'Class leakage in z_d', None),
    ('capture_d', 'Capture_d', 'Normalized: I(z_d;D|Y) / H(D)', None),
    ('it_zd_specificity', 'z_d Specificity', 'I(z_d;D|Y) - I(z_d;Y|D)', None),
]

PAGE4_METRICS = [
    ('it_I_zdy_Y_D', 'I(z_dy; Y; D)', 'Interaction information', None),
    ('it_I_zdy_joint', 'I(z_dy; Y,D)', 'Joint information in z_dy', None),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(csv_path: str) -> pd.DataFrame:
    """Load grid search results and compute DQS if not present."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['it_partition_quality'])

    # Compute DQS if not present
    if 'dqs' not in df.columns:
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


def load_baseline_results(baseline_dir: str = None) -> dict:
    """Load DANN/IRM baseline results for reference lines."""
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
                # Direct metrics from monolithic Z
                'I(Z;Y|D)': metrics.get('I(Z;Y|D)', 0),
                'I(Z;D|Y)': metrics.get('I(Z;D|Y)', 0),
                'I(Z;Y)': metrics.get('I(Z;Y)', 0),
                'I(Z;D)': metrics.get('I(Z;D)', 0),
                'I(Z;Y;D)': metrics.get('I(Z;Y;D)', 0),
                'domain_invariance': unified.get('domain_invariance_score', 0),
            }

    return baselines


def get_top_models_by_dqs(df: pd.DataFrame, n: int = 3) -> dict:
    """Get top N models by DQS for each model type."""
    top_models = {}
    for model_type in df['model_type'].unique():
        mt_df = df[df['model_type'] == model_type]
        if len(mt_df) > 0:
            top_models[model_type] = mt_df.nlargest(n, 'dqs')
    return top_models


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_metric_subplot(ax, df: pd.DataFrame, metric: str, title: str, subtitle: str,
                          top_models_per_type: dict, baselines: dict = None,
                          baseline_key: str = None):
    """Create a single metric subplot with box plots, top 3 markers, and optional baselines."""

    pos = 0
    x_ticks = []
    x_labels = []

    for mt in MODEL_TYPES:
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
                             fill=True, facecolor=MODEL_TYPE_COLORS.get(mt, '#888888'),
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
                value = row.get(metric)
                if pd.isna(value):
                    continue
                jitter = (rank - 1) * 0.12
                ax.scatter(pos + jitter, value, c=RANK_COLORS[rank], s=180,
                          marker=RANK_MARKERS[rank], edgecolors='black', linewidths=1.5, zorder=10)
                ax.annotate(f"#{rank+1}", (pos + jitter, value),
                           textcoords="offset points", xytext=(0, 8),
                           ha='center', fontsize=8, fontweight='bold', color=RANK_COLORS[rank])

        x_ticks.append(pos)
        x_labels.append(mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG'))
        pos += 1

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
    ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Calculate y-limits including baselines
    all_vals = df[metric].dropna().tolist()
    if baselines and baseline_key:
        for bm in baselines.values():
            if baseline_key in bm:
                all_vals.append(bm[baseline_key])

    if len(all_vals) > 0:
        ymin, ymax = min(all_vals), max(all_vals)
        yrange = ymax - ymin if ymax != ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.2*yrange)

    # Draw baseline reference lines
    if baselines and baseline_key:
        x_range = ax.get_xlim()
        for bm_name, bm_metrics in baselines.items():
            if baseline_key in bm_metrics:
                value = bm_metrics[baseline_key]
                ax.axhline(y=value, color=BASELINE_COLORS.get(bm_name, 'gray'),
                          linestyle='--', linewidth=2, alpha=0.8, zorder=5)
                ax.annotate(f'{BASELINE_LABELS.get(bm_name, bm_name)}: {value:.2f}',
                           xy=(x_range[1], value), xytext=(5, 0),
                           textcoords='offset points', fontsize=8, fontweight='bold',
                           color=BASELINE_COLORS.get(bm_name, 'gray'),
                           va='center', ha='left')


def create_page(df: pd.DataFrame, metrics_list: list, page_title: str, output_path: str,
                top_models_per_type: dict, baselines: dict = None):
    """Create a single page of the visualization."""

    n_metrics = len(metrics_list)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (metric, title, subtitle, baseline_key) in enumerate(metrics_list):
        create_metric_subplot(axes[idx], df, metric, title, subtitle,
                             top_models_per_type, baselines, baseline_key)

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 by DQS:'),
    ]
    for rank in range(3):
        legend_elements.append(plt.scatter([], [], c=RANK_COLORS[rank], s=100,
                                           marker=RANK_MARKERS[rank], edgecolors='black',
                                           label=f'#{rank+1} Best'))

    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Model Types:'))
    for mt in MODEL_TYPES:
        legend_elements.append(mpatches.Patch(facecolor=MODEL_TYPE_COLORS[mt], edgecolor='black', alpha=0.3,
                                              label=mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG')))

    # Add baseline legend if any metric uses baselines
    has_baselines = any(m[3] is not None for m in metrics_list)
    if baselines and has_baselines:
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label='Baselines:'))
        for bm_name in baselines:
            legend_elements.append(Line2D([0], [0], color=BASELINE_COLORS.get(bm_name, 'gray'),
                                         linestyle='--', linewidth=2,
                                         label=BASELINE_LABELS.get(bm_name, bm_name)))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=9, frameon=True, fancybox=True)

    fig.suptitle(f'{page_title}\nTop 3 models (by DQS) tracked consistently across all plots',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_all_visualizations(csv_path: str, output_dir: str, baseline_dir: str = None):
    """Create all visualization pages."""

    print("Loading grid search results...")
    df = load_results(csv_path)
    print(f"Processed {len(df)} experiments")

    print("\nSelecting top 3 models by DQS (tracked across all plots)...")
    top_models_per_type = get_top_models_by_dqs(df, n=3)
    for mt, top_df in top_models_per_type.items():
        print(f"  {mt.upper()}:")
        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"    #{rank}: {row['name']} (DQS={row['dqs']:.4f})")

    print("\nLoading baseline results (DANN/IRM)...")
    baselines = load_baseline_results(baseline_dir)
    if baselines:
        print(f"Loaded baselines: {list(baselines.keys())}")
        for name, metrics in baselines.items():
            print(f"  {name}: I(Z;Y|D)={metrics['I(Z;Y|D)']:.3f}, I(Z;D|Y)={metrics['I(Z;D|Y)']:.3f}")
    else:
        print("No baseline results found")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create each page
    pages = [
        (PAGE1_METRICS, 'Core Quality Metrics', 'it_page1_quality_metrics.png'),
        (PAGE2_METRICS, 'Class Information (z_y) Metrics + DANN/IRM Baselines', 'it_page2_class_metrics.png'),
        (PAGE3_METRICS, 'Domain Information (z_d) Metrics', 'it_page3_domain_metrics.png'),
        (PAGE4_METRICS, 'Interaction & Joint Metrics', 'it_page4_interaction_metrics.png'),
    ]

    for metrics_list, title, filename in pages:
        print(f"\nCreating: {filename}")
        create_page(df, metrics_list, title, str(output_path / filename),
                   top_models_per_type, baselines)
        print(f"  Saved to: {output_path / filename}")

    # Also create a combined single-page overview with key metrics
    print("\nCreating combined overview...")
    key_metrics = [
        ('dqs', 'DQS', 'Disentanglement Quality', None),
        ('it_partition_quality', 'Partition Quality', 'Overall IT quality', None),
        ('it_I_zy_Y_given_D', 'I(z_y; Y|D)', 'Class capture', 'I(Z;Y|D)'),
        ('it_I_zy_D_given_Y', 'I(z_y; D|Y)', 'Domain leakage', 'I(Z;D|Y)'),
        ('it_I_zd_D_given_Y', 'I(z_d; D|Y)', 'Domain capture', None),
        ('it_I_zdy_Y_D', 'I(z_dy; Y; D)', 'Interaction', None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title, subtitle, baseline_key) in enumerate(key_metrics):
        create_metric_subplot(axes[idx], df, metric, title, subtitle,
                             top_models_per_type, baselines, baseline_key)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='white', label='Top 3 by DQS:'),
    ]
    for rank in range(3):
        legend_elements.append(plt.scatter([], [], c=RANK_COLORS[rank], s=100,
                                           marker=RANK_MARKERS[rank], edgecolors='black',
                                           label=f'#{rank+1} Best'))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
    for mt in MODEL_TYPES:
        legend_elements.append(mpatches.Patch(facecolor=MODEL_TYPE_COLORS[mt], edgecolor='black', alpha=0.3,
                                              label=mt.upper().replace('DANN_AUGMENTED', 'DANN_AUG')))
    if baselines:
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
        for bm_name in baselines:
            legend_elements.append(Line2D([0], [0], color=BASELINE_COLORS.get(bm_name, 'gray'),
                                         linestyle='--', linewidth=2,
                                         label=BASELINE_LABELS.get(bm_name, bm_name)))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=6, fontsize=9, frameon=True, fancybox=True)

    fig.suptitle('Comprehensive IT Analysis Overview - Grid Search Results\n'
                 'Top 3 models (by DQS) tracked consistently; DANN/IRM baselines shown where applicable',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    overview_path = output_path / 'it_comprehensive_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to: {overview_path}")

    # Save updated CSV
    output_csv = output_path / 'all_results_with_dqs.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nSaved updated CSV to: {output_csv}")

    return df, top_models_per_type


def print_top_models_summary(df: pd.DataFrame, top_models_per_type: dict):
    """Print detailed summary of top 3 models per type."""
    print("\n" + "="*100)
    print("TOP 3 MODELS BY DQS - DETAILED SUMMARY")
    print("="*100)

    all_metrics = ['dqs', 'it_partition_quality', 'f1_y', 'f1_d', 'capture_y', 'capture_d',
                   'it_I_zy_Y_given_D', 'it_I_zy_D_given_Y', 'it_I_zd_D_given_Y', 'it_I_zd_Y_given_D',
                   'it_I_zdy_Y_D', 'it_zy_specificity', 'it_zd_specificity']

    for model_type, top_df in top_models_per_type.items():
        print(f"\n{'='*50}")
        print(f"  {model_type.upper()}")
        print(f"{'='*50}")

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"\n  #{rank}: {row['name']}")
            print(f"      Presets: adv={row.get('preset_adversarial', 'N/A')}, "
                  f"cls={row.get('preset_classifier', 'N/A')}, "
                  f"sp={row.get('preset_sparsity', 'N/A')}, "
                  f"kl={row.get('preset_kl', 'N/A')}")
            print(f"      Test Acc: {row['test_accuracy']:.4f}, Gen Gap: {row['gen_gap']:.4f}")
            print(f"      Metrics:")
            for m in all_metrics:
                if pd.notna(row.get(m)):
                    print(f"          {m}: {row[m]:.4f}")


def main():
    csv_path = '/workspace/NVAE/grid_search_results_separate_encoders/summary/all_results.csv'
    output_dir = '/workspace/NVAE/grid_search_results_separate_encoders/summary'
    baseline_dir = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'

    df, top_models = create_all_visualizations(csv_path, output_dir, baseline_dir)
    print_top_models_summary(df, top_models)

    print("\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print(f"  - it_page1_quality_metrics.png     : DQS, PQ, F1_y, F1_d")
    print(f"  - it_page2_class_metrics.png       : z_y metrics + DANN/IRM baselines")
    print(f"  - it_page3_domain_metrics.png      : z_d metrics (decomposed only)")
    print(f"  - it_page4_interaction_metrics.png : Synergy and joint metrics")
    print(f"  - it_comprehensive_overview.png    : Key metrics overview")
    print(f"  - all_results_with_dqs.csv         : Updated results with DQS")


if __name__ == '__main__':
    main()

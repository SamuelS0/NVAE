"""
Visualization functions for information-theoretic evaluation results.

Creates publication-quality plots and tables comparing models on their
adherence to the Minimal Information Partition theorem.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


def plot_model_comparison(
    comparison_results: Dict[str, any],
    output_dir: str,
    figsize: tuple = (16, 10)
):
    """
    Create comprehensive multi-panel comparison plot.

    Args:
        comparison_results: Output from MinimalInformationPartitionEvaluator.compare_models()
        output_dir: Directory to save plots
        figsize: Figure size in inches
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    metrics = comparison_results['metric_comparison']
    models = list(comparison_results['model_scores'].keys())

    # Panel 1: Class latent (z_y) specificity
    ax1 = fig.add_subplot(gs[0, 0])
    plot_latent_specificity(
        ax1,
        metrics,
        models,
        latent_type='z_y',
        title='Class Latent (z_y) Information'
    )

    # Panel 2: Domain latent (z_d) specificity
    ax2 = fig.add_subplot(gs[0, 1])
    plot_latent_specificity(
        ax2,
        metrics,
        models,
        latent_type='z_d',
        title='Domain Latent (z_d) Information'
    )

    # Panel 3: Interaction and residual
    ax3 = fig.add_subplot(gs[1, 0])
    plot_interaction_and_residual(ax3, metrics, models)

    # Panel 4: Overall partition quality
    ax4 = fig.add_subplot(gs[1, 1])
    plot_partition_quality(ax4, comparison_results['model_scores'], comparison_results['rankings'])

    plt.savefig(os.path.join(output_dir, 'it_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Comparison plot saved to: {os.path.join(output_dir, 'it_model_comparison.png')}")


def plot_latent_specificity(
    ax,
    metrics: Dict[str, Dict[str, float]],
    models: List[str],
    latent_type: str,
    title: str
):
    """
    Plot specificity metrics for a latent subspace (z_y or z_d).

    For z_y: Compare I(z_y;Y|D) [high=good] vs I(z_y;D|Y) [low=good]
    For z_d: Compare I(z_d;D|Y) [high=good] vs I(z_d;Y|D) [low=good]
    """
    if latent_type == 'z_y':
        desired_metric = 'I(z_y;Y|D)'  # Should be high
        unwanted_metric = 'I(z_y;D|Y)'  # Should be low
        desired_label = 'I(z_y;Y|D) - Class info'
        unwanted_label = 'I(z_y;D|Y) - Domain leakage'
    else:  # z_d
        desired_metric = 'I(z_d;D|Y)'  # Should be high
        unwanted_metric = 'I(z_d;Y|D)'  # Should be low
        desired_label = 'I(z_d;D|Y) - Domain info'
        unwanted_label = 'I(z_d;Y|D) - Class leakage'

    desired_values = [metrics[desired_metric].get(model, 0.0) for model in models]
    unwanted_values = [metrics[unwanted_metric].get(model, 0.0) for model in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, desired_values, width, label=desired_label,
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, unwanted_values, width, label=unwanted_label,
                   alpha=0.8, color='coral')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mutual Information (nats)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_interaction_and_residual(
    ax,
    metrics: Dict[str, Dict[str, float]],
    models: List[str]
):
    """
    Plot interaction information I(z_dy;Y;D) and residual information I(z_x;Y,D).
    """
    # Handle None values (null in JSON) by converting to 0.0
    interaction_values = [metrics['I(z_dy;Y;D)'].get(model, 0.0) or 0.0 for model in models]
    residual_values = [metrics['I(z_x;Y,D)'].get(model, 0.0) or 0.0 for model in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, interaction_values, width,
                   label='I(z_dy;Y;D) - Interaction (higher for NVAE/AugDANN)',
                   alpha=0.8, color='mediumseagreen')
    bars2 = ax.bar(x + width/2, residual_values, width,
                   label='I(z_x;Y,D) - Residual (should be low)',
                   alpha=0.8, color='indianred')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mutual Information (nats)', fontsize=11, fontweight='bold')
    ax.set_title('Interaction and Residual Information', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def plot_partition_quality(
    ax,
    model_scores: Dict[str, float],
    rankings: Dict[str, int]
):
    """
    Plot overall partition quality scores with rankings.
    """
    models = list(model_scores.keys())
    scores = [model_scores[model] for model in models]

    # Color bars based on score (green=good, yellow=ok, red=poor)
    colors = []
    for score in scores:
        if score >= 0.7:
            colors.append('mediumseagreen')
        elif score >= 0.5:
            colors.append('gold')
        elif score >= 0.3:
            colors.append('orange')
        else:
            colors.append('indianred')

    bars = ax.barh(models, scores, color=colors, alpha=0.8)

    # Add score labels and rankings
    for i, (model, score, bar) in enumerate(zip(models, scores, bars)):
        rank = rankings[model]
        ax.text(score + 0.02, i, f'{score:.3f} (Rank {rank})',
               va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Partition Quality Score', fontsize=11, fontweight='bold')
    ax.set_title('Overall Partition Quality (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)

    # Add reference lines
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Excellent (>0.7)')
    ax.axvline(x=0.5, color='gold', linestyle='--', linewidth=1, alpha=0.3, label='Good (>0.5)')
    ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Fair (>0.3)')


def create_comparison_table(
    comparison_results: Dict[str, any],
    output_dir: str,
    include_confidence_intervals: bool = True
):
    """
    Create detailed comparison table in both CSV and LaTeX formats.

    Args:
        comparison_results: Output from compare_models()
        output_dir: Directory to save tables
        include_confidence_intervals: Whether to include CI columns
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = comparison_results['metric_comparison']
    models = list(comparison_results['model_scores'].keys())

    # Metrics to include in table
    metric_names = [
        'I(z_y;Y|D)', 'I(z_y;D|Y)', 'z_y_specificity',
        'I(z_d;D|Y)', 'I(z_d;Y|D)', 'z_d_specificity',
        'I(z_dy;Y;D)', 'I(z_x;Y,D)', 'partition_quality'
    ]

    metric_labels = {
        'I(z_y;Y|D)': 'I(z_y;Y|D) ↑',
        'I(z_y;D|Y)': 'I(z_y;D|Y) ↓',
        'z_y_specificity': 'z_y Specificity ↑',
        'I(z_d;D|Y)': 'I(z_d;D|Y) ↑',
        'I(z_d;Y|D)': 'I(z_d;Y|D) ↓',
        'z_d_specificity': 'z_d Specificity ↑',
        'I(z_dy;Y;D)': 'I(z_dy;Y;D)',
        'I(z_x;Y,D)': 'I(z_x;Y,D) ↓',
        'partition_quality': 'Quality Score ↑'
    }

    # Build DataFrame
    data = []
    for metric in metric_names:
        row = {'Metric': metric_labels.get(metric, metric)}
        for model in models:
            value = metrics[metric].get(model, 0.0) or 0.0  # Handle None values

            if include_confidence_intervals and f'{model}_CI' in metrics[metric]:
                ci = metrics[metric][f'{model}_CI']
                row[model] = f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
            else:
                row[model] = f"{value:.4f}"

        data.append(row)

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = os.path.join(output_dir, 'it_comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"📄 CSV table saved to: {csv_path}")

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'it_comparison_table.tex')
    latex_table = df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * len(models),
        caption='Information-Theoretic Evaluation of Model Partitions. '
                'Arrows indicate desired direction (↑=higher is better, ↓=lower is better).',
        label='tab:it_evaluation'
    )

    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"📄 LaTeX table saved to: {latex_path}")

    return df


def plot_heatmap(
    comparison_results: Dict[str, any],
    output_dir: str,
    figsize: tuple = (12, 8)
):
    """
    Create heatmap of all information quantities across models.

    Args:
        comparison_results: Output from compare_models()
        output_dir: Directory to save plot
        figsize: Figure size in inches
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = comparison_results['metric_comparison']
    models = list(comparison_results['model_scores'].keys())

    # Select metrics for heatmap
    metric_names = [
        'I(z_y;Y|D)', 'I(z_y;D|Y)', 'I(z_y;Y)', 'I(z_y;D)',
        'I(z_d;D|Y)', 'I(z_d;Y|D)', 'I(z_d;D)', 'I(z_d;Y)',
        'I(z_dy;Y;D)', 'I(z_x;Y,D)',
        'z_y_specificity', 'z_d_specificity', 'partition_quality'
    ]

    # Build matrix
    data_matrix = []
    display_names = []
    for metric in metric_names:
        if metric in metrics:
            # Handle None values (null in JSON) by converting to 0.0
            row = [metrics[metric].get(model, 0.0) or 0.0 for model in models]
            data_matrix.append(row)
            display_names.append(metric)

    data_matrix = np.array(data_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize each row for better visualization
    data_normalized = np.zeros_like(data_matrix)
    for i in range(data_matrix.shape[0]):
        row_min = data_matrix[i].min()
        row_max = data_matrix[i].max()
        if row_max > row_min:
            data_normalized[i] = (data_matrix[i] - row_min) / (row_max - row_min)
        else:
            data_normalized[i] = 0.5  # Center if all values are the same

    sns.heatmap(
        data_normalized,
        xticklabels=models,
        yticklabels=display_names,
        annot=data_matrix,  # Show actual values
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        cbar_kws={'label': 'Normalized Score'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('Information-Theoretic Metrics Heatmap\n(Green=Better for that metric)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Information-Theoretic Quantity', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'it_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"🗺️  Heatmap saved to: {heatmap_path}")


def generate_summary_report(
    comparison_results: Dict[str, any],
    output_dir: str
):
    """
    Generate a text summary report of the comparison.

    Args:
        comparison_results: Output from compare_models()
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'it_summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INFORMATION-THEORETIC EVALUATION SUMMARY\n")
        f.write("Minimal Information Partition Theorem Adherence\n")
        f.write("="*80 + "\n\n")

        # Overall rankings
        f.write("OVERALL PARTITION QUALITY RANKINGS\n")
        f.write("-"*80 + "\n")
        sorted_models = sorted(
            comparison_results['model_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (model, score) in enumerate(sorted_models, 1):
            f.write(f"{rank}. {model:<20} Score: {score:.4f}\n")

        f.write("\n" + "="*80 + "\n\n")

        # Detailed metrics
        f.write("DETAILED METRIC COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<25} ")
        models = list(comparison_results['model_scores'].keys())
        for model in models:
            f.write(f"{model:<15} ")
        f.write("\n" + "-"*80 + "\n")

        metrics = comparison_results['metric_comparison']
        key_metrics = [
            'I(z_y;Y|D)', 'I(z_y;D|Y)', 'I(z_d;D|Y)', 'I(z_d;Y|D)',
            'I(z_dy;Y;D)', 'I(z_x;Y,D)', 'partition_quality'
        ]

        for metric in key_metrics:
            f.write(f"{metric:<25} ")
            for model in models:
                value = metrics[metric].get(model, 0.0) or 0.0  # Handle None values
                f.write(f"{value:<15.4f} ")
            f.write("\n")

        f.write("\n" + "="*80 + "\n\n")

        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-"*80 + "\n")
        f.write("For good adherence to Minimal Information Partition:\n")
        f.write("  ✓ I(z_y;Y|D) should be HIGH - class latent captures class info\n")
        f.write("  ✓ I(z_y;D|Y) should be LOW - class latent has minimal domain leakage\n")
        f.write("  ✓ I(z_d;D|Y) should be HIGH - domain latent captures domain info\n")
        f.write("  ✓ I(z_d;Y|D) should be LOW - domain latent has minimal class leakage\n")
        f.write("  ✓ I(z_dy;Y;D) should be POSITIVE for models with z_ay (NVAE/AugDANN)\n")
        f.write("  ✓ I(z_x;Y,D) should be LOW - residual has minimal label information\n")
        f.write("\n")

        best_model = sorted_models[0][0]
        f.write(f"Best performing model: {best_model}\n")
        f.write(f"This model best adheres to the Minimal Information Partition definition.\n")

        f.write("\n" + "="*80 + "\n")

    print(f"📋 Summary report saved to: {report_path}")


def visualize_all(
    comparison_results: Dict[str, any],
    output_dir: str
):
    """
    Generate all visualizations and reports.

    Args:
        comparison_results: Output from compare_models()
        output_dir: Directory to save all outputs
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("="*80 + "\n")

    plot_model_comparison(comparison_results, output_dir)
    plot_heatmap(comparison_results, output_dir)
    create_comparison_table(comparison_results, output_dir)
    generate_summary_report(comparison_results, output_dir)

    print("\n" + "="*80)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("="*80 + "\n")

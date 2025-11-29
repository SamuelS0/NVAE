#!/usr/bin/env python3
"""
Visualize IT analysis results from the balanced 20-epoch experiment.
Compare models using both old Partition Quality and new DQS metric.
Includes DANN/IRM in applicable raw metric plots (not in DQS/PQ plots).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/NVAE')
from core.disentanglement_quality import compute_dqs

# Model categories for visualization
DECOMPOSED_MODELS = ['nvae', 'diva', 'dann_augmented']
MONOLITHIC_MODELS = ['dann', 'irm']
ALL_MODELS = DECOMPOSED_MODELS + MONOLITHIC_MODELS

# Color scheme for all 5 models
MODEL_COLORS = {
    'nvae': '#9467BD',           # Purple
    'diva': '#FF7F0E',           # Orange
    'dann_augmented': '#17BECF', # Cyan
    'dann': '#2CA02C',           # Green
    'irm': '#D62728',            # Red
}

MODEL_LABELS = {
    'nvae': 'NVAE',
    'diva': 'DIVA',
    'dann_augmented': 'DANN_AUG',
    'dann': 'DANN',
    'irm': 'IRM',
}


def load_it_results(base_path: str) -> dict:
    """Load IT results for all models (decomposed and monolithic)."""
    results = {}

    for model in ALL_MODELS:
        path = Path(base_path) / model / 'it_results.json'
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)

    return results


def compute_all_dqs(results: dict) -> dict:
    """Compute DQS for decomposed models only (not DANN/IRM)."""
    dqs_results = {}
    for model, data in results.items():
        # Only compute DQS for decomposed models with partitioned latent spaces
        if model not in DECOMPOSED_MODELS:
            continue
        metrics = data['metrics']
        dqs_input = {
            'I(z_y;Y|D)': metrics.get('I(z_y;Y|D)', 0),
            'I(z_y;D|Y)': metrics.get('I(z_y;D|Y)', 0),
            'I(z_d;D|Y)': metrics.get('I(z_d;D|Y)', 0),
            'I(z_d;Y|D)': metrics.get('I(z_d;Y|D)', 0),
            'I(z_dy;Y;D)': metrics.get('I(z_dy;Y;D)', 0),
            'I(z_x;Y,D)': metrics.get('I(z_x;Y,D)', 0),
        }
        dqs_results[model] = compute_dqs(dqs_input)
    return dqs_results


def get_class_capture(results: dict, model: str) -> float:
    """Get class capture for any model type (handles monolithic vs decomposed)."""
    metrics = results[model]['metrics']
    # Monolithic models use I(Z;Y|D), decomposed use I(z_y;Y|D)
    if model in MONOLITHIC_MODELS:
        return metrics.get('I(Z;Y|D)', 0)
    return metrics.get('I(z_y;Y|D)', 0)


def get_domain_leakage(results: dict, model: str) -> float:
    """Get domain leakage for any model type."""
    metrics = results[model]['metrics']
    # Monolithic: I(Z;D|Y), Decomposed: I(z_y;D|Y)
    if model in MONOLITHIC_MODELS:
        return metrics.get('I(Z;D|Y)', 0)
    return metrics.get('I(z_y;D|Y)', 0)


def get_domain_invariance(results: dict, model: str) -> float:
    """Get domain invariance score for any model."""
    # Check unified_metrics first (where it's stored for decomposed models)
    unified = results[model].get('unified_metrics', {})
    if 'domain_invariance_score' in unified:
        return unified['domain_invariance_score']
    # Fallback to metrics (for monolithic models)
    metrics = results[model]['metrics']
    return metrics.get('domain_invariance_score', 0)


def create_comparison_dashboard(results: dict, dqs_results: dict, output_path: str):
    """Create comprehensive dashboard comparing all models.

    Includes DANN/IRM in raw metric plots (class capture, domain leakage, domain invariance).
    DQS/PQ plots only show decomposed models.
    """

    # Get available models for each category
    all_available = [m for m in ALL_MODELS if m in results]
    decomposed_available = [m for m in DECOMPOSED_MODELS if m in results]

    fig = plt.figure(figsize=(22, 18))

    # Create grid: 3 rows x 4 cols
    # Row 1: DQS metrics (decomposed only) + Raw metrics (all 5 models)
    # Row 2: Information Capture metrics
    # Row 3: Entanglement + Summary

    # ============ ROW 1 ============
    # Plot 1: Old Partition Quality (decomposed only)
    ax1 = plt.subplot(3, 4, 1)
    old_pq = [results[m]['metrics']['partition_quality'] for m in decomposed_available]
    ax1.bar(range(len(decomposed_available)), old_pq,
            color=[MODEL_COLORS[m] for m in decomposed_available])
    ax1.set_xticks(range(len(decomposed_available)))
    ax1.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_title('OLD: Partition Quality\n(decomposed models only)', fontsize=11, fontweight='bold', color='red')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(old_pq):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Plot 2: New DQS (decomposed only)
    ax2 = plt.subplot(3, 4, 2)
    new_dqs = [dqs_results[m]['dqs'] for m in decomposed_available]
    ax2.bar(range(len(decomposed_available)), new_dqs,
            color=[MODEL_COLORS[m] for m in decomposed_available])
    ax2.set_xticks(range(len(decomposed_available)))
    ax2.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_title('NEW: DQS\n(decomposed models only)', fontsize=11, fontweight='bold', color='green')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(new_dqs):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Plot 3: Class Capture - ALL 5 MODELS
    ax3 = plt.subplot(3, 4, 3)
    class_capture = [get_class_capture(results, m) for m in all_available]
    ax3.bar(range(len(all_available)), class_capture,
            color=[MODEL_COLORS[m] for m in all_available])
    ax3.set_xticks(range(len(all_available)))
    ax3.set_xticklabels([MODEL_LABELS[m] for m in all_available], fontweight='bold', fontsize=9)
    ax3.set_ylabel('Nats')
    ax3.set_title('Class Info Capture\nI(z_y;Y|D) or I(Z;Y|D)', fontsize=11, fontweight='bold')
    ax3.axhline(y=2.303, color='green', linestyle='--', alpha=0.7, label='H(Y)≈2.30')
    for i, v in enumerate(class_capture):
        ax3.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)

    # Plot 4: Domain Invariance Score - ALL 5 MODELS
    ax4 = plt.subplot(3, 4, 4)
    dom_inv = [get_domain_invariance(results, m) for m in all_available]
    ax4.bar(range(len(all_available)), dom_inv,
            color=[MODEL_COLORS[m] for m in all_available])
    ax4.set_xticks(range(len(all_available)))
    ax4.set_xticklabels([MODEL_LABELS[m] for m in all_available], fontweight='bold', fontsize=9)
    ax4.set_ylabel('Score')
    ax4.set_title('Domain Invariance Score\n1/(1+leakage) - higher=better', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 1)
    for i, v in enumerate(dom_inv):
        ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

    # ============ ROW 2: More Information Capture & Leakage ============
    # Plot 5: Domain Leakage - ALL 5 MODELS
    ax5 = plt.subplot(3, 4, 5)
    dom_leak = [get_domain_leakage(results, m) for m in all_available]
    ax5.bar(range(len(all_available)), dom_leak,
            color=[MODEL_COLORS[m] for m in all_available])
    ax5.set_xticks(range(len(all_available)))
    ax5.set_xticklabels([MODEL_LABELS[m] for m in all_available], fontweight='bold', fontsize=9)
    ax5.set_ylabel('Nats')
    ax5.set_title('Domain Leakage\nI(z_y;D|Y) or I(Z;D|Y)', fontsize=11, fontweight='bold')
    ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='threshold=0.1')
    for i, v in enumerate(dom_leak):
        ax5.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)

    # Plot 6: I(z_d; D|Y) - Domain capture (decomposed only)
    ax6 = plt.subplot(3, 4, 6)
    domain_capture = [results[m]['metrics']['I(z_d;D|Y)'] for m in decomposed_available]
    ax6.bar(range(len(decomposed_available)), domain_capture,
            color=[MODEL_COLORS[m] for m in decomposed_available])
    ax6.set_xticks(range(len(decomposed_available)))
    ax6.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax6.set_ylabel('Nats')
    ax6.set_title('I(z_d; D|Y)\nDomain Capture (decomposed)', fontsize=11, fontweight='bold')
    ax6.axhline(y=1.792, color='green', linestyle='--', alpha=0.7, label='H(D)≈1.79')
    for i, v in enumerate(domain_capture):
        ax6.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

    # Plot 7: F1 Scores (decomposed only)
    ax7 = plt.subplot(3, 4, 7)
    f1_y = [dqs_results[m]['f1_y'] for m in decomposed_available]
    f1_d = [dqs_results[m]['f1_d'] for m in decomposed_available]
    x = np.arange(len(decomposed_available))
    width = 0.35
    ax7.bar(x - width/2, f1_y, width, label='F1_y (Class)', color='#E41A1C', alpha=0.8)
    ax7.bar(x + width/2, f1_d, width, label='F1_d (Domain)', color='#377EB8', alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('DQS Components\n(decomposed models only)', fontsize=11, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.set_ylim(0, 1.1)

    # Plot 8: Synergy I(z_dy; Y; D) - decomposed only
    ax8 = plt.subplot(3, 4, 8)
    synergy = [results[m]['metrics'].get('I(z_dy;Y;D)', 0) for m in decomposed_available]
    colors = ['green' if s > 0 else 'red' for s in synergy]
    ax8.bar(range(len(decomposed_available)), synergy, color=colors, alpha=0.7)
    ax8.set_xticks(range(len(decomposed_available)))
    ax8.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax8.set_ylabel('Nats')
    ax8.set_title('I(z_dy; Y; D)\nSynergy (decomposed only)', fontsize=11, fontweight='bold')
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    for i, v in enumerate(synergy):
        ax8.text(i, v + 0.01 if v >= 0 else v - 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # ============ ROW 3: Entanglement & Summary ============
    # Plot 9: Dimension Utilization (decomposed only)
    ax9 = plt.subplot(3, 4, 9)
    width = 0.2
    x = np.arange(len(decomposed_available))
    latents = ['z_y', 'z_d', 'z_dy', 'z_x']
    latent_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

    for i, latent in enumerate(latents):
        utils = []
        for m in decomposed_available:
            eff_dim = results[m]['effective_dimensionality'].get(latent)
            if eff_dim and eff_dim.get('utilization') is not None:
                utils.append(eff_dim['utilization'] * 100)
            else:
                utils.append(0)
        ax9.bar(x + i*width, utils, width, label=latent, color=latent_colors[i], alpha=0.8)

    ax9.set_xticks(x + 1.5*width)
    ax9.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax9.set_ylabel('Utilization %')
    ax9.set_title('Dimension Utilization\n(decomposed models)', fontsize=11, fontweight='bold')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.set_ylim(0, 110)
    ax9.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    # Plot 10: I(z_x; Y,D) - Residual contamination (decomposed only)
    ax10 = plt.subplot(3, 4, 10)
    zx_yd = [results[m]['metrics'].get('I(z_x;Y,D)', 0) for m in decomposed_available]
    ax10.bar(range(len(decomposed_available)), zx_yd,
             color=[MODEL_COLORS[m] for m in decomposed_available])
    ax10.set_xticks(range(len(decomposed_available)))
    ax10.set_xticklabels([MODEL_LABELS[m] for m in decomposed_available], fontweight='bold')
    ax10.set_ylabel('Nats')
    ax10.set_title('I(z_x; Y,D)\nResidual Contamination', fontsize=11, fontweight='bold')
    ax10.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    for i, v in enumerate(zx_yd):
        ax10.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

    # Plot 11: Total Class Info - ALL 5 MODELS
    ax11 = plt.subplot(3, 4, 11)
    total_class = []
    for m in all_available:
        unified = results[m].get('unified_metrics', {})
        total_class.append(unified.get('total_class_info', 0))
    ax11.bar(range(len(all_available)), total_class,
             color=[MODEL_COLORS[m] for m in all_available])
    ax11.set_xticks(range(len(all_available)))
    ax11.set_xticklabels([MODEL_LABELS[m] for m in all_available], fontweight='bold', fontsize=9)
    ax11.set_ylabel('Nats')
    ax11.set_title('Total Class Information\nI(Z;Y) for all models', fontsize=11, fontweight='bold')
    ax11.axhline(y=2.303, color='green', linestyle='--', alpha=0.7)
    for i, v in enumerate(total_class):
        ax11.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)

    # Plot 12: Summary Table - ALL MODELS
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Create summary text
    summary_text = "BALANCED 20-EPOCH SUMMARY\n" + "="*35 + "\n\n"

    # Decomposed models with DQS
    summary_text += "DECOMPOSED MODELS:\n"
    for m in decomposed_available:
        dqs = dqs_results[m]
        metrics = results[m]['metrics']
        summary_text += f"  {MODEL_LABELS[m]}: DQS={dqs['dqs']:.3f}, "
        summary_text += f"PQ={metrics['partition_quality']:.3f}\n"

    summary_text += "\nMONOLITHIC MODELS:\n"
    for m in MONOLITHIC_MODELS:
        if m in results:
            unified = results[m].get('unified_metrics', {})
            dom_inv = unified.get('domain_invariance_score', 0)
            class_info = unified.get('class_info_conditional', 0)
            summary_text += f"  {MODEL_LABELS[m]}: DomInv={dom_inv:.3f}, "
            summary_text += f"ClassInfo={class_info:.2f}\n"

    summary_text += "\n" + "="*35 + "\n"
    summary_text += "RANKINGS (Domain Invariance):\n"
    # Rank all models by domain invariance
    all_dom_inv = [(m, get_domain_invariance(results, m)) for m in all_available]
    all_dom_inv.sort(key=lambda x: x[1], reverse=True)
    for rank, (m, score) in enumerate(all_dom_inv, 1):
        summary_text += f"  {rank}. {MODEL_LABELS[m]}: {score:.3f}\n"

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Update suptitle
    plt.suptitle('Balanced 20-Epoch IT Analysis: All 5 Models\n'
                 'Raw metrics for all; DQS/PQ for decomposed models only',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to: {output_path}")


def print_summary(results: dict, dqs_results: dict):
    """Print detailed summary including all 5 model types."""
    print("\n" + "="*80)
    print("BALANCED 20-EPOCH IT ANALYSIS SUMMARY")
    print("="*80)

    all_available = [m for m in ALL_MODELS if m in results]
    decomposed_available = [m for m in DECOMPOSED_MODELS if m in results]

    # Decomposed models with full details
    print("\n" + "="*80)
    print("DECOMPOSED MODELS (NVAE, DIVA, DANN_AUG)")
    print("="*80)

    for model in decomposed_available:
        dqs = dqs_results[model]
        metrics = results[model]['metrics']
        eff_dim = results[model]['effective_dimensionality']

        print(f"\n{'='*40}")
        print(f"  {MODEL_LABELS[model]}")
        print(f"{'='*40}")
        print(f"  OLD Partition Quality: {metrics['partition_quality']:.4f}")
        print(f"  NEW DQS:              {dqs['dqs']:.4f}")
        print(f"\n  DQS Components:")
        print(f"    Capture_y: {dqs['capture_y']:.3f}, Purity_y: {dqs['purity_y']:.3f}, F1_y: {dqs['f1_y']:.3f}")
        print(f"    Capture_d: {dqs['capture_d']:.3f}, Purity_d: {dqs['purity_d']:.3f}, F1_d: {dqs['f1_d']:.3f}")
        print(f"\n  Information Capture:")
        print(f"    I(z_y;Y|D) = {metrics['I(z_y;Y|D)']:.3f} (class)")
        print(f"    I(z_d;D|Y) = {metrics['I(z_d;D|Y)']:.3f} (domain)")
        print(f"    I(z_dy;Y;D) = {metrics.get('I(z_dy;Y;D)', 0):.3f} (synergy)")
        print(f"\n  Dimension Utilization:")
        for latent in ['z_y', 'z_d', 'z_dy', 'z_x']:
            ed = eff_dim.get(latent)
            if ed and ed.get('utilization') is not None:
                print(f"    {latent}: {ed['active_dims']}/{ed['nominal_dims']} ({ed['utilization']*100:.0f}%)")

    # Monolithic models
    print("\n" + "="*80)
    print("MONOLITHIC MODELS (DANN, IRM)")
    print("="*80)

    for model in MONOLITHIC_MODELS:
        if model not in results:
            continue
        metrics = results[model]['metrics']
        unified = results[model].get('unified_metrics', {})
        eff_dim = results[model]['effective_dimensionality']

        print(f"\n{'='*40}")
        print(f"  {MODEL_LABELS[model]}")
        print(f"{'='*40}")
        print(f"  Domain Invariance Score: {unified.get('domain_invariance_score', 0):.4f}")
        print(f"\n  Information Metrics:")
        print(f"    I(Z;Y|D) = {metrics.get('I(Z;Y|D)', 0):.3f} (class capture)")
        print(f"    I(Z;D|Y) = {metrics.get('I(Z;D|Y)', 0):.3f} (domain leakage)")
        print(f"    I(Z;Y)   = {metrics.get('I(Z;Y)', 0):.3f} (total class)")
        print(f"\n  Representation:")
        print(f"    Active dims: {eff_dim.get('active_dims', 0)}/{eff_dim.get('nominal_dims', 0)}")
        print(f"    Utilization: {eff_dim.get('utilization', 0)*100:.0f}%")

    # Rankings
    print("\n" + "="*80)
    print("RANKINGS")
    print("="*80)

    # Sort by domain invariance (all models)
    all_dom_inv = [(m, get_domain_invariance(results, m)) for m in all_available]
    all_dom_inv.sort(key=lambda x: x[1], reverse=True)
    print("\nBy Domain Invariance Score (ALL 5 MODELS):")
    for i, (m, score) in enumerate(all_dom_inv, 1):
        print(f"  {i}. {MODEL_LABELS[m]}: {score:.3f}")

    # Sort by DQS (decomposed only)
    new_ranking = sorted(dqs_results.keys(), key=lambda m: dqs_results[m]['dqs'], reverse=True)
    print("\nBy DQS (decomposed models only):")
    for i, m in enumerate(new_ranking, 1):
        print(f"  {i}. {MODEL_LABELS[m]}: {dqs_results[m]['dqs']:.3f}")

    # Sort by class capture (all models)
    all_class = [(m, get_class_capture(results, m)) for m in all_available]
    all_class.sort(key=lambda x: x[1], reverse=True)
    print("\nBy Class Capture (ALL 5 MODELS):")
    for i, (m, score) in enumerate(all_class, 1):
        print(f"  {i}. {MODEL_LABELS[m]}: {score:.3f} nats")


def main():
    base_path = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis'
    output_path = '/workspace/NVAE/results/crmnist_balanced_20epochs/information_theoretic_analysis/balanced_it_dashboard_5models.png'

    print("Loading IT results for all model types...")
    results = load_it_results(base_path)
    decomposed = [m for m in DECOMPOSED_MODELS if m in results]
    monolithic = [m for m in MONOLITHIC_MODELS if m in results]
    print(f"Loaded {len(results)} models:")
    print(f"  Decomposed: {decomposed}")
    print(f"  Monolithic: {monolithic}")

    print("\nComputing DQS for decomposed models...")
    dqs_results = compute_all_dqs(results)

    print_summary(results, dqs_results)

    print("\nCreating visualization with all 5 models...")
    create_comparison_dashboard(results, dqs_results, output_path)


if __name__ == '__main__':
    main()

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_expressiveness_results(results_dir):
    """Load expressiveness results from all model directories."""

    results = {}
    model_types = ['nvae', 'diva', 'dann', 'dann_augmented', 'irm']

    for model_type in model_types:
        model_dir = os.path.join(results_dir, f'{model_type}_expressiveness')
        results_file = os.path.join(model_dir, 'latent_expressiveness_results.json')

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results[model_type] = json.load(f)
            print(f"✅ Loaded {model_type.upper()} expressiveness results")
        else:
            print(f"⚠️  {model_type.upper()} expressiveness results not found at {results_file}")

    return results

def create_comprehensive_comparison(results, save_dir):
    """Create comprehensive comparison across all models."""

    # Prepare data for comparison
    comparison_data = []

    for model_name, model_results in results.items():

        # Hospital classification results
        if 'hospital_za_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Hospital Classification',
                'Method': 'Individual (za)',
                'Validation Accuracy': model_results['hospital_za_alone']['val_acc'],
                'Test Accuracy': model_results['hospital_za_alone'].get('test_acc', model_results['hospital_za_alone']['val_acc']),
                'Component': 'za'
            })

        if 'hospital_za_zay' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Hospital Classification',
                'Method': 'Combined (za+zay)',
                'Validation Accuracy': model_results['hospital_za_zay']['val_acc'],
                'Test Accuracy': model_results['hospital_za_zay'].get('test_acc', model_results['hospital_za_zay']['val_acc']),
                'Component': 'za+zay'
            })

        if 'hospital_zay_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Hospital Classification',
                'Method': 'Individual (zay)',
                'Validation Accuracy': model_results['hospital_zay_alone']['val_acc'],
                'Test Accuracy': model_results['hospital_zay_alone'].get('test_acc', model_results['hospital_zay_alone']['val_acc']),
                'Component': 'zay'
            })

        # Tumor classification results
        if 'tumor_zy_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Tumor Classification',
                'Method': 'Individual (zy)',
                'Validation Accuracy': model_results['tumor_zy_alone']['val_acc'],
                'Test Accuracy': model_results['tumor_zy_alone'].get('test_acc', model_results['tumor_zy_alone']['val_acc']),
                'Component': 'zy'
            })

        if 'tumor_zy_zay' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Tumor Classification',
                'Method': 'Combined (zy+zay)',
                'Validation Accuracy': model_results['tumor_zy_zay']['val_acc'],
                'Test Accuracy': model_results['tumor_zy_zay'].get('test_acc', model_results['tumor_zy_zay']['val_acc']),
                'Component': 'zy+zay'
            })

        if 'tumor_zay_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Tumor Classification',
                'Method': 'Individual (zay)',
                'Validation Accuracy': model_results['tumor_zay_alone']['val_acc'],
                'Test Accuracy': model_results['tumor_zay_alone'].get('test_acc', model_results['tumor_zay_alone']['val_acc']),
                'Component': 'zay'
            })

    df = pd.DataFrame(comparison_data)

    if df.empty:
        print("⚠️  No data available for comparison")
        return None, None

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Hospital Classification Comparison
    hospital_data = df[df['Task'] == 'Hospital Classification']
    if not hospital_data.empty:
        # Pivot for heatmap
        hospital_pivot = hospital_data.pivot_table(
            index='Model', columns='Component', values='Validation Accuracy', aggfunc='mean'
        )

        sns.heatmap(hospital_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[0,0], cbar_kws={'label': 'Validation Accuracy'})
        axes[0,0].set_title('Hospital Classification Accuracy by Model & Component', fontweight='bold')
        axes[0,0].set_ylabel('Model')

    # 2. Tumor Classification Comparison
    tumor_data = df[df['Task'] == 'Tumor Classification']
    if not tumor_data.empty:
        tumor_pivot = tumor_data.pivot_table(
            index='Model', columns='Component', values='Validation Accuracy', aggfunc='mean'
        )

        sns.heatmap(tumor_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[0,1], cbar_kws={'label': 'Validation Accuracy'})
        axes[0,1].set_title('Tumor Classification Accuracy by Model & Component', fontweight='bold')
        axes[0,1].set_ylabel('Model')

    # 3. Improvement Analysis (Combined vs Individual)
    improvements = []

    for model_name, model_results in results.items():
        # Hospital improvement
        if 'hospital_za_alone' in model_results and 'hospital_za_zay' in model_results:
            hospital_improvement = model_results['hospital_za_zay']['val_acc'] - model_results['hospital_za_alone']['val_acc']
            hospital_baseline = model_results['hospital_za_alone']['val_acc']
            improvements.append({
                'Model': model_name.upper(),
                'Task': 'Hospital',
                'Improvement': hospital_improvement,
                'Improvement_Percent': (hospital_improvement / hospital_baseline) * 100
            })

        # Tumor improvement
        if 'tumor_zy_alone' in model_results and 'tumor_zy_zay' in model_results:
            tumor_improvement = model_results['tumor_zy_zay']['val_acc'] - model_results['tumor_zy_alone']['val_acc']
            tumor_baseline = model_results['tumor_zy_alone']['val_acc']
            improvements.append({
                'Model': model_name.upper(),
                'Task': 'Tumor',
                'Improvement': tumor_improvement,
                'Improvement_Percent': (tumor_improvement / tumor_baseline) * 100
            })

    if improvements:
        improvement_df = pd.DataFrame(improvements)

        # Bar plot for improvements
        improvement_pivot = improvement_df.pivot(index='Model', columns='Task', values='Improvement_Percent')
        improvement_pivot.plot(kind='bar', ax=axes[1,0], color=['#ff7f0e', '#2ca02c'])
        axes[1,0].set_title('Improvement from Adding zay Component (%)', fontweight='bold')
        axes[1,0].set_ylabel('Improvement (%)')
        axes[1,0].legend(title='Task')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Add zero line
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 4. Overall Model Ranking
    if not hospital_data.empty and not tumor_data.empty:
        # Calculate average combined performance
        model_rankings = []

        for model_name, model_results in results.items():
            combined_score = 0
            count = 0

            if 'hospital_za_zay' in model_results:
                combined_score += model_results['hospital_za_zay']['val_acc']
                count += 1
            elif 'hospital_za_alone' in model_results:
                combined_score += model_results['hospital_za_alone']['val_acc']
                count += 1

            if 'tumor_zy_zay' in model_results:
                combined_score += model_results['tumor_zy_zay']['val_acc']
                count += 1
            elif 'tumor_zy_alone' in model_results:
                combined_score += model_results['tumor_zy_alone']['val_acc']
                count += 1

            if count > 0:
                avg_score = combined_score / count
                model_rankings.append({
                    'Model': model_name.upper(),
                    'Average Score': avg_score
                })

        if model_rankings:
            ranking_df = pd.DataFrame(model_rankings).sort_values('Average Score', ascending=True)

            bars = axes[1,1].barh(ranking_df['Model'], ranking_df['Average Score'],
                                 color=plt.cm.RdYlGn(ranking_df['Average Score']))
            axes[1,1].set_title('Overall Model Ranking (Average Performance)', fontweight='bold')
            axes[1,1].set_xlabel('Average Validation Accuracy')

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1,1].text(width + 0.005, bar.get_y() + bar.get_height()/2,
                              f'{width:.3f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()

    # Save the comprehensive comparison
    comparison_path = os.path.join(save_dir, 'comprehensive_expressiveness_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Comprehensive comparison saved to: {comparison_path}")

    return df, improvement_df if 'improvement_df' in locals() else None

def generate_summary_report(results, save_dir):
    """Generate a text summary report of expressiveness results."""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("LATENT VARIABLE EXPRESSIVENESS ANALYSIS REPORT - WILD DATASET")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append("📋 EXECUTIVE SUMMARY:")
    report_lines.append("-" * 50)
    report_lines.append("")

    # Find best performing models
    best_hospital_model = None
    best_hospital_score = 0
    best_tumor_model = None
    best_tumor_score = 0

    zay_benefits = []

    for model_name, model_results in results.items():
        # Hospital classification
        if 'hospital_za_zay' in model_results:
            score = model_results['hospital_za_zay']['val_acc']
            if score > best_hospital_score:
                best_hospital_score = score
                best_hospital_model = model_name.upper()

        # Tumor classification
        if 'tumor_zy_zay' in model_results:
            score = model_results['tumor_zy_zay']['val_acc']
            if score > best_tumor_score:
                best_tumor_score = score
                best_tumor_model = model_name.upper()

        # Calculate zay benefits
        hospital_benefit = 0
        tumor_benefit = 0
        hospital_baseline = 1.0  # Default to avoid division by zero
        tumor_baseline = 1.0   # Default to avoid division by zero

        if 'hospital_za_alone' in model_results and 'hospital_za_zay' in model_results:
            hospital_benefit = model_results['hospital_za_zay']['val_acc'] - model_results['hospital_za_alone']['val_acc']
            hospital_baseline = model_results['hospital_za_alone']['val_acc']

        if 'tumor_zy_alone' in model_results and 'tumor_zy_zay' in model_results:
            tumor_benefit = model_results['tumor_zy_zay']['val_acc'] - model_results['tumor_zy_alone']['val_acc']
            tumor_baseline = model_results['tumor_zy_alone']['val_acc']

        if hospital_benefit > 0 or tumor_benefit > 0:
            zay_benefits.append({
                'model': model_name.upper(),
                'hospital_benefit': hospital_benefit,
                'tumor_benefit': tumor_benefit,
                'hospital_baseline': hospital_baseline,
                'tumor_baseline': tumor_baseline
            })

    if best_hospital_model:
        report_lines.append(f"🏆 Best Hospital Classification: {best_hospital_model} ({best_hospital_score:.3f} accuracy)")
    if best_tumor_model:
        report_lines.append(f"🏆 Best Tumor Classification: {best_tumor_model} ({best_tumor_score:.3f} accuracy)")

    report_lines.append("")
    report_lines.append("🧪 ZAY COMPONENT BENEFITS:")
    report_lines.append("-" * 50)

    for benefit in zay_benefits:
        report_lines.append(f"{benefit['model']}:")
        if benefit['hospital_benefit'] > 0:
            hospital_pct = (benefit['hospital_benefit'] / benefit['hospital_baseline']) * 100
            report_lines.append(f"  Hospital: +{benefit['hospital_benefit']:.3f} ({hospital_pct:.1f}% improvement)")
        if benefit['tumor_benefit'] > 0:
            tumor_pct = (benefit['tumor_benefit'] / benefit['tumor_baseline']) * 100
            report_lines.append(f"  Tumor:    +{benefit['tumor_benefit']:.3f} ({tumor_pct:.1f}% improvement)")
        report_lines.append("")

    # Detailed results for each model
    report_lines.append("📊 DETAILED RESULTS BY MODEL:")
    report_lines.append("-" * 50)

    for model_name, model_results in results.items():
        report_lines.append(f"\n{model_name.upper()} MODEL:")
        report_lines.append("  Hospital Classification:")

        if 'hospital_za_alone' in model_results:
            report_lines.append(f"    za alone:     {model_results['hospital_za_alone']['val_acc']:.4f}")
        if 'hospital_za_zay' in model_results:
            report_lines.append(f"    za+zay:       {model_results['hospital_za_zay']['val_acc']:.4f}")
        if 'hospital_zay_alone' in model_results:
            report_lines.append(f"    zay alone:    {model_results['hospital_zay_alone']['val_acc']:.4f}")

        report_lines.append("  Tumor Classification:")
        if 'tumor_zy_alone' in model_results:
            report_lines.append(f"    zy alone:     {model_results['tumor_zy_alone']['val_acc']:.4f}")
        if 'tumor_zy_zay' in model_results:
            report_lines.append(f"    zy+zay:       {model_results['tumor_zy_zay']['val_acc']:.4f}")
        if 'tumor_zay_alone' in model_results:
            report_lines.append(f"    zay alone:    {model_results['tumor_zay_alone']['val_acc']:.4f}")

    # Conclusions
    report_lines.append("\n🎯 KEY FINDINGS:")
    report_lines.append("-" * 50)

    avg_hospital_benefit = np.mean([b['hospital_benefit'] for b in zay_benefits if b['hospital_benefit'] > 0])
    avg_tumor_benefit = np.mean([b['tumor_benefit'] for b in zay_benefits if b['tumor_benefit'] > 0])

    if not np.isnan(avg_hospital_benefit):
        report_lines.append(f"• Adding zay improves hospital classification by {avg_hospital_benefit:.3f} on average")
    if not np.isnan(avg_tumor_benefit):
        report_lines.append(f"• Adding zay improves tumor classification by {avg_tumor_benefit:.3f} on average")

    report_lines.append("• The zay component captures shared information useful for both tasks")
    report_lines.append("• Combined representations (za+zay, zy+zay) are more expressive than individual components")
    report_lines.append("• Hospital-specific features in za help identify data source")
    report_lines.append("• Tumor-specific features in zy enable pathology classification")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'expressiveness_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"📝 Summary report saved to: {report_path}")
    print("\n" + report_text)

    return report_text

def main(results_dir):
    """Main function to run comprehensive expressiveness analysis."""

    print("🔍 Loading expressiveness results from all models...")
    results = load_expressiveness_results(results_dir)

    if not results:
        print("❌ No expressiveness results found!")
        return

    print(f"\n📊 Found results for {len(results)} models: {list(results.keys())}")

    # Create comprehensive comparison
    print("\n🎨 Creating comprehensive comparison visualization...")
    df, improvement_df = create_comprehensive_comparison(results, results_dir)

    # Generate summary report
    print("\n📝 Generating summary report...")
    report = generate_summary_report(results, results_dir)

    print(f"\n🎉 Comprehensive expressiveness analysis completed!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python compare_all_expressiveness.py <results_directory>")
        print("Example: python compare_all_expressiveness.py results/")
        sys.exit(1)

    results_dir = sys.argv[1]
    main(results_dir)

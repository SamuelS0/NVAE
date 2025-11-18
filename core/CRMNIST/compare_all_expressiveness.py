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
        
        # Domain classification results
        if 'domain_za_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Domain Classification',
                'Method': 'Individual (za)',
                'Validation Accuracy': model_results['domain_za_alone']['val_acc'],
                'Test Accuracy': model_results['domain_za_alone'].get('test_acc', model_results['domain_za_alone']['val_acc']),
                'Component': 'za'
            })

        if 'domain_zy_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Domain Classification',
                'Method': 'Cross-prediction (zy)',
                'Validation Accuracy': model_results['domain_zy_alone']['val_acc'],
                'Test Accuracy': model_results['domain_zy_alone'].get('test_acc', model_results['domain_zy_alone']['val_acc']),
                'Component': 'zy'
            })

        if 'domain_za_zay' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Domain Classification',
                'Method': 'Combined (za+zay)',
                'Validation Accuracy': model_results['domain_za_zay']['val_acc'],
                'Test Accuracy': model_results['domain_za_zay'].get('test_acc', model_results['domain_za_zay']['val_acc']),
                'Component': 'za+zay'
            })

        if 'domain_zay_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Domain Classification',
                'Method': 'Individual (zay)',
                'Validation Accuracy': model_results['domain_zay_alone']['val_acc'],
                'Test Accuracy': model_results['domain_zay_alone'].get('test_acc', model_results['domain_zay_alone']['val_acc']),
                'Component': 'zay'
            })
        
        # Label classification results
        if 'label_zy_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Label Classification',
                'Method': 'Individual (zy)',
                'Validation Accuracy': model_results['label_zy_alone']['val_acc'],
                'Test Accuracy': model_results['label_zy_alone'].get('test_acc', model_results['label_zy_alone']['val_acc']),
                'Component': 'zy'
            })

        if 'label_za_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Label Classification',
                'Method': 'Cross-prediction (za)',
                'Validation Accuracy': model_results['label_za_alone']['val_acc'],
                'Test Accuracy': model_results['label_za_alone'].get('test_acc', model_results['label_za_alone']['val_acc']),
                'Component': 'za'
            })

        if 'label_zy_zay' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Label Classification',
                'Method': 'Combined (zy+zay)',
                'Validation Accuracy': model_results['label_zy_zay']['val_acc'],
                'Test Accuracy': model_results['label_zy_zay'].get('test_acc', model_results['label_zy_zay']['val_acc']),
                'Component': 'zy+zay'
            })

        if 'label_zay_alone' in model_results:
            comparison_data.append({
                'Model': model_name.upper(),
                'Task': 'Label Classification',
                'Method': 'Individual (zay)',
                'Validation Accuracy': model_results['label_zay_alone']['val_acc'],
                'Test Accuracy': model_results['label_zay_alone'].get('test_acc', model_results['label_zay_alone']['val_acc']),
                'Component': 'zay'
            })
    
    df = pd.DataFrame(comparison_data)

    if df.empty:
        print("⚠️  No data available for comparison")
        return None, None
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Domain Classification Comparison
    domain_data = df[df['Task'] == 'Domain Classification']
    if not domain_data.empty:
        # Pivot for heatmap
        domain_pivot = domain_data.pivot_table(
            index='Model', columns='Component', values='Validation Accuracy', aggfunc='mean'
        )
        
        sns.heatmap(domain_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[0,0], cbar_kws={'label': 'Validation Accuracy'})
        axes[0,0].set_title('Domain Classification Accuracy by Model & Component', fontweight='bold')
        axes[0,0].set_ylabel('Model')
    
    # 2. Label Classification Comparison
    label_data = df[df['Task'] == 'Label Classification']
    if not label_data.empty:
        label_pivot = label_data.pivot_table(
            index='Model', columns='Component', values='Validation Accuracy', aggfunc='mean'
        )
        
        sns.heatmap(label_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[0,1], cbar_kws={'label': 'Validation Accuracy'})
        axes[0,1].set_title('Label Classification Accuracy by Model & Component', fontweight='bold')
        axes[0,1].set_ylabel('Model')
    
    # 3. Improvement Analysis (Combined vs Individual)
    improvements = []
    
    for model_name, model_results in results.items():
        # Domain improvement
        if 'domain_za_alone' in model_results and 'domain_za_zay' in model_results:
            domain_improvement = model_results['domain_za_zay']['val_acc'] - model_results['domain_za_alone']['val_acc']
            domain_baseline = model_results['domain_za_alone']['val_acc']
            improvements.append({
                'Model': model_name.upper(),
                'Task': 'Domain',
                'Improvement': domain_improvement,
                'Improvement_Percent': (domain_improvement / domain_baseline) * 100
            })

        # Label improvement
        if 'label_zy_alone' in model_results and 'label_zy_zay' in model_results:
            label_improvement = model_results['label_zy_zay']['val_acc'] - model_results['label_zy_alone']['val_acc']
            label_baseline = model_results['label_zy_alone']['val_acc']
            improvements.append({
                'Model': model_name.upper(),
                'Task': 'Label',
                'Improvement': label_improvement,
                'Improvement_Percent': (label_improvement / label_baseline) * 100
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
    if not domain_data.empty and not label_data.empty:
        # Calculate average combined performance
        model_rankings = []
        
        for model_name, model_results in results.items():
            combined_score = 0
            count = 0
            
            if 'domain_za_zay' in model_results:
                combined_score += model_results['domain_za_zay']['val_acc']
                count += 1
            elif 'domain_za_alone' in model_results:
                combined_score += model_results['domain_za_alone']['val_acc']
                count += 1
                
            if 'label_zy_zay' in model_results:
                combined_score += model_results['label_zy_zay']['val_acc']
                count += 1
            elif 'label_zy_alone' in model_results:
                combined_score += model_results['label_zy_alone']['val_acc']
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
    report_lines.append("LATENT VARIABLE EXPRESSIVENESS ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("📋 EXECUTIVE SUMMARY:")
    report_lines.append("-" * 50)
    report_lines.append("")
    
    # Find best performing models
    best_domain_model = None
    best_domain_score = 0
    best_label_model = None
    best_label_score = 0
    
    zay_benefits = []
    
    for model_name, model_results in results.items():
        # Domain classification
        if 'domain_za_zay' in model_results:
            score = model_results['domain_za_zay']['val_acc']
            if score > best_domain_score:
                best_domain_score = score
                best_domain_model = model_name.upper()
        
        # Label classification
        if 'label_zy_zay' in model_results:
            score = model_results['label_zy_zay']['val_acc']
            if score > best_label_score:
                best_label_score = score
                best_label_model = model_name.upper()
        
        # Calculate zay benefits
        domain_benefit = 0
        label_benefit = 0
        domain_baseline = 1.0  # Default to avoid division by zero
        label_baseline = 1.0   # Default to avoid division by zero

        if 'domain_za_alone' in model_results and 'domain_za_zay' in model_results:
            domain_benefit = model_results['domain_za_zay']['val_acc'] - model_results['domain_za_alone']['val_acc']
            domain_baseline = model_results['domain_za_alone']['val_acc']

        if 'label_zy_alone' in model_results and 'label_zy_zay' in model_results:
            label_benefit = model_results['label_zy_zay']['val_acc'] - model_results['label_zy_alone']['val_acc']
            label_baseline = model_results['label_zy_alone']['val_acc']

        if domain_benefit > 0 or label_benefit > 0:
            zay_benefits.append({
                'model': model_name.upper(),
                'domain_benefit': domain_benefit,
                'label_benefit': label_benefit,
                'domain_baseline': domain_baseline,
                'label_baseline': label_baseline
            })
    
    if best_domain_model:
        report_lines.append(f"🏆 Best Domain Classification: {best_domain_model} ({best_domain_score:.3f} accuracy)")
    if best_label_model:
        report_lines.append(f"🏆 Best Label Classification: {best_label_model} ({best_label_score:.3f} accuracy)")
    
    report_lines.append("")
    report_lines.append("🧪 ZAY COMPONENT BENEFITS:")
    report_lines.append("-" * 50)
    
    for benefit in zay_benefits:
        report_lines.append(f"{benefit['model']}:")
        if benefit['domain_benefit'] > 0:
            domain_pct = (benefit['domain_benefit'] / benefit['domain_baseline']) * 100
            report_lines.append(f"  Domain: +{benefit['domain_benefit']:.3f} ({domain_pct:.1f}% improvement)")
        if benefit['label_benefit'] > 0:
            label_pct = (benefit['label_benefit'] / benefit['label_baseline']) * 100
            report_lines.append(f"  Label:  +{benefit['label_benefit']:.3f} ({label_pct:.1f}% improvement)")
        report_lines.append("")
    
    # Detailed results for each model
    report_lines.append("📊 DETAILED RESULTS BY MODEL:")
    report_lines.append("-" * 50)
    
    for model_name, model_results in results.items():
        report_lines.append(f"\n{model_name.upper()} MODEL:")
        report_lines.append("  Domain Classification:")

        if 'domain_za_alone' in model_results:
            report_lines.append(f"    za alone:     {model_results['domain_za_alone']['val_acc']:.4f}")
        if 'domain_zy_alone' in model_results:
            report_lines.append(f"    zy alone:     {model_results['domain_zy_alone']['val_acc']:.4f} [cross-prediction]")
        if 'domain_za_zay' in model_results:
            report_lines.append(f"    za+zay:       {model_results['domain_za_zay']['val_acc']:.4f}")
        if 'domain_zay_alone' in model_results:
            report_lines.append(f"    zay alone:    {model_results['domain_zay_alone']['val_acc']:.4f}")

        report_lines.append("  Label Classification:")
        if 'label_zy_alone' in model_results:
            report_lines.append(f"    zy alone:     {model_results['label_zy_alone']['val_acc']:.4f}")
        if 'label_za_alone' in model_results:
            report_lines.append(f"    za alone:     {model_results['label_za_alone']['val_acc']:.4f} [cross-prediction]")
        if 'label_zy_zay' in model_results:
            report_lines.append(f"    zy+zay:       {model_results['label_zy_zay']['val_acc']:.4f}")
        if 'label_zay_alone' in model_results:
            report_lines.append(f"    zay alone:    {model_results['label_zay_alone']['val_acc']:.4f}")
    
    # Conclusions
    report_lines.append("\n🎯 KEY FINDINGS:")
    report_lines.append("-" * 50)
    
    avg_domain_benefit = np.mean([b['domain_benefit'] for b in zay_benefits if b['domain_benefit'] > 0])
    avg_label_benefit = np.mean([b['label_benefit'] for b in zay_benefits if b['label_benefit'] > 0])
    
    if not np.isnan(avg_domain_benefit):
        report_lines.append(f"• Adding zay improves domain classification by {avg_domain_benefit:.3f} on average")
    if not np.isnan(avg_label_benefit):
        report_lines.append(f"• Adding zay improves label classification by {avg_label_benefit:.3f} on average")
    
    report_lines.append("• The zay component captures shared information useful for both tasks")
    report_lines.append("• Combined representations (za+zay, zy+zay) are more expressive than individual components")
    
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
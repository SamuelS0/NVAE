"""
Results aggregation and analysis for CRMNIST grid search.

Provides functionality to:
- Collect results from all experiments
- Generate comparison tables
- Create visualization plots
- Identify best configurations
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class ResultsAggregator:
    """
    Aggregates and analyzes results from grid search experiments.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the results aggregator.

        Args:
            output_dir: Directory containing grid search results
        """
        self.output_dir = output_dir
        self.experiments_dir = os.path.join(output_dir, 'experiments')
        self.summary_dir = os.path.join(output_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.results = []
        self.df = None

    def collect_results(self) -> pd.DataFrame:
        """
        Collect results from all completed experiments.

        Returns:
            DataFrame containing all results
        """
        self.results = []

        if not os.path.exists(self.experiments_dir):
            print(f"No experiments directory found at {self.experiments_dir}")
            return pd.DataFrame()

        for exp_name in os.listdir(self.experiments_dir):
            exp_dir = os.path.join(self.experiments_dir, exp_name)
            metrics_path = os.path.join(exp_dir, 'metrics.json')

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)

                    # Flatten the result for DataFrame
                    flat_result = self._flatten_result(metrics)
                    self.results.append(flat_result)
                except Exception as e:
                    print(f"Error loading {exp_name}: {e}")

        if not self.results:
            print("No completed experiments found")
            return pd.DataFrame()

        self.df = pd.DataFrame(self.results)
        print(f"Collected {len(self.results)} experiment results")

        return self.df

    def _flatten_result(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metrics dictionary for DataFrame."""
        flat = {
            'name': metrics.get('experiment_name', ''),
            'model_type': metrics.get('model_type', ''),
            'training_time': metrics.get('training_time', 0),
            'timestamp': metrics.get('timestamp', ''),
        }

        # Add preset names
        for key, value in metrics.get('preset_names', {}).items():
            flat[f'preset_{key}'] = value

        # Add training metrics
        training_metrics = metrics.get('training_metrics', {})
        if isinstance(training_metrics, dict):
            flat['best_val_loss'] = training_metrics.get('best_val_loss', None)
            flat['best_val_accuracy'] = training_metrics.get('best_val_accuracy', None)
            flat['epochs_trained'] = training_metrics.get('epochs_trained', None)
            flat['best_epoch'] = training_metrics.get('best_model_epoch', None)

        # Add evaluation metrics
        eval_metrics = metrics.get('eval_metrics', {})
        if isinstance(eval_metrics, dict):
            flat['test_accuracy'] = eval_metrics.get('test_accuracy', None)
            flat['id_accuracy'] = eval_metrics.get('id_accuracy', None)
            flat['ood_accuracy'] = eval_metrics.get('ood_accuracy', None)
            flat['gen_gap'] = eval_metrics.get('gen_gap', None)
            flat['total_samples'] = eval_metrics.get('total_samples', None)

        # Calculate generalization gap if not provided but both accuracies available
        if flat.get('gen_gap') is None and flat.get('test_accuracy') and flat.get('ood_accuracy'):
            flat['gen_gap'] = flat['test_accuracy'] - flat['ood_accuracy']

        return flat

    def get_best_configs(
        self,
        metric: str = 'test_accuracy',
        n: int = 5,
        model_type: str = None,
    ) -> pd.DataFrame:
        """
        Get the best configurations by a specified metric.

        Args:
            metric: Metric to rank by ('test_accuracy', 'ood_accuracy', etc.)
            n: Number of top configurations to return
            model_type: Filter by model type (optional)

        Returns:
            DataFrame with top n configurations
        """
        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            return pd.DataFrame()

        df = self.df.copy()

        # Filter by model type if specified
        if model_type:
            df = df[df['model_type'] == model_type]

        # Filter out rows with missing metric
        df = df[df[metric].notna()]

        # Sort and get top n
        ascending = metric in ['gen_gap', 'best_val_loss', 'training_time']
        df_sorted = df.sort_values(by=metric, ascending=ascending)

        return df_sorted.head(n)

    def generate_comparison_table(
        self,
        group_by: str = 'model_type',
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a comparison table grouped by a specified column.

        Args:
            group_by: Column to group by
            metrics: List of metrics to include

        Returns:
            DataFrame with aggregated statistics
        """
        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            return pd.DataFrame()

        if metrics is None:
            metrics = ['test_accuracy', 'training_time']

        # Calculate statistics
        agg_funcs = {metric: ['mean', 'std', 'min', 'max'] for metric in metrics if metric in self.df.columns}
        agg_funcs['name'] = 'count'  # Count experiments

        comparison = self.df.groupby(group_by).agg(agg_funcs)

        return comparison

    def save_summary(self):
        """Save summary results to files."""
        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            print("No results to save")
            return

        # Save full results as CSV
        csv_path = os.path.join(self.summary_dir, 'all_results.csv')
        self.df.to_csv(csv_path, index=False)
        print(f"Saved full results to {csv_path}")

        # Save best configs
        best_configs = {}
        for model_type in self.df['model_type'].unique():
            best = self.get_best_configs(metric='test_accuracy', n=3, model_type=model_type)
            if not best.empty:
                best_configs[model_type] = best.to_dict('records')

        best_path = os.path.join(self.summary_dir, 'best_configs.json')
        with open(best_path, 'w') as f:
            json.dump(best_configs, f, indent=2, default=str)
        print(f"Saved best configs to {best_path}")

        # Save comparison table as markdown
        comparison = self.generate_comparison_table()
        if not comparison.empty:
            md_path = os.path.join(self.summary_dir, 'comparison_table.md')
            with open(md_path, 'w') as f:
                f.write("# CRMNIST Grid Search Results\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write(f"Total experiments: {len(self.df)}\n\n")
                f.write("## Summary by Model Type\n\n")
                f.write(comparison.to_markdown())
                f.write("\n\n## Best Configurations\n\n")
                for model_type, configs in best_configs.items():
                    f.write(f"### {model_type}\n\n")
                    for i, config in enumerate(configs, 1):
                        f.write(f"{i}. **{config['name']}**\n")
                        f.write(f"   - Test Accuracy: {config.get('test_accuracy', 'N/A'):.4f}\n")
                        f.write(f"   - Training Time: {config.get('training_time', 'N/A'):.1f}s\n")
                        f.write("\n")
            print(f"Saved comparison table to {md_path}")

    def generate_heatmaps(self, model_type: str = 'nvae'):
        """
        Generate heatmaps showing performance across hyperparameter combinations.

        Args:
            model_type: Model type to visualize ('nvae' or 'diva')
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for heatmaps")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            print("No results to visualize")
            return

        # Filter by model type
        df_model = self.df[self.df['model_type'] == model_type].copy()

        if df_model.empty:
            print(f"No results for model type: {model_type}")
            return

        # Check which preset columns exist
        preset_cols = [col for col in df_model.columns if col.startswith('preset_')]

        if len(preset_cols) < 2:
            print(f"Not enough preset columns for heatmap: {preset_cols}")
            return

        # Create heatmaps for different preset combinations
        for metric in ['test_accuracy', 'training_time']:
            if metric not in df_model.columns:
                continue

            # Create pivot table for each pair of presets
            for i, row_preset in enumerate(preset_cols):
                for col_preset in preset_cols[i+1:]:
                    try:
                        pivot = df_model.pivot_table(
                            values=metric,
                            index=row_preset,
                            columns=col_preset,
                            aggfunc='mean'
                        )

                        if pivot.empty:
                            continue

                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax)
                        ax.set_title(f'{model_type.upper()}: {metric} by {row_preset} × {col_preset}')

                        # Save
                        filename = f'heatmap_{model_type}_{metric}_{row_preset}_{col_preset}.png'
                        filepath = os.path.join(self.summary_dir, filename)
                        plt.savefig(filepath, bbox_inches='tight', dpi=150)
                        plt.close()
                        print(f"Saved heatmap to {filepath}")

                    except Exception as e:
                        print(f"Error creating heatmap: {e}")

    def generate_model_comparison_plot(self):
        """Generate bar plot comparing all model types."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plots")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            print("No results to visualize")
            return

        # Get best result for each model type
        best_per_model = []
        for model_type in self.df['model_type'].unique():
            best = self.get_best_configs(metric='test_accuracy', n=1, model_type=model_type)
            if not best.empty:
                best_per_model.append({
                    'model_type': model_type,
                    'test_accuracy': best['test_accuracy'].values[0],
                    'name': best['name'].values[0],
                })

        if not best_per_model:
            print("No best results found")
            return

        df_best = pd.DataFrame(best_per_model)
        df_best = df_best.sort_values('test_accuracy', ascending=True)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_best)))
        bars = ax.barh(df_best['model_type'], df_best['test_accuracy'], color=colors)

        # Add value labels
        for bar, acc in zip(bars, df_best['test_accuracy']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{acc:.4f}', va='center', fontsize=10)

        ax.set_xlabel('Test Accuracy')
        ax.set_title('Best Test Accuracy by Model Type')
        ax.set_xlim(0, 1.1)

        filepath = os.path.join(self.summary_dir, 'model_comparison.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved model comparison to {filepath}")

    def print_summary(self):
        """Print a text summary of results."""
        if self.df is None or self.df.empty:
            self.collect_results()

        if self.df is None or self.df.empty:
            print("No results to summarize")
            return

        print("\n" + "="*60)
        print("CRMNIST GRID SEARCH RESULTS SUMMARY")
        print("="*60)

        print(f"\nTotal experiments: {len(self.df)}")
        print(f"Model types: {list(self.df['model_type'].unique())}")

        print("\n" + "-"*60)
        print("BEST CONFIGURATION PER MODEL TYPE")
        print("-"*60)

        for model_type in sorted(self.df['model_type'].unique()):
            best = self.get_best_configs(metric='test_accuracy', n=1, model_type=model_type)
            if not best.empty:
                row = best.iloc[0]
                print(f"\n{model_type.upper()}:")
                print(f"  Config: {row['name']}")
                print(f"  Test Accuracy: {row['test_accuracy']:.4f}" if pd.notna(row['test_accuracy']) else "  Test Accuracy: N/A")
                if pd.notna(row.get('training_time')):
                    print(f"  Training Time: {row['training_time']:.1f}s")

        print("\n" + "-"*60)
        print("OVERALL STATISTICS")
        print("-"*60)

        for metric in ['test_accuracy', 'training_time']:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                if len(values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Mean: {values.mean():.4f}")
                    print(f"  Std:  {values.std():.4f}")
                    print(f"  Min:  {values.min():.4f}")
                    print(f"  Max:  {values.max():.4f}")

        print("\n" + "="*60)


def analyze_results(output_dir: str, generate_plots: bool = True):
    """
    Analyze grid search results and generate reports.

    Args:
        output_dir: Directory containing grid search results
        generate_plots: If True, generate visualization plots
    """
    aggregator = ResultsAggregator(output_dir)
    aggregator.collect_results()
    aggregator.print_summary()
    aggregator.save_summary()

    if generate_plots:
        for model_type in ['nvae', 'diva']:
            aggregator.generate_heatmaps(model_type)
        aggregator.generate_model_comparison_plot()

    return aggregator

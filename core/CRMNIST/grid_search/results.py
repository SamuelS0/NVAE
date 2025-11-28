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

        # Add Information-Theoretic metrics (if available)
        it_metrics = metrics.get('it_metrics', {})
        if isinstance(it_metrics, dict) and 'error' not in it_metrics:
            flat['it_partition_quality'] = it_metrics.get('partition_quality', None)
            flat['it_zy_specificity'] = it_metrics.get('z_y_specificity', None)
            flat['it_zd_specificity'] = it_metrics.get('z_d_specificity', None)
            flat['it_I_zy_Y_given_D'] = it_metrics.get('I_zy_Y_given_D', None)
            flat['it_I_zy_D_given_Y'] = it_metrics.get('I_zy_D_given_Y', None)
            flat['it_I_zd_D_given_Y'] = it_metrics.get('I_zd_D_given_Y', None)
            flat['it_I_zd_Y_given_D'] = it_metrics.get('I_zd_Y_given_D', None)
            flat['it_I_zdy_Y_D'] = it_metrics.get('I_zdy_Y_D', None)
            flat['it_I_zdy_joint'] = it_metrics.get('I_zdy_joint', None)

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
                # Try to use to_markdown, fall back to to_string if tabulate not available
                try:
                    f.write(comparison.to_markdown())
                except ImportError:
                    f.write("```\n")
                    f.write(comparison.to_string())
                    f.write("\n```")
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
                if pd.notna(row.get('ood_accuracy')):
                    print(f"  OOD Accuracy: {row['ood_accuracy']:.4f}")
                if pd.notna(row.get('gen_gap')):
                    print(f"  Gen Gap: {row['gen_gap']:.4f}")
                if pd.notna(row.get('training_time')):
                    print(f"  Training Time: {row['training_time']:.1f}s")
                # IT metrics (only for supported models)
                if pd.notna(row.get('it_partition_quality')):
                    print(f"  IT Partition Quality: {row['it_partition_quality']:.4f}")
                if pd.notna(row.get('it_zy_specificity')):
                    print(f"  IT z_y Specificity: {row['it_zy_specificity']:.4f}")
                if pd.notna(row.get('it_zd_specificity')):
                    print(f"  IT z_d Specificity: {row['it_zd_specificity']:.4f}")

        print("\n" + "-"*60)
        print("OVERALL STATISTICS")
        print("-"*60)

        for metric in ['test_accuracy', 'ood_accuracy', 'gen_gap', 'training_time', 'it_partition_quality']:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                if len(values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Mean: {values.mean():.4f}")
                    print(f"  Std:  {values.std():.4f}")
                    print(f"  Min:  {values.min():.4f}")
                    print(f"  Max:  {values.max():.4f}")

        print("\n" + "="*60)

    # =========================================================================
    # Information-Theoretic (IT) Visualization Methods
    # =========================================================================

    def generate_it_model_comparison(self):
        """
        Generate bar charts comparing IT metrics across model types.
        Creates a multi-panel figure showing partition quality, z_y specificity,
        and z_d specificity for each model type.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for IT plots")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter to models with IT metrics
        df_it = self.df[self.df['it_partition_quality'].notna()].copy()

        if df_it.empty:
            print("No IT metrics available for visualization")
            return

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        it_metrics = [
            ('it_partition_quality', 'Partition Quality', 'viridis'),
            ('it_zy_specificity', 'z_y Specificity (Class)', 'Blues'),
            ('it_zd_specificity', 'z_d Specificity (Domain)', 'Oranges'),
        ]

        for ax, (metric, title, cmap) in zip(axes, it_metrics):
            # Group by model type and compute mean
            model_means = df_it.groupby('model_type')[metric].agg(['mean', 'std']).reset_index()
            model_means = model_means.sort_values('mean', ascending=True)

            colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.8, len(model_means)))
            bars = ax.barh(model_means['model_type'], model_means['mean'],
                          xerr=model_means['std'], color=colors, capsize=3)

            # Add value labels
            for bar, mean_val in zip(bars, model_means['mean']):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{mean_val:.3f}', va='center', fontsize=9)

            ax.set_xlabel(title)
            ax.set_title(f'{title} by Model Type')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'it_model_comparison.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved IT model comparison to {filepath}")

    def generate_it_heatmaps(self, model_type: str = 'nvae'):
        """
        Generate heatmaps showing IT metrics across hyperparameter combinations.

        Args:
            model_type: Model type to visualize ('nvae', 'diva', or 'dann_augmented')
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for IT heatmaps")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter by model type and IT metrics availability
        df_model = self.df[(self.df['model_type'] == model_type) &
                          (self.df['it_partition_quality'].notna())].copy()

        if df_model.empty:
            print(f"No IT results for model type: {model_type}")
            return

        # Check which preset columns exist
        preset_cols = [col for col in df_model.columns if col.startswith('preset_')]

        if len(preset_cols) < 2:
            print(f"Not enough preset columns for IT heatmap: {preset_cols}")
            return

        # IT metrics to visualize
        it_metrics = [
            ('it_partition_quality', 'Partition Quality'),
            ('it_zy_specificity', 'z_y Specificity'),
            ('it_zd_specificity', 'z_d Specificity'),
        ]

        for metric, metric_name in it_metrics:
            if metric not in df_model.columns or df_model[metric].isna().all():
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

                        if pivot.empty or pivot.isna().all().all():
                            continue

                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))

                        # Choose colormap based on metric
                        if 'specificity' in metric:
                            cmap = 'RdYlGn'  # Red-Yellow-Green for specificity
                            center = 0
                        else:
                            cmap = 'viridis'
                            center = None

                        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                                   ax=ax, center=center)
                        ax.set_title(f'{model_type.upper()}: {metric_name}\nby {row_preset} × {col_preset}')

                        # Save
                        filename = f'it_heatmap_{model_type}_{metric}_{row_preset}_{col_preset}.png'
                        filepath = os.path.join(self.summary_dir, filename)
                        plt.savefig(filepath, bbox_inches='tight', dpi=150)
                        plt.close()
                        print(f"Saved IT heatmap to {filepath}")

                    except Exception as e:
                        print(f"Error creating IT heatmap: {e}")

    def generate_it_correlation_plots(self):
        """
        Generate scatter plots showing correlation between IT metrics and
        accuracy/generalization metrics.
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            print("matplotlib and scipy required for correlation plots")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter to models with IT metrics
        df_it = self.df[self.df['it_partition_quality'].notna()].copy()

        if df_it.empty:
            print("No IT metrics available for correlation plots")
            return

        # Correlation pairs to plot
        correlations = [
            ('it_partition_quality', 'ood_accuracy', 'Partition Quality vs OOD Accuracy'),
            ('it_partition_quality', 'gen_gap', 'Partition Quality vs Generalization Gap'),
            ('it_zy_specificity', 'test_accuracy', 'z_y Specificity vs Test Accuracy'),
            ('it_zd_specificity', 'ood_accuracy', 'z_d Specificity vs OOD Accuracy'),
        ]

        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, (x_metric, y_metric, title) in zip(axes, correlations):
            # Filter valid data
            valid = df_it[[x_metric, y_metric, 'model_type']].dropna()

            if len(valid) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_title(title)
                continue

            # Scatter plot with model type colors
            model_types = valid['model_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
            color_map = dict(zip(model_types, colors))

            for model in model_types:
                model_data = valid[valid['model_type'] == model]
                ax.scatter(model_data[x_metric], model_data[y_metric],
                          c=[color_map[model]], label=model, alpha=0.7, s=50)

            # Add regression line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid[x_metric], valid[y_metric]
                )
                x_line = np.linspace(valid[x_metric].min(), valid[x_metric].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'k--', alpha=0.5,
                       label=f'r={r_value:.3f}, p={p_value:.3f}')
            except Exception:
                pass

            ax.set_xlabel(x_metric.replace('_', ' ').title())
            ax.set_ylabel(y_metric.replace('_', ' ').title())
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)

        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'it_correlation_plots.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved IT correlation plots to {filepath}")

    def generate_it_specificity_scatter(self):
        """
        Generate scatter plot of z_y specificity vs z_d specificity,
        colored by model type, with point size indicating partition quality.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for specificity scatter")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter to models with IT metrics
        df_it = self.df[self.df['it_partition_quality'].notna()].copy()

        if df_it.empty:
            print("No IT metrics available for specificity scatter")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Color by model type
        model_types = df_it['model_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
        color_map = dict(zip(model_types, colors))

        for model in model_types:
            model_data = df_it[df_it['model_type'] == model]

            # Point size based on partition quality (scaled)
            sizes = 50 + 200 * model_data['it_partition_quality']

            ax.scatter(model_data['it_zy_specificity'],
                      model_data['it_zd_specificity'],
                      c=[color_map[model]],
                      s=sizes,
                      label=model,
                      alpha=0.6,
                      edgecolors='white',
                      linewidth=0.5)

        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[1]*0.7, ylim[1]*0.8, 'Good z_y\nGood z_d', ha='center', fontsize=9, alpha=0.7)
        ax.text(xlim[0]*0.3 if xlim[0] < 0 else 0.1, ylim[1]*0.8, 'Poor z_y\nGood z_d', ha='center', fontsize=9, alpha=0.7)
        ax.text(xlim[1]*0.7, ylim[0]*0.8 if ylim[0] < 0 else -0.5, 'Good z_y\nPoor z_d', ha='center', fontsize=9, alpha=0.7)
        ax.text(xlim[0]*0.3 if xlim[0] < 0 else 0.1, ylim[0]*0.8 if ylim[0] < 0 else -0.5, 'Poor z_y\nPoor z_d', ha='center', fontsize=9, alpha=0.7)

        ax.set_xlabel('z_y Specificity: I(z_y;Y|D) - I(z_y;D|Y)')
        ax.set_ylabel('z_d Specificity: I(z_d;D|Y) - I(z_d;Y|D)')
        ax.set_title('Latent Space Specificity\n(point size = partition quality)')
        ax.legend(loc='best')

        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'it_specificity_scatter.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved IT specificity scatter to {filepath}")

    def generate_it_distribution_plots(self):
        """
        Generate box plots showing distribution of IT metrics per model type.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for distribution plots")
            return

        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter to models with IT metrics
        df_it = self.df[self.df['it_partition_quality'].notna()].copy()

        if df_it.empty:
            print("No IT metrics available for distribution plots")
            return

        # IT metrics to plot
        it_metrics = [
            'it_partition_quality',
            'it_zy_specificity',
            'it_zd_specificity',
            'it_I_zy_Y_given_D',
            'it_I_zd_D_given_Y',
            'it_I_zdy_Y_D',
        ]

        # Filter to existing metrics
        it_metrics = [m for m in it_metrics if m in df_it.columns and df_it[m].notna().any()]

        if not it_metrics:
            print("No IT metrics to plot")
            return

        # Create subplots
        n_metrics = len(it_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, metric in enumerate(it_metrics):
            ax = axes[i]

            # Box plot
            sns.boxplot(data=df_it, x='model_type', y=metric, ax=ax, palette='Set2')

            # Add individual points
            sns.stripplot(data=df_it, x='model_type', y=metric, ax=ax,
                         color='black', alpha=0.3, size=3)

            ax.set_xlabel('Model Type')
            ax.set_ylabel(metric.replace('_', ' ').replace('it ', ''))
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)

            # Add zero line for specificity metrics
            if 'specificity' in metric or 'I_zdy' in metric:
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'it_distribution_plots.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved IT distribution plots to {filepath}")

    def generate_it_summary_table(self):
        """
        Generate and save a summary table of IT metrics by model type.
        """
        if self.df is None or self.df.empty:
            self.collect_results()

        # Filter to models with IT metrics
        df_it = self.df[self.df['it_partition_quality'].notna()].copy()

        if df_it.empty:
            print("No IT metrics available for summary table")
            return

        # Aggregate IT metrics by model type
        it_cols = [c for c in df_it.columns if c.startswith('it_')]

        summary = df_it.groupby('model_type')[it_cols].agg(['mean', 'std', 'min', 'max'])

        # Save to CSV
        csv_path = os.path.join(self.summary_dir, 'it_summary_by_model.csv')
        summary.to_csv(csv_path)
        print(f"Saved IT summary table to {csv_path}")

        # Also save a simpler version
        simple_summary = df_it.groupby('model_type')[it_cols].mean()
        simple_path = os.path.join(self.summary_dir, 'it_means_by_model.csv')
        simple_summary.to_csv(simple_path)
        print(f"Saved IT means table to {simple_path}")

        return summary

    def generate_all_it_plots(self):
        """Generate all IT-related visualizations."""
        print("\n" + "-"*60)
        print("Generating Information-Theoretic Visualizations...")
        print("-"*60)

        self.generate_it_model_comparison()
        self.generate_it_specificity_scatter()
        self.generate_it_correlation_plots()
        self.generate_it_distribution_plots()
        self.generate_it_summary_table()

        # Generate IT heatmaps for each supported model type
        for model_type in ['nvae', 'diva', 'dann_augmented']:
            self.generate_it_heatmaps(model_type)

        print("-"*60)
        print("IT visualization complete!")
        print("-"*60 + "\n")


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
        # Standard accuracy plots
        for model_type in ['nvae', 'diva', 'dann_augmented']:
            aggregator.generate_heatmaps(model_type)
        aggregator.generate_model_comparison_plot()

        # IT analysis plots
        aggregator.generate_all_it_plots()

    return aggregator

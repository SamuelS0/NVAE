"""
Grid search runner for CRMNIST experiments.

Handles execution of individual experiments and manages the overall grid search process.
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import torch.optim as optim

from .config import get_all_configs, get_quick_configs, FIXED_PARAMS


class GridSearchRunner:
    """
    Runs grid search experiments for CRMNIST models.
    """

    def __init__(
        self,
        output_dir: str,
        device: str = None,
        resume: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the grid search runner.

        Args:
            output_dir: Directory to save results
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            resume: If True, skip already completed experiments
            verbose: If True, print detailed progress
        """
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume = resume
        self.verbose = verbose

        # Create output directories
        self.experiments_dir = os.path.join(output_dir, 'experiments')
        self.logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Track completed experiments
        self.completed_experiments = set()
        if resume:
            self._load_completed_experiments()

    def _load_completed_experiments(self):
        """Load list of completed experiments from disk."""
        if not os.path.exists(self.experiments_dir):
            return
        for exp_name in os.listdir(self.experiments_dir):
            metrics_path = os.path.join(self.experiments_dir, exp_name, 'metrics.json')
            if os.path.exists(metrics_path):
                self.completed_experiments.add(exp_name)
        if self.verbose:
            print(f"Found {len(self.completed_experiments)} completed experiments")

    def _create_args(self, config: Dict[str, Any]) -> argparse.Namespace:
        """Create an args namespace from configuration dictionary."""
        params = config['params']
        args = argparse.Namespace()

        # Set all parameters from config
        for key, value in params.items():
            setattr(args, key, value)

        # Set additional required attributes
        args.device = self.device
        args.out = os.path.join(self.experiments_dir, config['name'])
        args.dataset = 'crmnist'
        args.setting = 'standard'

        # Ensure output directory exists
        os.makedirs(args.out, exist_ok=True)

        return args

    def _load_data(self, args: argparse.Namespace):
        """Load CRMNIST dataset with proper train/val/ID-test/OOD-test splits."""
        from core.CRMNIST.data_generation import generate_crmnist_dataset
        from core.data_utils import create_ood_split

        # Load spec_data from config file
        config_path = 'conf/crmnist.json'
        with open(config_path, 'r') as f:
            spec_data = json.load(f)

        # Prepare domain data
        domain_data = {int(k): v for k, v in spec_data['domain_data'].items()}
        spec_data['domain_data'] = domain_data

        # Override rotation values based on rotation_step
        rotation_step = getattr(args, 'rotation_step', 15)
        for domain_idx in domain_data:
            domain_data[domain_idx]['rotation'] = domain_idx * rotation_step

        # OOD domain (always domain 5)
        ood_domain = getattr(args, 'ood_domain_idx', 5)

        if self.verbose:
            print(f"Loading datasets (OOD domain: {ood_domain})...")

        # Generate training dataset (exclude OOD domain)
        train_dataset = generate_crmnist_dataset(
            spec_data, train=True,
            transform_intensity=getattr(args, 'intensity', 1.5),
            transform_decay=getattr(args, 'intensity_decay', 1.0),
            use_cache=True,
            exclude_domains=[ood_domain],
            base_split='train',
            base_split_ratio=0.8,
            base_split_seed=42
        )

        # Generate validation dataset (different base images, exclude OOD domain)
        val_dataset = generate_crmnist_dataset(
            spec_data, train=True,
            transform_intensity=getattr(args, 'intensity', 1.5),
            transform_decay=getattr(args, 'intensity_decay', 1.0),
            use_cache=True,
            exclude_domains=[ood_domain],
            base_split='val',
            base_split_ratio=0.8,
            base_split_seed=42
        )

        # Generate full test dataset then split into ID and OOD
        full_test_dataset = generate_crmnist_dataset(
            spec_data, train=False,
            transform_intensity=getattr(args, 'intensity', 1.5),
            transform_decay=getattr(args, 'intensity_decay', 1.0),
            use_cache=True
        )

        id_test_dataset, ood_test_dataset = create_ood_split(
            full_test_dataset, ood_domain, dataset_type='crmnist'
        )

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        id_test_loader = torch.utils.data.DataLoader(
            id_test_dataset, batch_size=args.batch_size, shuffle=False
        )
        ood_test_loader = torch.utils.data.DataLoader(
            ood_test_dataset, batch_size=args.batch_size, shuffle=False
        )

        if self.verbose:
            print(f"  Train: {len(train_dataset)} samples")
            print(f"  Val: {len(val_dataset)} samples")
            print(f"  ID Test: {len(id_test_dataset)} samples")
            print(f"  OOD Test: {len(ood_test_dataset)} samples")

        return train_loader, val_loader, id_test_loader, ood_test_loader, spec_data

    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            config: Configuration dictionary with 'name', 'model_type', 'params'

        Returns:
            Dictionary containing metrics and results
        """
        exp_name = config['name']
        model_type = config['model_type']

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"Model type: {model_type}")
            print(f"{'='*60}")

        # Check if already completed
        if exp_name in self.completed_experiments:
            if self.verbose:
                print(f"Experiment already completed, skipping...")
            return self._load_experiment_results(exp_name)

        # Create args and load data
        args = self._create_args(config)
        train_loader, val_loader, id_test_loader, ood_test_loader, spec_data = self._load_data(args)

        # Save config
        config_path = os.path.join(args.out, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Start timing
        start_time = time.time()

        # Train model based on type
        try:
            if model_type == 'nvae':
                model, training_metrics = self._train_nvae(args, spec_data, train_loader, val_loader)
            elif model_type == 'diva':
                model, training_metrics = self._train_diva(args, spec_data, train_loader, val_loader)
            elif model_type == 'dann':
                model, training_metrics = self._train_dann(args, spec_data, train_loader, val_loader)
            elif model_type == 'dann_augmented':
                model, training_metrics = self._train_dann_augmented(args, spec_data, train_loader, val_loader)
            elif model_type == 'irm':
                model, training_metrics = self._train_irm(args, spec_data, train_loader, val_loader)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            training_time = time.time() - start_time

            # Save best model checkpoint
            if training_metrics.get('best_model_state') is not None:
                model_path = os.path.join(args.out, 'best_model.pt')
                torch.save(training_metrics['best_model_state'], model_path)
                if self.verbose:
                    print(f"Saved model checkpoint to {model_path}")

            # Evaluate model on both ID and OOD test sets
            eval_metrics = self._evaluate_model(model, id_test_loader, ood_test_loader, model_type, args)

            # Prepare training metrics for JSON (exclude non-serializable state dict)
            save_training_metrics = {k: v for k, v in training_metrics.items() if k != 'best_model_state'}

            # Compile results
            results = {
                'experiment_name': exp_name,
                'model_type': model_type,
                'preset_names': config['preset_names'],
                'training_time': training_time,
                'training_metrics': save_training_metrics,
                'eval_metrics': eval_metrics,
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
            }

            # Save metrics
            metrics_path = os.path.join(args.out, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Mark as completed
            self.completed_experiments.add(exp_name)

            if self.verbose:
                print(f"\nExperiment completed in {training_time:.1f}s")
                print(f"ID Test accuracy: {eval_metrics.get('test_accuracy', 'N/A'):.4f}")
                print(f"OOD accuracy: {eval_metrics.get('ood_accuracy', 'N/A'):.4f}")
                print(f"Generalization gap: {eval_metrics.get('gen_gap', 'N/A'):.4f}")

            return results

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            if self.verbose:
                print(f"\nExperiment FAILED: {error_msg}")
                print(error_traceback)

            # Save error info
            error_results = {
                'experiment_name': exp_name,
                'model_type': model_type,
                'error': error_msg,
                'traceback': error_traceback,
                'timestamp': datetime.now().isoformat(),
            }
            error_path = os.path.join(args.out, 'error.json')
            with open(error_path, 'w') as f:
                json.dump(error_results, f, indent=2)

            return error_results

    def _train_nvae(self, args, spec_data, train_loader, val_loader):
        """Train NVAE model."""
        from core.comparison.train import train_nvae
        return train_nvae(args, spec_data, train_loader, val_loader, 'crmnist')

    def _train_diva(self, args, spec_data, train_loader, val_loader):
        """Train DIVA model."""
        from core.comparison.train import train_diva
        return train_diva(args, spec_data, train_loader, val_loader, 'crmnist')

    def _train_dann(self, args, spec_data, train_loader, val_loader):
        """Train DANN model."""
        from core.comparison.train import train_dann
        return train_dann(args, spec_data, train_loader, val_loader, 'crmnist')

    def _train_dann_augmented(self, args, spec_data, train_loader, val_loader):
        """Train AugmentedDANN model."""
        from core.comparison.train import train_dann_augmented
        return train_dann_augmented(args, spec_data, train_loader, val_loader, 'crmnist')

    def _train_irm(self, args, spec_data, train_loader, val_loader):
        """Train IRM model."""
        from core.comparison.train import train_irm
        return train_irm(args, spec_data, train_loader, val_loader, 'crmnist')

    def _compute_accuracy(self, model, data_loader, model_type: str, args) -> float:
        """Compute accuracy on a data loader."""
        from core.utils import process_batch

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y, r = process_batch(batch, args.device, dataset_type='crmnist')

                # Get predictions based on model type
                if model_type in ['nvae', 'diva']:
                    # VAE models
                    outputs = model.forward(y, x, r)
                    y_hat = outputs[7]  # y_hat is at index 7
                    predictions = y_hat.argmax(dim=1)
                elif model_type == 'dann':
                    # DANN
                    y_logits, _ = model.forward(x, y, r)
                    predictions = y_logits.argmax(dim=1)
                elif model_type == 'dann_augmented':
                    # AugmentedDANN
                    outputs = model.dann_forward(x)
                    predictions = outputs['y_pred_main'].argmax(dim=1)
                elif model_type == 'irm':
                    # IRM
                    y_logits, _ = model.forward(x, y, r)
                    predictions = y_logits.argmax(dim=1)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Convert labels if one-hot
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = y.argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0

    def _evaluate_model(
        self,
        model,
        id_test_loader,
        ood_test_loader,
        model_type: str,
        args,
    ) -> Dict[str, float]:
        """
        Evaluate model on both ID and OOD test sets.

        Returns metrics including ID accuracy, OOD accuracy, and generalization gap.
        """
        # Evaluate on ID test set
        id_accuracy = self._compute_accuracy(model, id_test_loader, model_type, args)

        # Evaluate on OOD test set
        ood_accuracy = self._compute_accuracy(model, ood_test_loader, model_type, args)

        # Calculate generalization gap
        gen_gap = id_accuracy - ood_accuracy

        return {
            'test_accuracy': id_accuracy,      # ID accuracy (for backward compatibility)
            'id_accuracy': id_accuracy,        # ID accuracy (explicit name)
            'ood_accuracy': ood_accuracy,      # OOD accuracy
            'gen_gap': gen_gap,                # Generalization gap
        }

    def _load_experiment_results(self, exp_name: str) -> Dict[str, Any]:
        """Load results from a completed experiment."""
        metrics_path = os.path.join(self.experiments_dir, exp_name, 'metrics.json')
        with open(metrics_path, 'r') as f:
            return json.load(f)

    def run_grid_search(
        self,
        configs: List[Dict[str, Any]] = None,
        quick: bool = False,
        models: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the full grid search.

        Args:
            configs: List of configurations to run. If None, generates from presets.
            quick: If True, run quick screening configs only.
            models: List of model types to include.

        Returns:
            List of results from all experiments
        """
        if configs is None:
            if quick:
                configs = get_quick_configs(models)
            else:
                configs = get_all_configs(models)

        total = len(configs)
        remaining = len([c for c in configs if c['name'] not in self.completed_experiments])

        print(f"\n{'='*60}")
        print(f"CRMNIST Grid Search")
        print(f"{'='*60}")
        print(f"Total configurations: {total}")
        print(f"Already completed: {total - remaining}")
        print(f"Remaining: {remaining}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        results = []
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{total}] ", end="")
            result = self.run_experiment(config)
            results.append(result)

        # Save summary
        summary_path = os.path.join(self.output_dir, 'grid_search_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'total_experiments': total,
                'completed': len(self.completed_experiments),
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Grid search completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

        return results


def run_single_config(
    config_name: str,
    output_dir: str,
    models: List[str] = None,
    device: str = None,
):
    """
    Run a single configuration by name.

    Args:
        config_name: Name of the configuration to run
        output_dir: Directory to save results
        models: List of model types to search in
        device: Device to use
    """
    # Find the configuration
    all_configs = get_all_configs(models)
    config = None
    for c in all_configs:
        if c['name'] == config_name:
            config = c
            break

    if config is None:
        print(f"Configuration '{config_name}' not found!")
        print("Available configurations:")
        for c in all_configs:
            print(f"  - {c['name']}")
        return None

    # Run the experiment
    runner = GridSearchRunner(output_dir, device=device)
    return runner.run_experiment(config)

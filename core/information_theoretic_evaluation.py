"""
Information-Theoretic Evaluation of Minimal Information Partition

This module implements rigorous information-theoretic testing to evaluate whether
learned representations adhere to the Minimal Information Partition theorem.

The Minimal Information Partition theorem states that entropy H(X) decomposes as:
    H(X) = I_Y + I_D + I_YD + I_X
where:
    I_Y = I(X;Y|D) - class-specific information
    I_D = I(X;D|Y) - domain-specific information
    I_YD = I(X;Y;D) - shared/interaction information
    I_X = H(X) - I(X;Y,D) - residual information

For a representation Z = (Z_y, Z_dy, Z_d, Z_x) to be minimally partitioned:
    1. Z_y captures class-specific: I(Z_y;Y|D) = I_Y and I(Z_y;D|Y) = 0
    2. Z_d captures domain-specific: I(Z_d;D|Y) = I_D and I(Z_d;Y|D) = 0
    3. Z_dy captures shared: I(Z_dy;Y;D) = I_YD
    4. Z_x is residual: I(Z_x;Y,D) = 0

Compatible Models:
    This evaluation framework requires models with explicit VAE-style latent
    decomposition. Compatible models must have:
    - qz() method for extracting latent distributions
    - index_range attributes (zy_index_range, zx_index_range, za_index_range, zay_index_range)
    - diva flag indicating model type

    Currently supported:
    - NVAE: Full 4-space decomposition (z_y, z_dy, z_d, z_x)
    - DIVA: 3-space decomposition (z_y, z_d, z_x) - no z_dy

    Not supported:
    - Baseline DANN/IRM: Use monolithic feature representations without explicit
      latent partitioning, making them incompatible with information partition
      evaluation.
"""

import numpy as np
import torch
from npeet import entropy_estimators as ee
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import json
import os


class MinimalInformationPartitionEvaluator:
    """
    Evaluate whether learned representations satisfy the Minimal Information
    Partition definition using information-theoretic quantities.

    Uses KNN-based estimators (Kraskov et al. 2003) for continuous-discrete
    mutual information estimation via the NPEET library.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_bootstrap: int = 100,
        max_dims: int = 30,
        pca_variance: float = 0.95,
        random_state: int = 42
    ):
        """
        Args:
            n_neighbors: Number of neighbors for KNN estimation (default=5)
            n_bootstrap: Number of bootstrap samples for confidence intervals (default=100)
            max_dims: Maximum dimensions before applying PCA (default=30)
            pca_variance: Variance to preserve when applying PCA (default=0.95)
            random_state: Random seed for reproducibility
        """
        self.k = n_neighbors
        self.n_bootstrap = n_bootstrap
        self.max_dims = max_dims
        self.pca_variance = pca_variance
        self.random_state = random_state
        np.random.seed(random_state)

    def _to_list_format(self, arr: np.ndarray) -> List[List[float]]:
        """Convert numpy array to list of lists for NPEET"""
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        return arr.tolist()

    def _apply_pca_if_needed(
        self,
        Z: np.ndarray,
        name: str = "Z"
    ) -> Tuple[np.ndarray, Optional[PCA]]:
        """
        Apply PCA dimensionality reduction if Z has more than max_dims dimensions.

        Args:
            Z: Latent array of shape (n_samples, n_dims)
            name: Name of latent for logging

        Returns:
            Tuple of (potentially reduced Z, PCA object or None)
        """
        if Z.shape[1] <= self.max_dims:
            return Z, None

        print(f"  {name} has {Z.shape[1]} dims > {self.max_dims}, applying PCA...")
        pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
        Z_reduced = pca.fit_transform(Z)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"    Reduced to {Z_reduced.shape[1]} dims (explained variance: {explained_var:.2%})")

        return Z_reduced, pca

    def mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        apply_pca: bool = True
    ) -> float:
        """
        Compute I(X;Y) - mutual information.

        Args:
            X: Continuous variable array of shape (n_samples, n_dims)
            Y: Discrete variable array of shape (n_samples,) or (n_samples, 1)
            apply_pca: Whether to apply PCA if X is high-dimensional

        Returns:
            Mutual information in nats
        """
        if apply_pca:
            X, _ = self._apply_pca_if_needed(X, "X")

        x_list = self._to_list_format(X)
        y_list = self._to_list_format(Y)

        try:
            mi = ee.mi(x_list, y_list, k=self.k)
            return max(0.0, mi)  # MI cannot be negative
        except Exception as e:
            print(f"Warning: MI estimation failed: {e}")
            return 0.0

    def conditional_mi(
        self,
        Z: np.ndarray,
        Y: np.ndarray,
        D_cond: np.ndarray,
        apply_pca: bool = True
    ) -> float:
        """
        Compute I(Z;Y|D) - conditional mutual information.

        Measures how much information Z shares with Y when D is known.

        Args:
            Z: Continuous latent array of shape (n_samples, n_dims)
            Y: Discrete target array of shape (n_samples,)
            D_cond: Discrete conditioning variable of shape (n_samples,)
            apply_pca: Whether to apply PCA if Z is high-dimensional

        Returns:
            Conditional mutual information in nats
        """
        if apply_pca:
            Z, _ = self._apply_pca_if_needed(Z, "Z")

        z_list = self._to_list_format(Z)
        y_list = self._to_list_format(Y)
        d_list = self._to_list_format(D_cond)

        try:
            cmi = ee.cmi(z_list, y_list, d_list, k=self.k)
            return max(0.0, cmi)  # CMI cannot be negative
        except Exception as e:
            print(f"Warning: CMI estimation failed: {e}")
            return 0.0

    def interaction_information(
        self,
        Z: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        apply_pca: bool = True
    ) -> float:
        """
        Compute I(Z;Y;D) - three-way interaction information (co-information).

        Measures the synergistic interaction between Z, Y, and D.
        Uses formula: I(Z;Y;D) = I(Z;Y) - I(Z;Y|D)

        Validated with alternative: I(Z;D) - I(Z;D|Y)

        Args:
            Z: Continuous latent array of shape (n_samples, n_dims)
            Y: Discrete class labels of shape (n_samples,)
            D: Discrete domain labels of shape (n_samples,)
            apply_pca: Whether to apply PCA if Z is high-dimensional

        Returns:
            Interaction information in nats (can be positive or negative)
        """
        if apply_pca:
            Z, _ = self._apply_pca_if_needed(Z, "Z")

        z_list = self._to_list_format(Z)
        y_list = self._to_list_format(Y)
        d_list = self._to_list_format(D)

        try:
            # Formula 1: I(Z;Y;D) = I(Z;Y) - I(Z;Y|D)
            i_zy = ee.mi(z_list, y_list, k=self.k)
            i_zy_given_d = ee.cmi(z_list, y_list, d_list, k=self.k)
            interaction1 = i_zy - i_zy_given_d

            # Formula 2: I(Z;Y;D) = I(Z;D) - I(Z;D|Y) (validation)
            i_zd = ee.mi(z_list, d_list, k=self.k)
            i_zd_given_y = ee.cmi(z_list, d_list, y_list, k=self.k)
            interaction2 = i_zd - i_zd_given_y

            # Return average of both estimates for robustness
            return (interaction1 + interaction2) / 2.0

        except Exception as e:
            print(f"Warning: Interaction information estimation failed: {e}")
            return 0.0

    def joint_mutual_information(
        self,
        Z: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        apply_pca: bool = True
    ) -> float:
        """
        Compute I(Z;Y,D) - joint mutual information.

        Measures how much information Z shares with the joint variable (Y,D).
        Uses formula: I(Z;Y,D) = I(Z;Y) + I(Z;D|Y)

        Args:
            Z: Continuous latent array of shape (n_samples, n_dims)
            Y: Discrete class labels of shape (n_samples,)
            D: Discrete domain labels of shape (n_samples,)
            apply_pca: Whether to apply PCA if Z is high-dimensional

        Returns:
            Joint mutual information in nats
        """
        if apply_pca:
            Z, _ = self._apply_pca_if_needed(Z, "Z")

        z_list = self._to_list_format(Z)
        y_list = self._to_list_format(Y)
        d_list = self._to_list_format(D)

        try:
            # I(Z;Y,D) = I(Z;Y) + I(Z;D|Y)
            i_zy = ee.mi(z_list, y_list, k=self.k)
            i_zd_given_y = ee.cmi(z_list, d_list, y_list, k=self.k)

            return max(0.0, i_zy + i_zd_given_y)

        except Exception as e:
            print(f"Warning: Joint MI estimation failed: {e}")
            return 0.0

    def bootstrap_estimate(
        self,
        estimator_func,
        *args,
        **kwargs
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for an information-theoretic quantity.

        Args:
            estimator_func: Function to estimate (e.g., self.conditional_mi)
            *args: Arrays to pass to estimator
            **kwargs: Additional arguments for estimator

        Returns:
            Tuple of (mean_estimate, (lower_ci, upper_ci))
        """
        n_samples = args[0].shape[0]
        estimates = []

        for _ in range(self.n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            args_resampled = [arr[indices] if isinstance(arr, np.ndarray) else arr
                            for arr in args]

            # Compute estimate on bootstrap sample
            estimate = estimator_func(*args_resampled, **kwargs)
            estimates.append(estimate)

        estimates = np.array(estimates)
        mean_est = np.mean(estimates)
        lower_ci = np.percentile(estimates, 2.5)
        upper_ci = np.percentile(estimates, 97.5)

        return mean_est, (lower_ci, upper_ci)

    def evaluate_latent_partition(
        self,
        z_y: np.ndarray,
        z_d: np.ndarray,
        z_dy: Optional[np.ndarray],
        z_x: np.ndarray,
        y_labels: np.ndarray,
        d_labels: np.ndarray,
        compute_bootstrap: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate all information-theoretic quantities for the learned partition.

        Args:
            z_y: Class-specific latent of shape (n_samples, dim_zy)
            z_d: Domain-specific latent of shape (n_samples, dim_zd)
            z_dy: Interaction latent of shape (n_samples, dim_zdy) or None for DIVA
            z_x: Residual latent of shape (n_samples, dim_zx)
            y_labels: Class labels of shape (n_samples,)
            d_labels: Domain labels of shape (n_samples,)
            compute_bootstrap: Whether to compute bootstrap CIs (slower)

        Returns:
            Dictionary of all computed information quantities with optional CIs
        """
        print("\n" + "="*80)
        print("Computing Information-Theoretic Quantities...")
        print("="*80)
        print(f"Samples: {z_y.shape[0]}")
        print(f"Latent dims: z_y={z_y.shape[1]}, z_d={z_d.shape[1]}, ", end="")
        if z_dy is not None:
            print(f"z_dy={z_dy.shape[1]}, ", end="")
        print(f"z_x={z_x.shape[1]}")
        print(f"Bootstrap: {'Yes (n={})'.format(self.n_bootstrap) if compute_bootstrap else 'No'}")
        print(f"k-neighbors: {self.k}")
        print("="*80 + "\n")

        results = {
            'config': {
                'n_samples': int(z_y.shape[0]),
                'n_neighbors': int(self.k),
                'n_bootstrap': int(self.n_bootstrap) if compute_bootstrap else 0,
                'dims': {
                    'z_y': int(z_y.shape[1]),
                    'z_d': int(z_d.shape[1]),
                    'z_dy': int(z_dy.shape[1]) if z_dy is not None else 0,
                    'z_x': int(z_x.shape[1])
                }
            },
            'metrics': {},
            'confidence_intervals': {} if compute_bootstrap else None
        }

        # 1. Evaluate z_y (class-specific latent)
        print("📊 Evaluating z_y (class-specific latent)...")

        if compute_bootstrap:
            results['metrics']['I(z_y;Y|D)'], results['confidence_intervals']['I(z_y;Y|D)'] = \
                self.bootstrap_estimate(self.conditional_mi, z_y, y_labels, d_labels)
            results['metrics']['I(z_y;D|Y)'], results['confidence_intervals']['I(z_y;D|Y)'] = \
                self.bootstrap_estimate(self.conditional_mi, z_y, d_labels, y_labels)
            results['metrics']['I(z_y;Y)'], results['confidence_intervals']['I(z_y;Y)'] = \
                self.bootstrap_estimate(self.mutual_information, z_y, y_labels, apply_pca=True)
            results['metrics']['I(z_y;D)'], results['confidence_intervals']['I(z_y;D)'] = \
                self.bootstrap_estimate(self.mutual_information, z_y, d_labels, apply_pca=True)
        else:
            results['metrics']['I(z_y;Y|D)'] = self.conditional_mi(z_y, y_labels, d_labels)
            results['metrics']['I(z_y;D|Y)'] = self.conditional_mi(z_y, d_labels, y_labels)
            results['metrics']['I(z_y;Y)'] = self.mutual_information(z_y, y_labels)
            results['metrics']['I(z_y;D)'] = self.mutual_information(z_y, d_labels)

        results['metrics']['z_y_specificity'] = \
            results['metrics']['I(z_y;Y|D)'] - results['metrics']['I(z_y;D|Y)']

        print(f"  I(z_y;Y|D) = {results['metrics']['I(z_y;Y|D)']:.4f} (should be HIGH)")
        print(f"  I(z_y;D|Y) = {results['metrics']['I(z_y;D|Y)']:.4f} (should be LOW)")
        print(f"  Specificity = {results['metrics']['z_y_specificity']:.4f}")

        # 2. Evaluate z_d (domain-specific latent)
        print("\n📊 Evaluating z_d (domain-specific latent)...")

        if compute_bootstrap:
            results['metrics']['I(z_d;D|Y)'], results['confidence_intervals']['I(z_d;D|Y)'] = \
                self.bootstrap_estimate(self.conditional_mi, z_d, d_labels, y_labels)
            results['metrics']['I(z_d;Y|D)'], results['confidence_intervals']['I(z_d;Y|D)'] = \
                self.bootstrap_estimate(self.conditional_mi, z_d, y_labels, d_labels)
            results['metrics']['I(z_d;D)'], results['confidence_intervals']['I(z_d;D)'] = \
                self.bootstrap_estimate(self.mutual_information, z_d, d_labels, apply_pca=True)
            results['metrics']['I(z_d;Y)'], results['confidence_intervals']['I(z_d;Y)'] = \
                self.bootstrap_estimate(self.mutual_information, z_d, y_labels, apply_pca=True)
        else:
            results['metrics']['I(z_d;D|Y)'] = self.conditional_mi(z_d, d_labels, y_labels)
            results['metrics']['I(z_d;Y|D)'] = self.conditional_mi(z_d, y_labels, d_labels)
            results['metrics']['I(z_d;D)'] = self.mutual_information(z_d, d_labels)
            results['metrics']['I(z_d;Y)'] = self.mutual_information(z_d, y_labels)

        results['metrics']['z_d_specificity'] = \
            results['metrics']['I(z_d;D|Y)'] - results['metrics']['I(z_d;Y|D)']

        print(f"  I(z_d;D|Y) = {results['metrics']['I(z_d;D|Y)']:.4f} (should be HIGH)")
        print(f"  I(z_d;Y|D) = {results['metrics']['I(z_d;Y|D)']:.4f} (should be LOW)")
        print(f"  Specificity = {results['metrics']['z_d_specificity']:.4f}")

        # 3. Evaluate z_dy (interaction latent) if it exists
        if z_dy is not None:
            print("\n📊 Evaluating z_dy (interaction latent)...")

            if compute_bootstrap:
                results['metrics']['I(z_dy;Y;D)'], results['confidence_intervals']['I(z_dy;Y;D)'] = \
                    self.bootstrap_estimate(self.interaction_information, z_dy, y_labels, d_labels)
                results['metrics']['I(z_dy;Y)'], results['confidence_intervals']['I(z_dy;Y)'] = \
                    self.bootstrap_estimate(self.mutual_information, z_dy, y_labels, apply_pca=True)
                results['metrics']['I(z_dy;D)'], results['confidence_intervals']['I(z_dy;D)'] = \
                    self.bootstrap_estimate(self.mutual_information, z_dy, d_labels, apply_pca=True)
                results['metrics']['I(z_dy;Y,D)'], results['confidence_intervals']['I(z_dy;Y,D)'] = \
                    self.bootstrap_estimate(self.joint_mutual_information, z_dy, y_labels, d_labels)
            else:
                results['metrics']['I(z_dy;Y;D)'] = self.interaction_information(z_dy, y_labels, d_labels)
                results['metrics']['I(z_dy;Y)'] = self.mutual_information(z_dy, y_labels)
                results['metrics']['I(z_dy;D)'] = self.mutual_information(z_dy, d_labels)
                results['metrics']['I(z_dy;Y,D)'] = self.joint_mutual_information(z_dy, y_labels, d_labels)

            print(f"  I(z_dy;Y;D) = {results['metrics']['I(z_dy;Y;D)']:.4f} (interaction, can be +/-)")
            print(f"  I(z_dy;Y,D) = {results['metrics']['I(z_dy;Y,D)']:.4f} (joint info)")
        else:
            print("\n⚠️  No z_dy latent (DIVA model) - skipping interaction evaluation")
            results['metrics']['I(z_dy;Y;D)'] = 0.0
            results['metrics']['I(z_dy;Y)'] = 0.0
            results['metrics']['I(z_dy;D)'] = 0.0
            results['metrics']['I(z_dy;Y,D)'] = 0.0

        # 4. Evaluate z_x (residual latent)
        print("\n📊 Evaluating z_x (residual latent)...")

        if compute_bootstrap:
            results['metrics']['I(z_x;Y,D)'], results['confidence_intervals']['I(z_x;Y,D)'] = \
                self.bootstrap_estimate(self.joint_mutual_information, z_x, y_labels, d_labels)
            results['metrics']['I(z_x;Y)'], results['confidence_intervals']['I(z_x;Y)'] = \
                self.bootstrap_estimate(self.mutual_information, z_x, y_labels, apply_pca=True)
            results['metrics']['I(z_x;D)'], results['confidence_intervals']['I(z_x;D)'] = \
                self.bootstrap_estimate(self.mutual_information, z_x, d_labels, apply_pca=True)
        else:
            results['metrics']['I(z_x;Y,D)'] = self.joint_mutual_information(z_x, y_labels, d_labels)
            results['metrics']['I(z_x;Y)'] = self.mutual_information(z_x, y_labels)
            results['metrics']['I(z_x;D)'] = self.mutual_information(z_x, d_labels)

        print(f"  I(z_x;Y,D) = {results['metrics']['I(z_x;Y,D)']:.4f} (should be LOW)")

        # 5. Compute overall partition quality score
        results['metrics']['partition_quality'] = self._compute_partition_quality(results['metrics'])

        print("\n" + "="*80)
        print("📈 OVERALL PARTITION QUALITY SCORE")
        print("="*80)
        print(f"Score: {results['metrics']['partition_quality']:.4f} / 1.000")
        print("(Higher is better - indicates better adherence to Minimal Information Partition)")
        print("="*80 + "\n")

        return results

    def _compute_partition_quality(self, metrics: Dict[str, float]) -> float:
        """
        Compute overall partition quality score (0-1, higher is better).

        Good partition should have:
        - High I(z_y;Y|D) and low I(z_y;D|Y)
        - High I(z_d;D|Y) and low I(z_d;Y|D)
        - High positive I(z_dy;Y;D) if z_dy exists
        - Low I(z_x;Y,D)
        """
        # Normalize by typical entropy values (log(num_classes) ~ 1-3 nats)
        typical_entropy = 2.5

        # Positive contributions (what we want high)
        score = 0.0
        score += metrics['I(z_y;Y|D)'] / typical_entropy  # Class specificity
        score += metrics['I(z_d;D|Y)'] / typical_entropy  # Domain specificity
        if metrics['I(z_dy;Y;D)'] > 0:
            score += metrics['I(z_dy;Y;D)'] / typical_entropy  # Interaction capture

        # Negative contributions (penalties for what should be low)
        score -= metrics['I(z_y;D|Y)'] / typical_entropy  # Class shouldn't have domain info
        score -= metrics['I(z_d;Y|D)'] / typical_entropy  # Domain shouldn't have class info
        score -= metrics['I(z_x;Y,D)'] / typical_entropy  # Residual should have no label info

        # Normalize to 0-1 range
        max_score = 3.0  # Assuming 3 positive terms
        normalized_score = score / max_score

        return float(max(0.0, min(1.0, normalized_score)))

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Compare multiple models on partition quality.

        Args:
            model_results: Dict mapping model_name -> results_dict from evaluate_latent_partition

        Returns:
            Comparison statistics and rankings
        """
        comparison = {
            'model_scores': {},
            'rankings': {},
            'metric_comparison': {}
        }

        # Extract partition quality scores
        for model_name, results in model_results.items():
            comparison['model_scores'][model_name] = results['metrics']['partition_quality']

        # Rank models
        sorted_models = sorted(
            comparison['model_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        comparison['rankings'] = {
            model: rank + 1
            for rank, (model, score) in enumerate(sorted_models)
        }

        # Detailed metric comparison
        metrics_to_compare = [
            'I(z_y;Y|D)', 'I(z_y;D|Y)', 'z_y_specificity',
            'I(z_d;D|Y)', 'I(z_d;Y|D)', 'z_d_specificity',
            'I(z_dy;Y;D)', 'I(z_x;Y,D)',
            'partition_quality'
        ]

        for metric in metrics_to_compare:
            comparison['metric_comparison'][metric] = {}
            for model_name, results in model_results.items():
                value = results['metrics'].get(metric, 0.0)
                comparison['metric_comparison'][metric][model_name] = value

                # Add CI if available
                if results.get('confidence_intervals') and metric in results['confidence_intervals']:
                    ci = results['confidence_intervals'][metric]
                    comparison['metric_comparison'][metric][f'{model_name}_CI'] = ci

        return comparison

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Results saved to: {output_path}")


def extract_latents_from_model(
    model,
    dataloader,
    device,
    max_batches: int = 200
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract latent representations from a trained model.

    Args:
        model: Trained VAE model (NVAE or DIVA)
        dataloader: DataLoader with (x, y, metadata) tuples
        device: torch device
        max_batches: Maximum number of batches to process

    Returns:
        Tuple of (z_y, z_d, z_dy, z_x, y_labels, d_labels)
        where z_dy is None for DIVA models
    """
    model.eval()

    # Detect AugmentedDANN model (uses 3-component latent structure)
    is_augmented_dann = (hasattr(model, 'extract_features') and
                         hasattr(model, 'name') and
                         model.name == 'dann')

    if is_augmented_dann:
        print("Detected AugmentedDANN model - using extract_features() for 3-space latent extraction")
        has_zdy = True  # AugmentedDANN always has zdy
    else:
        has_zdy = not model.diva

    all_zy, all_zd, all_zdy, all_zx = [], [], [], []
    all_y, all_d = [], []

    print(f"Extracting latents from model (max {max_batches} batches)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=max_batches)):
            if batch_idx >= max_batches:
                break

            x, y = batch[0].to(device), batch[1].to(device)

            # Extract domain labels (dataset-specific logic)
            if len(batch) > 2:
                if hasattr(batch[2], 'shape') and len(batch[2].shape) > 1:
                    # WILD: metadata[:, 0] is hospital_id
                    d = batch[2][:, 0].to(device)
                else:
                    # CRMNIST: direct domain label
                    d = batch[2].to(device)
            else:
                # Fallback if no domain labels
                d = torch.zeros_like(y)

            if is_augmented_dann:
                # AugmentedDANN has 3 true latent spaces: zy, zd, zdy
                # Use extract_features() to get them without duplicates
                try:
                    features = model.extract_features(x)
                    assert len(features) == 3, f"Expected 3 latent spaces from extract_features(), got {len(features)}"
                    zy, zd, zdy = features
                except Exception as e:
                    raise RuntimeError(f"Failed to extract AugmentedDANN features: {e}")

                # AugmentedDANN has no residual space (no decoder/reconstruction)
                # Use zero tensor for zx to represent this correctly in IT metrics
                # This will correctly show I(z_x;Y,D) ≈ 0 (no residual information)
                zx = torch.zeros_like(zd)
            else:
                # VAE models (NVAE/DIVA) - use qz() for latent distribution
                qz_loc, qz_scale = model.qz(x)

                zy = qz_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = qz_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                zd = qz_loc[:, model.za_index_range[0]:model.za_index_range[1]]

                if has_zdy:
                    zdy = qz_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]

            all_zy.append(zy.cpu())
            all_zd.append(zd.cpu())
            all_zx.append(zx.cpu())
            if has_zdy:
                all_zdy.append(zdy.cpu())
            all_y.append(y.cpu())
            all_d.append(d.cpu())

    # Concatenate all batches
    z_y = torch.cat(all_zy, dim=0).numpy()
    z_d = torch.cat(all_zd, dim=0).numpy()
    z_x = torch.cat(all_zx, dim=0).numpy()
    z_dy = torch.cat(all_zdy, dim=0).numpy() if has_zdy else None
    y_labels = torch.cat(all_y, dim=0).numpy()
    d_labels = torch.cat(all_d, dim=0).numpy()

    print(f"Extracted {z_y.shape[0]} samples")
    print(f"  z_y: {z_y.shape}, z_d: {z_d.shape}, z_x: {z_x.shape}")
    if z_dy is not None:
        print(f"  z_dy: {z_dy.shape}")

    return z_y, z_d, z_dy, z_x, y_labels, d_labels


def evaluate_model(
    model,
    dataloader,
    device,
    max_batches: int = 200,
    n_bootstrap: int = 100
) -> Dict[str, any]:
    """
    Complete evaluation pipeline for a single model.

    Args:
        model: Trained VAE model
        dataloader: DataLoader for evaluation
        device: torch device
        max_batches: Number of batches to use (controls sample size)
        n_bootstrap: Number of bootstrap iterations for CIs

    Returns:
        Dictionary of evaluation results
    """
    # Extract latents
    z_y, z_d, z_dy, z_x, y_labels, d_labels = extract_latents_from_model(
        model, dataloader, device, max_batches
    )

    # Evaluate partition
    evaluator = MinimalInformationPartitionEvaluator(
        n_neighbors=5,
        n_bootstrap=n_bootstrap,
        max_dims=30,
        pca_variance=0.95
    )

    results = evaluator.evaluate_latent_partition(
        z_y, z_d, z_dy, z_x,
        y_labels, d_labels,
        compute_bootstrap=(n_bootstrap > 0)
    )

    return results

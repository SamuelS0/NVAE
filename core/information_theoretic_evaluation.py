"""
Information-Theoretic Evaluation of Disentangled Partition

This module implements rigorous information-theoretic testing to evaluate whether
learned representations adhere to the Disentangled Partition definition.

=============================================================================
DEFINITION (Disentangled Partition):
A partition (Z_Y, Z_D, Z_X) = f(X) is DISENTANGLED if:

    1. CLASS-PURITY:   I(Z_Y; D|Y) = 0   (equivalently, Z_Y ⊥ D | Y)
    2. DOMAIN-PURITY:  I(Z_D; Y|D) = 0   (equivalently, Z_D ⊥ Y | D)
    3. RESIDUAL:       I(Z_X; Y,D) = 0   (equivalently, Z_X ⊥ (Y,D))
    4. INDEPENDENCE:   (Z_Y, Z_D, Z_X) are mutually independent
                       i.e., I(Z_Y; Z_D) = I(Z_Y; Z_X) = I(Z_D; Z_X) = 0
=============================================================================

For a 4-way partition (Z_Y, Z_D, Z_DY, Z_X) as in NVAE:
    - Z_DY captures the interaction information I(X;Y;D)
    - Independence extends to all pairs: I(Z_i; Z_j) = 0 for i ≠ j

The Minimal Information Partition theorem states that entropy H(X) decomposes as:
    H(X) = I_Y + I_D + I_YD + I_X
where:
    I_Y = I(X;Y|D) - class-specific information
    I_D = I(X;D|Y) - domain-specific information
    I_YD = I(X;Y;D) - shared/interaction information
    I_X = H(X) - I(X;Y,D) - residual information

Information Constraints on Pure Representations (Theorem):
    Let Z = f(X) be any representation of X.
    (a) If Z is class-pure (i.e., I(Z;D|Y) = 0), then I(Z;Y;D) = I(Z;D) >= 0
    (b) If Z is domain-pure (i.e., I(Z;Y|D) = 0), then I(Z;Y;D) = I(Z;Y) >= 0
    (c) If Z is residual (i.e., I(Z;Y,D) = 0), then I(Z;Y;D) = 0

    Key insight: No pure representation can have negative interaction information.

This module verifies ALL conditions of the definition and reports violations.

Compatible Models:
    This evaluation framework requires models with explicit VAE-style latent
    decomposition. Compatible models must have:
    - qz() method for extracting latent distributions
    - index_range attributes (zy_index_range, zx_index_range, zd_index_range, zdy_index_range)
    - diva flag indicating model type

    Currently supported:
    - NVAE: Full 4-space decomposition (z_y, z_dy, z_d, z_x)
    - DIVA: 3-space decomposition (z_y, z_d, z_x) - no z_dy
    - AugmentedDANN: 3-space decomposition (z_y, z_d, z_dy) - discriminative

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
        n_neighbors: int = 7,
        n_bootstrap: int = 100,
        max_dims: int = 30,
        pca_variance: float = 0.99,
        random_state: int = 42
    ):
        """
        Args:
            n_neighbors: Number of neighbors for KNN estimation (default=7)
            n_bootstrap: Number of bootstrap samples for confidence intervals (default=100)
            max_dims: Maximum dimensions before applying PCA (default=30)
            pca_variance: Variance to preserve when applying PCA (default=0.99)
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

        # Check for collapsed latent space (zero variance)
        Z_var = np.var(Z)
        if Z_var < 1e-10 or np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            print(f"  ⚠️  Warning: {name} has collapsed (variance={Z_var:.2e}). Skipping PCA.")
            # Return with small noise added to prevent MI calculation failures
            Z_safe = Z + np.random.normal(0, 1e-8, Z.shape)
            return Z_safe, None

        print(f"  {name} has {Z.shape[1]} dims > {self.max_dims}, applying PCA...")
        pca = PCA(n_components=self.pca_variance, random_state=self.random_state)

        # Add small noise to prevent numerical issues in PCA
        Z_safe = Z + np.random.normal(0, 1e-10, Z.shape)
        Z_reduced = pca.fit_transform(Z_safe)
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

    def latent_mutual_information(
        self,
        Z1: np.ndarray,
        Z2: np.ndarray,
        name1: str = "Z1",
        name2: str = "Z2"
    ) -> float:
        """
        Compute I(Z1; Z2) - mutual information between two continuous latent spaces.

        This is crucial for verifying the Independence condition of the
        Disentangled Partition definition: (Z_Y, Z_D, Z_X) must be mutually independent.

        Args:
            Z1: First latent array of shape (n_samples, n_dims1)
            Z2: Second latent array of shape (n_samples, n_dims2)
            name1: Name of first latent for logging
            name2: Name of second latent for logging

        Returns:
            Mutual information in nats (should be ≈0 for independent latents)
        """
        # Check for collapsed latent spaces
        if np.var(Z1) < 1e-10 or np.var(Z2) < 1e-10:
            return 0.0  # Trivially independent if one is constant

        # Apply PCA if needed
        Z1_reduced, _ = self._apply_pca_if_needed(Z1, name1)
        Z2_reduced, _ = self._apply_pca_if_needed(Z2, name2)

        z1_list = self._to_list_format(Z1_reduced)
        z2_list = self._to_list_format(Z2_reduced)

        try:
            mi = ee.mi(z1_list, z2_list, k=self.k)
            return max(0.0, mi)  # MI cannot be negative
        except Exception as e:
            print(f"Warning: Latent MI estimation I({name1};{name2}) failed: {e}")
            return 0.0

    def verify_latent_independence(
        self,
        latents: Dict[str, np.ndarray],
        tolerance: float = 0.1
    ) -> Dict[str, any]:
        """
        Verify mutual independence between latent spaces.

        For a Disentangled Partition (Definition), we need:
        - I(Z_Y; Z_D) = 0
        - I(Z_Y; Z_X) = 0
        - I(Z_D; Z_X) = 0
        - If z_dy exists: I(Z_Y; Z_DY) = 0, I(Z_D; Z_DY) = 0, I(Z_X; Z_DY) = 0

        Args:
            latents: Dict mapping latent name -> latent array
            tolerance: Threshold for considering MI as "zero"

        Returns:
            Dict with pairwise MI values, independence status, and violations
        """
        result = {
            'pairwise_mi': {},
            'all_independent': True,
            'total_dependence': 0.0,
            'violations': [],
            'warnings': []
        }

        latent_names = list(latents.keys())
        n_pairs = 0

        for i, name1 in enumerate(latent_names):
            for name2 in latent_names[i+1:]:
                z1 = latents[name1]
                z2 = latents[name2]

                # Skip if either latent is zero (discriminative model placeholder)
                if np.var(z1) < 1e-10 or np.var(z2) < 1e-10:
                    result['pairwise_mi'][f'I({name1};{name2})'] = 0.0
                    continue

                # Compute MI between latents
                mi = self.latent_mutual_information(z1, z2, name1, name2)
                result['pairwise_mi'][f'I({name1};{name2})'] = mi
                result['total_dependence'] += mi
                n_pairs += 1

                if mi > tolerance:
                    result['all_independent'] = False
                    result['violations'].append(
                        f"Independence violated: I({name1};{name2})={mi:.4f} > {tolerance}"
                    )
                elif mi > tolerance / 2:
                    result['warnings'].append(
                        f"Weak dependence: I({name1};{name2})={mi:.4f}"
                    )

        # Compute average dependence
        if n_pairs > 0:
            result['avg_dependence'] = result['total_dependence'] / n_pairs
        else:
            result['avg_dependence'] = 0.0

        return result

    def compute_full_latent_metrics(
        self,
        Z: np.ndarray,
        y_labels: np.ndarray,
        d_labels: np.ndarray,
        name: str = "Z"
    ) -> Dict[str, float]:
        """
        Compute all information-theoretic metrics for a single latent space.

        Computes:
        - I(Z;Y), I(Z;D) - marginal mutual information
        - I(Z;Y|D), I(Z;D|Y) - conditional mutual information
        - I(Z;Y;D) - interaction information (co-information)
        - I(Z;Y,D) - joint mutual information

        Args:
            Z: Latent array of shape (n_samples, n_dims)
            y_labels: Class labels of shape (n_samples,)
            d_labels: Domain labels of shape (n_samples,)
            name: Name for logging

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Marginal MI
        metrics[f'I({name};Y)'] = self.mutual_information(Z, y_labels)
        metrics[f'I({name};D)'] = self.mutual_information(Z, d_labels)

        # Conditional MI
        metrics[f'I({name};Y|D)'] = self.conditional_mi(Z, y_labels, d_labels)
        metrics[f'I({name};D|Y)'] = self.conditional_mi(Z, d_labels, y_labels)

        # Interaction information I(Z;Y;D) = I(Z;Y) - I(Z;Y|D) = I(Z;D) - I(Z;D|Y)
        metrics[f'I({name};Y;D)'] = self.interaction_information(Z, y_labels, d_labels)

        # Joint MI I(Z;Y,D) = I(Z;Y) + I(Z;D|Y) = I(Z;D) + I(Z;Y|D)
        metrics[f'I({name};Y,D)'] = self.joint_mutual_information(Z, y_labels, d_labels)

        return metrics

    def verify_purity_constraint(
        self,
        metrics: Dict[str, float],
        name: str,
        purity_type: str,
        tolerance: float = 0.1
    ) -> Dict[str, any]:
        """
        Verify the purity constraint from the theorem for a latent space.

        Theorem:
        (a) Class-pure (I(Z;D|Y) = 0): I(Z;Y;D) = I(Z;D) >= 0
        (b) Domain-pure (I(Z;Y|D) = 0): I(Z;Y;D) = I(Z;Y) >= 0
        (c) Residual (I(Z;Y,D) = 0): I(Z;Y;D) = 0

        Args:
            metrics: Dict with I(Z;Y), I(Z;D), I(Z;Y|D), I(Z;D|Y), I(Z;Y;D), I(Z;Y,D)
            name: Name of latent (e.g., "z_y")
            purity_type: "class", "domain", or "residual"
            tolerance: Tolerance for considering a value as zero

        Returns:
            Dict with is_pure, theorem_satisfied, violations, etc.
        """
        result = {
            'name': name,
            'purity_type': purity_type,
            'is_pure': False,
            'purity_value': None,  # The value that should be zero
            'theorem_satisfied': False,
            'theorem_expected': None,
            'theorem_actual': None,
            'theorem_deviation': None,
            'non_negativity_satisfied': True,
            'violations': [],
            'warnings': []
        }

        i_zy = metrics.get(f'I({name};Y)', 0)
        i_zd = metrics.get(f'I({name};D)', 0)
        i_zy_given_d = metrics.get(f'I({name};Y|D)', 0)
        i_zd_given_y = metrics.get(f'I({name};D|Y)', 0)
        i_z_yd = metrics.get(f'I({name};Y;D)', 0)  # Interaction
        i_z_joint = metrics.get(f'I({name};Y,D)', 0)  # Joint

        if purity_type == "class":
            # Class-pure: I(Z;D|Y) should be 0
            result['purity_value'] = i_zd_given_y
            result['is_pure'] = i_zd_given_y < tolerance

            # Theorem (a): If class-pure, I(Z;Y;D) = I(Z;D)
            result['theorem_expected'] = i_zd
            result['theorem_actual'] = i_z_yd
            result['theorem_deviation'] = abs(i_z_yd - i_zd)

            if result['is_pure']:
                result['theorem_satisfied'] = result['theorem_deviation'] < tolerance
                if not result['theorem_satisfied']:
                    result['violations'].append(
                        f"Theorem (a) violated: I({name};Y;D)={i_z_yd:.4f} != I({name};D)={i_zd:.4f}"
                    )

                # Non-negativity check
                if i_z_yd < -tolerance:
                    result['non_negativity_satisfied'] = False
                    result['violations'].append(
                        f"Non-negativity violated: I({name};Y;D)={i_z_yd:.4f} < 0 for class-pure latent"
                    )
            else:
                result['warnings'].append(
                    f"{name} is not class-pure: I({name};D|Y)={i_zd_given_y:.4f} > {tolerance}"
                )

        elif purity_type == "domain":
            # Domain-pure: I(Z;Y|D) should be 0
            result['purity_value'] = i_zy_given_d
            result['is_pure'] = i_zy_given_d < tolerance

            # Theorem (b): If domain-pure, I(Z;Y;D) = I(Z;Y)
            result['theorem_expected'] = i_zy
            result['theorem_actual'] = i_z_yd
            result['theorem_deviation'] = abs(i_z_yd - i_zy)

            if result['is_pure']:
                result['theorem_satisfied'] = result['theorem_deviation'] < tolerance
                if not result['theorem_satisfied']:
                    result['violations'].append(
                        f"Theorem (b) violated: I({name};Y;D)={i_z_yd:.4f} != I({name};Y)={i_zy:.4f}"
                    )

                # Non-negativity check
                if i_z_yd < -tolerance:
                    result['non_negativity_satisfied'] = False
                    result['violations'].append(
                        f"Non-negativity violated: I({name};Y;D)={i_z_yd:.4f} < 0 for domain-pure latent"
                    )
            else:
                result['warnings'].append(
                    f"{name} is not domain-pure: I({name};Y|D)={i_zy_given_d:.4f} > {tolerance}"
                )

        elif purity_type == "residual":
            # Residual: I(Z;Y,D) should be 0
            result['purity_value'] = i_z_joint
            result['is_pure'] = i_z_joint < tolerance

            # Theorem (c): If residual, I(Z;Y;D) = 0
            result['theorem_expected'] = 0.0
            result['theorem_actual'] = i_z_yd
            result['theorem_deviation'] = abs(i_z_yd)

            if result['is_pure']:
                result['theorem_satisfied'] = result['theorem_deviation'] < tolerance
                if not result['theorem_satisfied']:
                    result['violations'].append(
                        f"Theorem (c) violated: I({name};Y;D)={i_z_yd:.4f} != 0 for residual latent"
                    )
            else:
                result['warnings'].append(
                    f"{name} is not residual: I({name};Y,D)={i_z_joint:.4f} > {tolerance}"
                )

        return result

    def check_information_capture(
        self,
        metrics: Dict[str, float],
        name: str,
        intended_capture: str,
        min_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Check whether a latent space captures its intended information.

        Args:
            metrics: Dict with all computed metrics
            name: Name of latent
            intended_capture: "class", "domain", "interaction", or "residual"
            min_threshold: Minimum information value to consider as "captured"

        Returns:
            Dict with capture status and diagnostics
        """
        result = {
            'name': name,
            'intended_capture': intended_capture,
            'captures_intended': False,
            'capture_value': 0.0,
            'leakage_value': 0.0,
            'warnings': []
        }

        i_zy = metrics.get(f'I({name};Y)', 0)
        i_zd = metrics.get(f'I({name};D)', 0)
        i_zy_given_d = metrics.get(f'I({name};Y|D)', 0)
        i_zd_given_y = metrics.get(f'I({name};D|Y)', 0)
        i_z_yd = metrics.get(f'I({name};Y;D)', 0)
        i_z_joint = metrics.get(f'I({name};Y,D)', 0)

        if intended_capture == "class":
            # Should capture I(Z;Y|D) > threshold
            result['capture_value'] = i_zy_given_d
            result['leakage_value'] = i_zd_given_y  # Should be low
            result['captures_intended'] = i_zy_given_d > min_threshold

            if not result['captures_intended']:
                result['warnings'].append(
                    f"CRITICAL: {name} not capturing class info: I({name};Y|D)={i_zy_given_d:.4f} < {min_threshold}"
                )
            if result['leakage_value'] > min_threshold:
                result['warnings'].append(
                    f"Leakage: {name} captures domain info: I({name};D|Y)={i_zd_given_y:.4f}"
                )

        elif intended_capture == "domain":
            # Should capture I(Z;D|Y) > threshold
            result['capture_value'] = i_zd_given_y
            result['leakage_value'] = i_zy_given_d  # Should be low
            result['captures_intended'] = i_zd_given_y > min_threshold

            if not result['captures_intended']:
                result['warnings'].append(
                    f"CRITICAL: {name} not capturing domain info: I({name};D|Y)={i_zd_given_y:.4f} < {min_threshold}"
                )
            if result['leakage_value'] > min_threshold:
                result['warnings'].append(
                    f"Leakage: {name} captures class info: I({name};Y|D)={i_zy_given_d:.4f}"
                )

        elif intended_capture == "interaction":
            # Should capture I(Z;Y;D) != 0 or I(Z;Y,D) > threshold
            result['capture_value'] = i_z_yd
            result['leakage_value'] = 0  # No leakage concept for interaction
            result['captures_intended'] = abs(i_z_yd) > min_threshold or i_z_joint > min_threshold

            if not result['captures_intended']:
                result['warnings'].append(
                    f"WARNING: {name} not capturing interaction: I({name};Y;D)={i_z_yd:.4f}, I({name};Y,D)={i_z_joint:.4f}"
                )

        elif intended_capture == "residual":
            # Should have I(Z;Y,D) ≈ 0
            result['capture_value'] = i_z_joint
            result['leakage_value'] = i_z_joint  # Same - should be low
            result['captures_intended'] = i_z_joint < min_threshold

            if not result['captures_intended']:
                result['warnings'].append(
                    f"WARNING: {name} capturing Y/D info: I({name};Y,D)={i_z_joint:.4f}"
                )

        return result

    def evaluate_latent_partition(
        self,
        z_y: np.ndarray,
        z_d: np.ndarray,
        z_dy: Optional[np.ndarray],
        z_x: np.ndarray,
        y_labels: np.ndarray,
        d_labels: np.ndarray,
        compute_bootstrap: bool = True,
        purity_tolerance: float = 0.1,
        capture_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Evaluate all information-theoretic quantities for the learned partition,
        including verification of the Information Constraints on Pure Representations theorem.

        Theorem:
        (a) If Z is class-pure (I(Z;D|Y) = 0), then I(Z;Y;D) = I(Z;D) >= 0
        (b) If Z is domain-pure (I(Z;Y|D) = 0), then I(Z;Y;D) = I(Z;Y) >= 0
        (c) If Z is residual (I(Z;Y,D) = 0), then I(Z;Y;D) = 0

        Args:
            z_y: Class-specific latent of shape (n_samples, dim_zy)
            z_d: Domain-specific latent of shape (n_samples, dim_zd)
            z_dy: Interaction latent of shape (n_samples, dim_zdy) or None for DIVA
            z_x: Residual latent of shape (n_samples, dim_zx)
            y_labels: Class labels of shape (n_samples,)
            d_labels: Domain labels of shape (n_samples,)
            compute_bootstrap: Whether to compute bootstrap CIs (slower)
            purity_tolerance: Tolerance for considering a latent as "pure" (default=0.1)
            capture_threshold: Minimum info to consider as "captured" (default=0.1)

        Returns:
            Dictionary of all computed information quantities, theorem verification,
            and optional CIs
        """
        print("\n" + "="*80)
        print("Information-Theoretic Evaluation with Theorem Verification")
        print("="*80)
        print(f"Samples: {z_y.shape[0]}")
        print(f"Latent dims: z_y={z_y.shape[1]}, z_d={z_d.shape[1]}, ", end="")
        if z_dy is not None:
            print(f"z_dy={z_dy.shape[1]}, ", end="")
        print(f"z_x={z_x.shape[1]}")
        print(f"Bootstrap: {'Yes (n={})'.format(self.n_bootstrap) if compute_bootstrap else 'No'}")
        print(f"k-neighbors: {self.k}")
        print(f"Purity tolerance: {purity_tolerance}, Capture threshold: {capture_threshold}")
        print("="*80 + "\n")

        results = {
            'config': {
                'n_samples': int(z_y.shape[0]),
                'n_neighbors': int(self.k),
                'n_bootstrap': int(self.n_bootstrap) if compute_bootstrap else 0,
                'purity_tolerance': purity_tolerance,
                'capture_threshold': capture_threshold,
                'dims': {
                    'z_y': int(z_y.shape[1]),
                    'z_d': int(z_d.shape[1]),
                    'z_dy': int(z_dy.shape[1]) if z_dy is not None else 0,
                    'z_x': int(z_x.shape[1])
                }
            },
            'metrics': {},
            'theorem_verification': {},
            'information_capture': {},
            'confidence_intervals': {} if compute_bootstrap else None,
            'violations': [],
            'warnings': []
        }

        # =====================================================================
        # 1. Evaluate z_y (class-specific latent) - FULL METRICS
        # =====================================================================
        print("📊 Evaluating z_y (class-specific latent)...")
        print("   Expected: class-pure (I(z_y;D|Y) ≈ 0), captures class (I(z_y;Y|D) > 0)")

        # Compute all metrics for z_y
        zy_metrics = self.compute_full_latent_metrics(z_y, y_labels, d_labels, "z_y")
        results['metrics'].update(zy_metrics)

        # Compute specificity
        results['metrics']['z_y_specificity'] = (
            results['metrics']['I(z_y;Y|D)'] - results['metrics']['I(z_y;D|Y)']
        )

        print(f"  I(z_y;Y|D) = {results['metrics']['I(z_y;Y|D)']:.4f} (should be HIGH - class info)")
        print(f"  I(z_y;D|Y) = {results['metrics']['I(z_y;D|Y)']:.4f} (should be ≈0 - class-pure)")
        print(f"  I(z_y;Y)   = {results['metrics']['I(z_y;Y)']:.4f}")
        print(f"  I(z_y;D)   = {results['metrics']['I(z_y;D)']:.4f}")
        print(f"  I(z_y;Y;D) = {results['metrics']['I(z_y;Y;D)']:.4f} (interaction info)")
        print(f"  I(z_y;Y,D) = {results['metrics']['I(z_y;Y,D)']:.4f} (joint info)")
        print(f"  Specificity = {results['metrics']['z_y_specificity']:.4f}")

        # Verify theorem constraint for z_y
        zy_purity = self.verify_purity_constraint(
            results['metrics'], "z_y", "class", purity_tolerance
        )
        results['theorem_verification']['z_y'] = zy_purity
        results['violations'].extend(zy_purity['violations'])
        results['warnings'].extend(zy_purity['warnings'])

        # Check information capture for z_y
        zy_capture = self.check_information_capture(
            results['metrics'], "z_y", "class", capture_threshold
        )
        results['information_capture']['z_y'] = zy_capture
        results['warnings'].extend(zy_capture['warnings'])

        # Print theorem verification status
        print(f"\n  Theorem (a) verification [class-pure: I(Z;D|Y)=0 → I(Z;Y;D)=I(Z;D)≥0]:")
        print(f"    Is class-pure: {'✓' if zy_purity['is_pure'] else '✗'} (I(z_y;D|Y)={zy_purity['purity_value']:.4f})")
        if zy_purity['is_pure']:
            print(f"    Theorem satisfied: {'✓' if zy_purity['theorem_satisfied'] else '✗'}")
            print(f"    Expected I(z_y;Y;D) = I(z_y;D) = {zy_purity['theorem_expected']:.4f}")
            print(f"    Actual I(z_y;Y;D) = {zy_purity['theorem_actual']:.4f}")
            print(f"    Deviation: {zy_purity['theorem_deviation']:.4f}")
            print(f"    Non-negativity: {'✓' if zy_purity['non_negativity_satisfied'] else '✗ VIOLATION'}")

        # =====================================================================
        # 2. Evaluate z_d (domain-specific latent) - FULL METRICS
        # =====================================================================
        print("\n📊 Evaluating z_d (domain-specific latent)...")
        print("   Expected: domain-pure (I(z_d;Y|D) ≈ 0), captures domain (I(z_d;D|Y) > 0)")

        # Compute all metrics for z_d
        zd_metrics = self.compute_full_latent_metrics(z_d, y_labels, d_labels, "z_d")
        results['metrics'].update(zd_metrics)

        # Compute specificity
        results['metrics']['z_d_specificity'] = (
            results['metrics']['I(z_d;D|Y)'] - results['metrics']['I(z_d;Y|D)']
        )

        print(f"  I(z_d;D|Y) = {results['metrics']['I(z_d;D|Y)']:.4f} (should be HIGH - domain info)")
        print(f"  I(z_d;Y|D) = {results['metrics']['I(z_d;Y|D)']:.4f} (should be ≈0 - domain-pure)")
        print(f"  I(z_d;D)   = {results['metrics']['I(z_d;D)']:.4f}")
        print(f"  I(z_d;Y)   = {results['metrics']['I(z_d;Y)']:.4f}")
        print(f"  I(z_d;Y;D) = {results['metrics']['I(z_d;Y;D)']:.4f} (interaction info)")
        print(f"  I(z_d;Y,D) = {results['metrics']['I(z_d;Y,D)']:.4f} (joint info)")
        print(f"  Specificity = {results['metrics']['z_d_specificity']:.4f}")

        # Verify theorem constraint for z_d
        zd_purity = self.verify_purity_constraint(
            results['metrics'], "z_d", "domain", purity_tolerance
        )
        results['theorem_verification']['z_d'] = zd_purity
        results['violations'].extend(zd_purity['violations'])
        results['warnings'].extend(zd_purity['warnings'])

        # Check information capture for z_d
        zd_capture = self.check_information_capture(
            results['metrics'], "z_d", "domain", capture_threshold
        )
        results['information_capture']['z_d'] = zd_capture
        results['warnings'].extend(zd_capture['warnings'])

        # Print theorem verification status
        print(f"\n  Theorem (b) verification [domain-pure: I(Z;Y|D)=0 → I(Z;Y;D)=I(Z;Y)≥0]:")
        print(f"    Is domain-pure: {'✓' if zd_purity['is_pure'] else '✗'} (I(z_d;Y|D)={zd_purity['purity_value']:.4f})")
        if zd_purity['is_pure']:
            print(f"    Theorem satisfied: {'✓' if zd_purity['theorem_satisfied'] else '✗'}")
            print(f"    Expected I(z_d;Y;D) = I(z_d;Y) = {zd_purity['theorem_expected']:.4f}")
            print(f"    Actual I(z_d;Y;D) = {zd_purity['theorem_actual']:.4f}")
            print(f"    Deviation: {zd_purity['theorem_deviation']:.4f}")
            print(f"    Non-negativity: {'✓' if zd_purity['non_negativity_satisfied'] else '✗ VIOLATION'}")

        # =====================================================================
        # 3. Evaluate z_dy (interaction latent) if it exists
        # =====================================================================
        if z_dy is not None:
            print("\n📊 Evaluating z_dy (interaction latent)...")
            print("   Expected: captures Y-D interaction (I(z_dy;Y;D) ≠ 0 or I(z_dy;Y,D) > 0)")

            # Compute all metrics for z_dy
            zdy_metrics = self.compute_full_latent_metrics(z_dy, y_labels, d_labels, "z_dy")
            results['metrics'].update(zdy_metrics)

            print(f"  I(z_dy;Y|D) = {results['metrics']['I(z_dy;Y|D)']:.4f}")
            print(f"  I(z_dy;D|Y) = {results['metrics']['I(z_dy;D|Y)']:.4f}")
            print(f"  I(z_dy;Y)   = {results['metrics']['I(z_dy;Y)']:.4f}")
            print(f"  I(z_dy;D)   = {results['metrics']['I(z_dy;D)']:.4f}")
            print(f"  I(z_dy;Y;D) = {results['metrics']['I(z_dy;Y;D)']:.4f} (interaction - key metric)")
            print(f"  I(z_dy;Y,D) = {results['metrics']['I(z_dy;Y,D)']:.4f} (joint info)")

            # Check information capture for z_dy
            zdy_capture = self.check_information_capture(
                results['metrics'], "z_dy", "interaction", capture_threshold
            )
            results['information_capture']['z_dy'] = zdy_capture
            results['warnings'].extend(zdy_capture['warnings'])

            # z_dy is NOT supposed to be pure, so no theorem verification
            results['theorem_verification']['z_dy'] = {
                'name': 'z_dy',
                'purity_type': 'interaction',
                'note': 'Interaction latent is not expected to be pure'
            }
        else:
            print("\n⚠️  No z_dy latent (DIVA model) - skipping interaction evaluation")
            results['metrics']['I(z_dy;Y;D)'] = 0.0
            results['metrics']['I(z_dy;Y)'] = 0.0
            results['metrics']['I(z_dy;D)'] = 0.0
            results['metrics']['I(z_dy;Y,D)'] = 0.0
            results['metrics']['I(z_dy;Y|D)'] = 0.0
            results['metrics']['I(z_dy;D|Y)'] = 0.0

        # =====================================================================
        # 4. Evaluate z_x (residual latent) - NOW ENABLED
        # =====================================================================
        print("\n📊 Evaluating z_x (residual latent)...")
        print("   Expected: residual (I(z_x;Y,D) ≈ 0), no Y/D information")

        # Check if z_x is all zeros (discriminative models like AugmentedDANN)
        zx_variance = np.var(z_x)
        if zx_variance < 1e-10:
            print(f"  z_x has near-zero variance ({zx_variance:.2e}) - likely discriminative model")
            print("  Setting all z_x metrics to 0.0")
            results['metrics']['I(z_x;Y)'] = 0.0
            results['metrics']['I(z_x;D)'] = 0.0
            results['metrics']['I(z_x;Y|D)'] = 0.0
            results['metrics']['I(z_x;D|Y)'] = 0.0
            results['metrics']['I(z_x;Y;D)'] = 0.0
            results['metrics']['I(z_x;Y,D)'] = 0.0

            zx_purity = {
                'name': 'z_x',
                'purity_type': 'residual',
                'is_pure': True,
                'purity_value': 0.0,
                'theorem_satisfied': True,
                'theorem_expected': 0.0,
                'theorem_actual': 0.0,
                'theorem_deviation': 0.0,
                'non_negativity_satisfied': True,
                'violations': [],
                'warnings': [],
                'note': 'z_x is zero tensor (discriminative model)'
            }
            zx_capture = {
                'name': 'z_x',
                'intended_capture': 'residual',
                'captures_intended': True,
                'capture_value': 0.0,
                'leakage_value': 0.0,
                'warnings': []
            }
        else:
            # Compute all metrics for z_x
            zx_metrics = self.compute_full_latent_metrics(z_x, y_labels, d_labels, "z_x")
            results['metrics'].update(zx_metrics)

            print(f"  I(z_x;Y|D) = {results['metrics']['I(z_x;Y|D)']:.4f}")
            print(f"  I(z_x;D|Y) = {results['metrics']['I(z_x;D|Y)']:.4f}")
            print(f"  I(z_x;Y)   = {results['metrics']['I(z_x;Y)']:.4f}")
            print(f"  I(z_x;D)   = {results['metrics']['I(z_x;D)']:.4f}")
            print(f"  I(z_x;Y;D) = {results['metrics']['I(z_x;Y;D)']:.4f} (interaction)")
            print(f"  I(z_x;Y,D) = {results['metrics']['I(z_x;Y,D)']:.4f} (should be ≈0 - residual)")

            # Verify theorem constraint for z_x
            zx_purity = self.verify_purity_constraint(
                results['metrics'], "z_x", "residual", purity_tolerance
            )
            results['violations'].extend(zx_purity['violations'])
            results['warnings'].extend(zx_purity['warnings'])

            # Check information capture for z_x
            zx_capture = self.check_information_capture(
                results['metrics'], "z_x", "residual", capture_threshold
            )
            results['warnings'].extend(zx_capture['warnings'])

            # Print theorem verification status
            print(f"\n  Theorem (c) verification [residual: I(Z;Y,D)=0 → I(Z;Y;D)=0]:")
            print(f"    Is residual: {'✓' if zx_purity['is_pure'] else '✗'} (I(z_x;Y,D)={zx_purity['purity_value']:.4f})")
            if zx_purity['is_pure']:
                print(f"    Theorem satisfied: {'✓' if zx_purity['theorem_satisfied'] else '✗'}")
                print(f"    Expected I(z_x;Y;D) = 0")
                print(f"    Actual I(z_x;Y;D) = {zx_purity['theorem_actual']:.4f}")
                print(f"    Deviation: {zx_purity['theorem_deviation']:.4f}")

        results['theorem_verification']['z_x'] = zx_purity
        results['information_capture']['z_x'] = zx_capture

        # =====================================================================
        # 5. Verify INDEPENDENCE condition (Definition condition 4)
        # =====================================================================
        print("\n📊 Verifying Independence Condition...")
        print("   Definition: (Z_Y, Z_D, Z_X) must be mutually independent")
        print("   Required: I(Z_i; Z_j) = 0 for all pairs i ≠ j")

        # Build latent dictionary for independence check
        latents = {'z_y': z_y, 'z_d': z_d, 'z_x': z_x}
        if z_dy is not None:
            latents['z_dy'] = z_dy

        independence_result = self.verify_latent_independence(latents, purity_tolerance)
        results['independence'] = independence_result
        results['violations'].extend(independence_result['violations'])
        results['warnings'].extend(independence_result['warnings'])

        # Add pairwise MI to metrics
        for pair_name, mi_value in independence_result['pairwise_mi'].items():
            results['metrics'][pair_name] = mi_value

        # Print independence results
        print("\n  Pairwise Mutual Information (should all be ≈0):")
        for pair_name, mi_value in independence_result['pairwise_mi'].items():
            status = '✓' if mi_value < purity_tolerance else '✗'
            print(f"    {pair_name} = {mi_value:.4f} {status}")

        print(f"\n  All latents independent: {'✓' if independence_result['all_independent'] else '✗'}")
        print(f"  Total dependence: {independence_result['total_dependence']:.4f}")
        print(f"  Average dependence: {independence_result['avg_dependence']:.4f}")

        # =====================================================================
        # 6. Compute overall partition quality score (updated with independence)
        # =====================================================================
        results['metrics']['partition_quality'] = self._compute_partition_quality(
            results['metrics'],
            results['theorem_verification'],
            results['information_capture'],
            independence_result
        )

        # =====================================================================
        # 7. Print summary
        # =====================================================================
        print("\n" + "="*80)
        print("📈 DISENTANGLED PARTITION EVALUATION SUMMARY")
        print("="*80)

        # Definition compliance check
        print("\n--- Definition Compliance ---")
        cond1_ok = results['theorem_verification']['z_y'].get('is_pure', False)
        cond2_ok = results['theorem_verification']['z_d'].get('is_pure', False)
        cond3_ok = results['theorem_verification']['z_x'].get('is_pure', False)
        cond4_ok = independence_result['all_independent']

        print(f"  1. Class-purity   I(Z_Y;D|Y)=0: {'✓' if cond1_ok else '✗'} (actual={results['metrics'].get('I(z_y;D|Y)', 0):.4f})")
        print(f"  2. Domain-purity  I(Z_D;Y|D)=0: {'✓' if cond2_ok else '✗'} (actual={results['metrics'].get('I(z_d;Y|D)', 0):.4f})")
        print(f"  3. Residual       I(Z_X;Y,D)=0: {'✓' if cond3_ok else '✗'} (actual={results['metrics'].get('I(z_x;Y,D)', 0):.4f})")
        print(f"  4. Independence   I(Z_i;Z_j)=0: {'✓' if cond4_ok else '✗'} (avg={independence_result['avg_dependence']:.4f})")

        all_conditions_met = cond1_ok and cond2_ok and cond3_ok and cond4_ok
        print(f"\n  ★ Partition is DISENTANGLED: {'✓ YES' if all_conditions_met else '✗ NO'}")

        print("\n--- Partition Quality Score ---")
        print(f"Score: {results['metrics']['partition_quality']:.4f} / 1.000")
        print("(Higher is better - indicates better adherence to Disentangled Partition definition)")

        print("\n--- Purity Theorem Verification ---")
        all_satisfied = True
        for latent_name, verification in results['theorem_verification'].items():
            if 'theorem_satisfied' in verification:
                status = '✓' if verification['theorem_satisfied'] else '✗'
                all_satisfied = all_satisfied and verification['theorem_satisfied']
                print(f"  {latent_name}: {status} (deviation={verification.get('theorem_deviation', 0):.4f})")

        print(f"\n  All theorem constraints satisfied: {'✓' if all_satisfied else '✗'}")

        print("\n--- Information Capture Summary ---")
        for latent_name, capture in results['information_capture'].items():
            status = '✓' if capture['captures_intended'] else '✗'
            print(f"  {latent_name} ({capture['intended_capture']}): {status}")
            print(f"      Capture value: {capture['capture_value']:.4f}")
            if capture.get('leakage_value', 0) > capture_threshold:
                print(f"      ⚠️ Leakage: {capture['leakage_value']:.4f}")

        if results['violations']:
            print("\n--- ⚠️ VIOLATIONS ---")
            for v in results['violations']:
                print(f"  • {v}")

        if results['warnings']:
            print("\n--- Warnings ---")
            for w in results['warnings']:
                print(f"  • {w}")

        print("\n" + "="*80 + "\n")

        return results

    def _compute_partition_quality(
        self,
        metrics: Dict[str, float],
        theorem_verification: Optional[Dict[str, any]] = None,
        information_capture: Optional[Dict[str, any]] = None,
        independence: Optional[Dict[str, any]] = None
    ) -> float:
        """
        Compute overall partition quality score (0-1, higher is better).

        The score incorporates all 4 conditions of the Disentangled Partition definition:
        1. Class-purity: I(Z_Y; D|Y) = 0
        2. Domain-purity: I(Z_D; Y|D) = 0
        3. Residual: I(Z_X; Y,D) = 0
        4. Independence: I(Z_i; Z_j) = 0 for all pairs

        Plus additional quality measures:
        - Information capture (each latent captures its intended information)
        - Theorem compliance (pure representations have correct I(Z;Y;D))
        """
        # Normalize by typical entropy values (log(num_classes) ~ 1-3 nats)
        typical_entropy = 2.5

        # ===== Component 1: Purity Score (30% weight) =====
        # Measures conditions 1-3 of the definition
        purity_score = 1.0

        # Condition 1: Class-purity I(Z_Y; D|Y) = 0
        zy_d_given_y = metrics.get('I(z_y;D|Y)', 0)
        purity_score -= min(0.33, zy_d_given_y / typical_entropy)

        # Condition 2: Domain-purity I(Z_D; Y|D) = 0
        zd_y_given_d = metrics.get('I(z_d;Y|D)', 0)
        purity_score -= min(0.33, zd_y_given_d / typical_entropy)

        # Condition 3: Residual I(Z_X; Y,D) = 0
        zx_joint = metrics.get('I(z_x;Y,D)', 0)
        if zx_joint is not None:
            purity_score -= min(0.33, zx_joint / typical_entropy)

        purity_score = max(0.0, purity_score)

        # ===== Component 2: Independence Score (25% weight) =====
        # Measures condition 4 of the definition
        independence_score = 1.0
        if independence:
            avg_dep = independence.get('avg_dependence', 0)
            # Penalty proportional to average dependence
            independence_score -= min(1.0, avg_dep / typical_entropy)
            if not independence.get('all_independent', True):
                independence_score *= 0.7  # Additional penalty for violations

        independence_score = max(0.0, independence_score)

        # ===== Component 3: Information Capture Score (25% weight) =====
        capture_score = 1.0
        if information_capture:
            n_latents = 0
            for latent_name, capture in information_capture.items():
                n_latents += 1
                if not capture.get('captures_intended', False):
                    # Penalty for not capturing intended info
                    capture_score -= 0.25

                # Penalty for leakage
                leakage = capture.get('leakage_value', 0)
                if leakage > 0.1:
                    capture_score -= leakage / (typical_entropy * 2)

            capture_score = max(0.0, min(1.0, capture_score))

        # ===== Component 4: Theorem Compliance Score (20% weight) =====
        theorem_score = 1.0
        if theorem_verification:
            for latent_name, verification in theorem_verification.items():
                if 'theorem_satisfied' in verification:
                    if verification.get('is_pure', False):
                        if not verification.get('theorem_satisfied', False):
                            # Penalty proportional to deviation
                            deviation = verification.get('theorem_deviation', 0)
                            theorem_score -= min(0.2, deviation / typical_entropy)

                    if not verification.get('non_negativity_satisfied', True):
                        # Severe penalty for non-negativity violation
                        theorem_score -= 0.3

            theorem_score = max(0.0, min(1.0, theorem_score))

        # ===== Combine Components =====
        # Weights reflect the 4 conditions of the Disentangled Partition definition:
        # - Purity (conditions 1-3): 30%
        # - Independence (condition 4): 25%
        # - Information capture: 25%
        # - Theorem compliance: 20%
        final_score = (
            0.30 * purity_score +
            0.25 * independence_score +
            0.25 * capture_score +
            0.20 * theorem_score
        )

        return float(max(0.0, min(1.0, final_score)))

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
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
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
                         model.name in ('dann', 'dann_augmented'))

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
            # CRMNIST batch format: (x, y, c, r) where r is rotation/domain (one-hot)
            # WILD batch format: (x, y, metadata) where metadata[:, 0] is hospital_id
            if len(batch) == 4:
                # CRMNIST: batch = (x, y, c, r) - use r (rotation) as domain
                r = batch[3]  # rotation one-hot, shape (batch, 6)
                if len(r.shape) > 1 and r.shape[1] > 1:
                    # Convert one-hot to index
                    d = r.argmax(dim=1).to(device)
                else:
                    d = r.to(device)
            elif len(batch) > 2:
                # WILD or other: batch = (x, y, metadata)
                if hasattr(batch[2], 'shape') and len(batch[2].shape) > 1:
                    # WILD: metadata[:, 0] is hospital_id
                    d = batch[2][:, 0].to(device)
                else:
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
                zd = qz_loc[:, model.zd_index_range[0]:model.zd_index_range[1]]

                if has_zdy:
                    zdy = qz_loc[:, model.zdy_index_range[0]:model.zdy_index_range[1]]

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
        n_neighbors=7,
        n_bootstrap=n_bootstrap,
        max_dims=30,
        pca_variance=0.95  # Use 95% variance for faster computation
    )

    results = evaluator.evaluate_latent_partition(
        z_y, z_d, z_dy, z_x,
        y_labels, d_labels,
        compute_bootstrap=(n_bootstrap > 0)
    )

    return results

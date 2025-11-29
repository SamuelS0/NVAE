#!/usr/bin/env python3
"""
Disentanglement Quality Score (DQS)

A proper metric for evaluating disentangled representations that requires BOTH:
1. Information Capture: Each latent must capture its intended information
2. Information Purity: Each latent must not leak unintended information

This avoids the "dimension collapse" loophole where models achieve high purity
by simply not encoding any information.

The metric uses an F1-style harmonic mean to ensure both capture AND purity
must be high for a good score.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def compute_capture_efficiency(info_captured: float, max_info: float) -> float:
    """
    Compute how much of the available information was captured.

    Args:
        info_captured: Mutual information captured (e.g., I(z_y; Y|D))
        max_info: Maximum possible information (e.g., H(Y|D))

    Returns:
        Capture efficiency in [0, 1]
    """
    if max_info <= 0:
        return 0.0
    return min(info_captured / max_info, 1.0)


def compute_purity(leakage: float, scale: float = 1.0) -> float:
    """
    Compute purity score from leakage amount.

    Uses inverse scaling: purity = 1 / (1 + leakage/scale)
    - leakage = 0 → purity = 1.0
    - leakage = scale → purity = 0.5
    - leakage → ∞ → purity → 0

    Args:
        leakage: Amount of unintended information (e.g., I(z_y; D|Y))
        scale: Scaling factor for leakage severity

    Returns:
        Purity score in (0, 1]
    """
    return 1.0 / (1.0 + leakage / scale)


def harmonic_mean(a: float, b: float, eps: float = 1e-10) -> float:
    """
    Compute harmonic mean of two values (F1-style).

    H(a,b) = 2ab / (a + b)

    This ensures BOTH values must be high for a high result.
    If either is 0, the result is 0.
    """
    return 2 * a * b / (a + b + eps)


def compute_dqs(
    metrics: Dict[str, float],
    H_Y: float = 2.303,  # log(10) for 10 classes
    H_D: float = 1.792,  # log(6) for 6 domains
    synergy_weight: float = 0.2,
    residual_weight: float = 0.5,
    purity_scale: float = 1.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute Disentanglement Quality Score (DQS).

    The DQS combines:
    1. F1(capture, purity) for z_y - must capture class info without domain leakage
    2. F1(capture, purity) for z_d - must capture domain info without class leakage
    3. Synergy bonus for z_dy - reward capturing interaction information
    4. Residual penalty for z_x - penalize capturing Y/D info in residual

    Args:
        metrics: Dictionary containing IT metrics:
            - 'I(z_y;Y|D)': Class info in z_y (conditional on domain)
            - 'I(z_y;D|Y)': Domain leakage in z_y
            - 'I(z_d;D|Y)': Domain info in z_d (conditional on class)
            - 'I(z_d;Y|D)': Class leakage in z_d
            - 'I(z_dy;Y;D)': Synergy/interaction info (optional)
            - 'I(z_x;Y,D)': Y/D info in residual (optional)
        H_Y: Entropy of class labels H(Y) ≈ log(num_classes)
        H_D: Entropy of domain labels H(D) ≈ log(num_domains)
        synergy_weight: Weight for synergy bonus [0, 1]
        residual_weight: Weight for residual penalty [0, 1]
        purity_scale: Scale factor for purity computation
        verbose: Print detailed breakdown

    Returns:
        Dictionary with DQS and component scores
    """
    # Extract metrics (with defaults for missing values)
    I_zy_Y_given_D = metrics.get('I(z_y;Y|D)', 0)
    I_zy_D_given_Y = metrics.get('I(z_y;D|Y)', 0)
    I_zd_D_given_Y = metrics.get('I(z_d;D|Y)', 0)
    I_zd_Y_given_D = metrics.get('I(z_d;Y|D)', 0)
    I_zdy_Y_D = metrics.get('I(z_dy;Y;D)', 0)
    I_zx_YD = metrics.get('I(z_x;Y,D)', 0)

    # === Z_Y SCORE ===
    # Capture: How much class information did z_y capture?
    capture_y = compute_capture_efficiency(I_zy_Y_given_D, H_Y)

    # Purity: How little domain information leaked into z_y?
    purity_y = compute_purity(I_zy_D_given_Y, purity_scale)

    # F1 combination: Requires BOTH capture AND purity
    f1_y = harmonic_mean(capture_y, purity_y)

    # === Z_D SCORE ===
    # Capture: How much domain information did z_d capture?
    capture_d = compute_capture_efficiency(I_zd_D_given_Y, H_D)

    # Purity: How little class information leaked into z_d?
    purity_d = compute_purity(I_zd_Y_given_D, purity_scale)

    # F1 combination
    f1_d = harmonic_mean(capture_d, purity_d)

    # === BASE SCORE ===
    # Average of z_y and z_d F1 scores
    base_score = (f1_y + f1_d) / 2

    # === SYNERGY COMPONENT ===
    # Bonus for capturing interaction information in z_dy
    # Normalized by geometric mean of entropies
    max_synergy = np.sqrt(H_Y * H_D)
    synergy_raw = max(0, I_zdy_Y_D)  # Only positive synergy counts
    synergy_score = synergy_raw / max_synergy

    # === RESIDUAL COMPONENT ===
    # Penalty for capturing Y/D information in z_x (should be pure residual)
    max_residual = H_Y + H_D
    residual_score = I_zx_YD / max_residual if max_residual > 0 else 0

    # === FINAL DQS ===
    # Apply synergy bonus and residual penalty
    dqs = base_score * (1 + synergy_weight * synergy_score) * (1 - residual_weight * residual_score)
    dqs = max(0, min(1, dqs))  # Clamp to [0, 1]

    result = {
        'dqs': dqs,
        'base_score': base_score,
        # Z_y components
        'capture_y': capture_y,
        'purity_y': purity_y,
        'f1_y': f1_y,
        # Z_d components
        'capture_d': capture_d,
        'purity_d': purity_d,
        'f1_d': f1_d,
        # Modifiers
        'synergy_score': synergy_score,
        'residual_score': residual_score,
    }

    if verbose:
        print("\n" + "="*60)
        print("DISENTANGLEMENT QUALITY SCORE (DQS) BREAKDOWN")
        print("="*60)
        print(f"\nZ_Y (Class Latent):")
        print(f"  Capture: {capture_y:.3f} (I(z_y;Y|D)={I_zy_Y_given_D:.3f} / H(Y)={H_Y:.3f})")
        print(f"  Purity:  {purity_y:.3f} (leakage I(z_y;D|Y)={I_zy_D_given_Y:.3f})")
        print(f"  F1:      {f1_y:.3f}")
        print(f"\nZ_D (Domain Latent):")
        print(f"  Capture: {capture_d:.3f} (I(z_d;D|Y)={I_zd_D_given_Y:.3f} / H(D)={H_D:.3f})")
        print(f"  Purity:  {purity_d:.3f} (leakage I(z_d;Y|D)={I_zd_Y_given_D:.3f})")
        print(f"  F1:      {f1_d:.3f}")
        print(f"\nBase Score: {base_score:.3f} (average of F1_y and F1_d)")
        print(f"\nModifiers:")
        print(f"  Synergy bonus: +{synergy_weight * synergy_score * 100:.1f}% (I(z_dy;Y;D)={I_zdy_Y_D:.3f})")
        print(f"  Residual penalty: -{residual_weight * residual_score * 100:.1f}% (I(z_x;Y,D)={I_zx_YD:.3f})")
        print(f"\n{'='*60}")
        print(f"FINAL DQS: {dqs:.3f}")
        print("="*60)

    return result


def compare_models(model_metrics: Dict[str, Dict[str, float]], **kwargs) -> Dict[str, Dict]:
    """
    Compare multiple models using DQS.

    Args:
        model_metrics: Dict mapping model names to their IT metrics
        **kwargs: Additional arguments passed to compute_dqs

    Returns:
        Dict with DQS results for each model, sorted by DQS
    """
    results = {}
    for model_name, metrics in model_metrics.items():
        results[model_name] = compute_dqs(metrics, **kwargs)

    # Sort by DQS descending
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['dqs'], reverse=True)

    print("\n" + "="*70)
    print("MODEL COMPARISON BY DISENTANGLEMENT QUALITY SCORE (DQS)")
    print("="*70)
    print(f"\n{'Model':<30} {'DQS':>8} {'F1_y':>8} {'F1_d':>8} {'Cap_y':>8} {'Cap_d':>8}")
    print("-"*70)
    for rank, model in enumerate(sorted_models, 1):
        r = results[model]
        print(f"{rank}. {model:<27} {r['dqs']:>8.3f} {r['f1_y']:>8.3f} {r['f1_d']:>8.3f} "
              f"{r['capture_y']:>8.3f} {r['capture_d']:>8.3f}")

    return {model: results[model] for model in sorted_models}


# Example usage and verification
if __name__ == '__main__':
    # Test with actual model results

    # DANN_AUG #1 (collapsed z_d)
    dann_aug_1 = {
        'I(z_y;Y|D)': 0.41,
        'I(z_y;D|Y)': 0.0,
        'I(z_d;D|Y)': 0.0,
        'I(z_d;Y|D)': 0.0,
        'I(z_dy;Y;D)': -0.0003,
        'I(z_x;Y,D)': 0.0,
    }

    # NVAE #1 (balanced)
    nvae_1 = {
        'I(z_y;Y|D)': 2.29,
        'I(z_y;D|Y)': 0.22,
        'I(z_d;D|Y)': 2.46,
        'I(z_d;Y|D)': 0.06,
        'I(z_dy;Y;D)': 0.05,
        'I(z_x;Y,D)': 0.17,
    }

    # DIVA #1 (collapsed z_d)
    diva_1 = {
        'I(z_y;Y|D)': 3.04,
        'I(z_y;D|Y)': 0.23,
        'I(z_d;D|Y)': 0.002,
        'I(z_d;Y|D)': 0.0,
        'I(z_dy;Y;D)': 0.0,
        'I(z_x;Y,D)': 0.11,
    }

    print("\n" + "#"*70)
    print("# DANN_AUG #1 (High partition quality but collapsed)")
    print("#"*70)
    compute_dqs(dann_aug_1, verbose=True)

    print("\n" + "#"*70)
    print("# NVAE #1 (Balanced - captures both class and domain)")
    print("#"*70)
    compute_dqs(nvae_1, verbose=True)

    print("\n" + "#"*70)
    print("# DIVA #1 (Good class capture, collapsed z_d)")
    print("#"*70)
    compute_dqs(diva_1, verbose=True)

    # Compare all
    print("\n")
    compare_models({
        'DANN_AUG #1': dann_aug_1,
        'NVAE #1': nvae_1,
        'DIVA #1': diva_1,
    })

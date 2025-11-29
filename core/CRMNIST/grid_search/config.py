"""
Hyperparameter configuration presets for CRMNIST grid search.

This module defines grouped hyperparameter presets to make the grid search
more tractable. Instead of searching individual hyperparameters, we search
combinations of related hyperparameter groups.
"""

from typing import Dict, List, Any, Iterator
import copy

# =============================================================================
# SPARSITY PRESETS (L1 penalties for NVAE/DIVA)
# =============================================================================
# Rationalized intervals:
#   zy/zd: step=10 (0, 10, 20, 30, 40)
#   zx: step=5, ratio=0.5×zy (0, 5, 10, 15, 20)
#   zdy: step=20, ratio=2×zy (0, 20, 40, 60, 80)
SPARSITY_PRESETS = {
    'none': {
        'l1_lambda_zy': 0.0,
        'l1_lambda_zx': 0.0,
        'l1_lambda_zdy': 0.0,
        'l1_lambda_zd': 0.0,
    },
    # Only z_dy penalty - encourages z_dy selectivity without affecting main latents
    'zdy_light': {
        'l1_lambda_zy': 0.0,
        'l1_lambda_zx': 0.0,
        'l1_lambda_zdy': 20.0,
        'l1_lambda_zd': 0.0,
    },
    'low': {
        'l1_lambda_zy': 10.0,
        'l1_lambda_zx': 5.0,
        'l1_lambda_zdy': 20.0,
        'l1_lambda_zd': 10.0,
    },
    'medium': {
        'l1_lambda_zy': 20.0,
        'l1_lambda_zx': 10.0,
        'l1_lambda_zdy': 40.0,
        'l1_lambda_zd': 20.0,
    },
    'high': {
        'l1_lambda_zy': 30.0,
        'l1_lambda_zx': 15.0,
        'l1_lambda_zdy': 60.0,
        'l1_lambda_zd': 30.0,
    },
    'very_high': {
        'l1_lambda_zy': 40.0,
        'l1_lambda_zx': 20.0,
        'l1_lambda_zdy': 80.0,
        'l1_lambda_zd': 40.0,
    },
}

# =============================================================================
# CLASSIFIER PRESETS (Alpha weights for auxiliary classifiers)
# =============================================================================
# Rationalized intervals: geometric ~×2.5 progression (10, 25, 50, 100)
# Grid search showed classifier=low (25) achieved best DQS
CLASSIFIER_PRESETS = {
    'very_low': {
        'alpha_y': 10.0,  # class prediction weight
        'alpha_d': 10.0,  # domain prediction weight
    },
    'low': {
        'alpha_y': 25.0,
        'alpha_d': 25.0,
    },
    'medium': {
        'alpha_y': 50.0,
        'alpha_d': 50.0,
    },
    'high': {
        'alpha_y': 100.0,
        'alpha_d': 100.0,
    },
}

# =============================================================================
# KL DIVERGENCE PRESETS (Beta weights for VAE regularization)
# =============================================================================
# Rationalized intervals: 1, 5, 10
KL_PRESETS = {
    'low': {
        'beta_zy': 1.0,   # zy KL weight (class-specific)
        'beta_zx': 1.0,   # zx KL weight (residual)
        'beta_zdy': 1.0,  # zdy KL weight (domain-class interaction)
        'beta_zd': 1.0,   # zd KL weight (domain-specific)
    },
    'medium': {
        'beta_zy': 5.0,
        'beta_zx': 5.0,
        'beta_zdy': 5.0,
        'beta_zd': 5.0,
    },
    'high': {
        'beta_zy': 10.0,
        'beta_zx': 10.0,
        'beta_zdy': 10.0,
        'beta_zd': 10.0,
    },
}

# =============================================================================
# AUGMENTED DANN SPARSITY PRESETS (L1 penalties for latent spaces)
# =============================================================================
# 6 presets: 4 symmetric (none/low/medium/high) + 2 asymmetric (balanced/zdy_focus)
# Rationale: symmetric range covers sparsity strength, asymmetric tests which latent to penalize
DANN_AUG_SPARSITY_PRESETS = {
    # --- Symmetric presets (zy = zd) ---
    'none': {
        'sparsity_weight_zdy': 0.0,
        'sparsity_weight_zy': 0.0,
        'sparsity_weight_zd': 0.0,
    },
    'low': {
        'sparsity_weight_zdy': 1.0,
        'sparsity_weight_zy': 0.5,
        'sparsity_weight_zd': 0.5,
    },
    'medium': {
        'sparsity_weight_zdy': 2.0,
        'sparsity_weight_zy': 1.0,
        'sparsity_weight_zd': 1.0,
    },
    'high': {
        'sparsity_weight_zdy': 4.0,
        'sparsity_weight_zy': 2.0,
        'sparsity_weight_zd': 2.0,
    },
    # --- Asymmetric presets ---
    'balanced': {  # Higher z_d sparsity (domain should be sparser)
        'sparsity_weight_zdy': 1.5,
        'sparsity_weight_zy': 0.5,
        'sparsity_weight_zd': 2.5,
    },
    'zdy_focus': {  # Focus sparsity on interaction latent
        'sparsity_weight_zdy': 8.0,
        'sparsity_weight_zy': 0.5,
        'sparsity_weight_zd': 1.0,
    },
}

# =============================================================================
# AUGMENTED DANN ADVERSARIAL PRESETS (adversarial training dynamics)
# =============================================================================
# Rationalized intervals: beta_adv step=0.25, gamma step=2.5
DANN_AUG_ADVERSARIAL_PRESETS = {
    'low': {
        'beta_adv': 0.25,              # adversarial loss weight
        'lambda_schedule_gamma': 2.5,  # ramp-up speed
    },
    'medium': {
        'beta_adv': 0.5,
        'lambda_schedule_gamma': 5.0,
    },
    'medium_fast': {
        'beta_adv': 0.5,
        'lambda_schedule_gamma': 7.5,
    },
    'high': {
        'beta_adv': 1.0,
        'lambda_schedule_gamma': 10.0,
    },
}

# =============================================================================
# IRM PRESETS (invariance penalty settings)
# =============================================================================
# Tests penalty weight and annealing schedule
# Parallel structure to DANN: baseline + 2 penalty variants + 2 anneal variants
IRM_PRESETS = {
    'baseline': {
        'irm_penalty_weight': 10.0,     # Standard penalty
        'irm_anneal_iters': 500,        # Standard annealing
    },
    'weak_penalty': {
        'irm_penalty_weight': 1.0,      # Weaker invariance enforcement
        'irm_anneal_iters': 500,
    },
    'strong_penalty': {
        'irm_penalty_weight': 50.0,     # Stronger invariance enforcement
        'irm_anneal_iters': 500,
    },
    'early_anneal': {
        'irm_penalty_weight': 10.0,
        'irm_anneal_iters': 250,        # Earlier penalty activation
    },
    'late_anneal': {
        'irm_penalty_weight': 10.0,
        'irm_anneal_iters': 1000,       # Later penalty activation
    },
}

# =============================================================================
# DANN PRESETS (adversarial domain adaptation)
# =============================================================================
# Tests domain loss weight and lambda schedule speed
DANN_PRESETS = {
    'baseline': {
        'dann_domain_weight': 1.0,      # Standard equal weighting
        'dann_lambda_gamma': 10.0,      # Standard DANN paper schedule
    },
    'weak_adversarial': {
        'dann_domain_weight': 0.5,      # Weaker domain confusion
        'dann_lambda_gamma': 10.0,
    },
    'strong_adversarial': {
        'dann_domain_weight': 2.0,      # Stronger domain confusion
        'dann_lambda_gamma': 10.0,
    },
    'slow_schedule': {
        'dann_domain_weight': 1.0,
        'dann_lambda_gamma': 5.0,       # Slower λ ramp-up
    },
    'fast_schedule': {
        'dann_domain_weight': 1.0,
        'dann_lambda_gamma': 15.0,      # Faster λ ramp-up
    },
}

# =============================================================================
# FIXED PARAMETERS (not varied in grid search)
# =============================================================================
# NOTE: Default values are based on grid search results from IT analysis.
# Best configurations identified:
#   - NVAE/DIVA: classifier-high, kl-low, sparsity-none (PQ ~0.51)
#   - AugmentedDANN: adversarial-high, classifier-high, sparsity-domain_focus (PQ ~0.83)
FIXED_PARAMS = {
    # Latent dimensions
    'zy_dim': 8,
    'zx_dim': 8,
    'zdy_dim': 8,
    'zd_dim': 8,
    # Training
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'patience': 5,
    # Dataset
    'rotation_step': 15,
    'intensity': 1.5,
    'intensity_decay': 1.0,
    # OOD settings
    'ood_domain_idx': 5,  # withheld domain for testing
    # Trainer parameters (needed by trainers)
    'beta_annealing': False,  # Beta annealing for KL divergence
    'beta_scale': 1.0,  # Scaling factor for beta weights (used by trainers)
    # Encoder architecture
    'separate_encoders': False,  # Use separate encoders for each latent (achieves I(Z_i; Z_j) = 0)
    # Backward compatible parameter names (some trainers/code may expect these)
    # Alpha weights - DEFAULT: high (150.0) based on grid search showing classifier-high best
    'alpha_1': 150.0,
    'alpha_2': 150.0,
    # Beta weights - DEFAULT: low (1.0) based on grid search showing kl-low best for VAEs
    'beta_1': 1.0,  # default fallback for beta_zy
    'beta_2': 1.0,  # default fallback for beta_zx
    'beta_3': 1.0,  # default fallback for beta_zdy
    'beta_4': 1.0,  # default fallback for beta_zd
}


def get_config_name(model_type: str, preset_names: Dict[str, str]) -> str:
    """Generate a unique name for a configuration."""
    parts = [model_type]
    for key, value in sorted(preset_names.items()):
        parts.append(f"{key}-{value}")
    return "_".join(parts)


def merge_configs(*configs: Dict) -> Dict:
    """Merge multiple config dictionaries."""
    result = {}
    for config in configs:
        result.update(config)
    return result


def get_nvae_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all NVAE configurations.

    Yields:
        Dict containing:
            - name: unique configuration name
            - model_type: 'nvae'
            - preset_names: which presets were used
            - params: merged hyperparameters
    """
    for sparsity_name, sparsity_params in SPARSITY_PRESETS.items():
        for classifier_name, classifier_params in CLASSIFIER_PRESETS.items():
            for kl_name, kl_params in KL_PRESETS.items():
                preset_names = {
                    'sparsity': sparsity_name,
                    'classifier': classifier_name,
                    'kl': kl_name,
                }

                params = merge_configs(
                    copy.deepcopy(FIXED_PARAMS),
                    sparsity_params,
                    classifier_params,
                    kl_params,
                )
                params['diva'] = False

                yield {
                    'name': get_config_name('nvae', preset_names),
                    'model_type': 'nvae',
                    'preset_names': preset_names,
                    'params': params,
                }


def get_diva_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all DIVA configurations.

    DIVA uses the same hyperparameter space as NVAE, but with diva=True.
    Note: DIVA doesn't have zdy, so l1_lambda_zdy and beta_zdy are ignored.
    """
    for sparsity_name, sparsity_params in SPARSITY_PRESETS.items():
        for classifier_name, classifier_params in CLASSIFIER_PRESETS.items():
            for kl_name, kl_params in KL_PRESETS.items():
                preset_names = {
                    'sparsity': sparsity_name,
                    'classifier': classifier_name,
                    'kl': kl_name,
                }

                params = merge_configs(
                    copy.deepcopy(FIXED_PARAMS),
                    sparsity_params,
                    classifier_params,
                    kl_params,
                )
                params['diva'] = True

                yield {
                    'name': get_config_name('diva', preset_names),
                    'model_type': 'diva',
                    'preset_names': preset_names,
                    'params': params,
                }


def get_dann_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all DANN configurations.

    Tests:
    - Domain loss weight: 0.5 (weak), 1.0 (baseline), 2.0 (strong)
    - Lambda schedule gamma: 5 (slow), 10 (baseline), 15 (fast)

    Total: 5 configurations
    """
    for preset_name, preset_params in DANN_PRESETS.items():
        params = merge_configs(
            copy.deepcopy(FIXED_PARAMS),
            preset_params,
        )

        yield {
            'name': get_config_name('dann', {'config': preset_name}),
            'model_type': 'dann',
            'preset_names': {'config': preset_name},
            'params': params,
        }


def get_dann_aug_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all AugmentedDANN configurations.

    Iterates over 3 independent dimensions:
    - Sparsity: 6 options (none, low, medium, high, balanced, zdy_focus)
    - Adversarial: 4 options (low, medium, medium_fast, high)
    - Classifier: 4 options (very_low, low, medium, high)

    Total: 6 × 4 × 4 = 96 configurations
    """
    for sparsity_name, sparsity_params in DANN_AUG_SPARSITY_PRESETS.items():
        for adv_name, adv_params in DANN_AUG_ADVERSARIAL_PRESETS.items():
            for classifier_name, classifier_params in CLASSIFIER_PRESETS.items():
                preset_names = {
                    'sparsity': sparsity_name,
                    'adversarial': adv_name,
                    'classifier': classifier_name,
                }

                params = merge_configs(
                    copy.deepcopy(FIXED_PARAMS),
                    sparsity_params,
                    adv_params,
                    classifier_params,
                )

                yield {
                    'name': get_config_name('dann_aug', preset_names),
                    'model_type': 'dann_augmented',
                    'preset_names': preset_names,
                    'params': params,
                }


def get_irm_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all IRM configurations.
    """
    for preset_name, preset_params in IRM_PRESETS.items():
        params = merge_configs(
            copy.deepcopy(FIXED_PARAMS),
            preset_params,
        )

        yield {
            'name': get_config_name('irm', {'penalty': preset_name}),
            'model_type': 'irm',
            'preset_names': {'penalty': preset_name},
            'params': params,
        }


def get_all_configs(models: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get all configurations for specified models.

    Args:
        models: List of model types to include. If None, includes all.
                Options: 'nvae', 'diva', 'dann', 'dann_augmented', 'irm'

    Returns:
        List of configuration dictionaries
    """
    if models is None:
        models = ['nvae', 'diva', 'dann', 'dann_augmented', 'irm']

    configs = []

    if 'nvae' in models:
        configs.extend(list(get_nvae_configs()))

    if 'diva' in models:
        configs.extend(list(get_diva_configs()))

    if 'dann' in models:
        configs.extend(list(get_dann_configs()))

    if 'dann_augmented' in models:
        configs.extend(list(get_dann_aug_configs()))

    if 'irm' in models:
        configs.extend(list(get_irm_configs()))

    return configs


def get_quick_configs(models: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get a reduced set of configurations for quick screening.

    Uses only medium/balanced presets to quickly identify if models are working.

    Args:
        models: List of model types to include. If None, includes all.

    Returns:
        List of configuration dictionaries
    """
    if models is None:
        models = ['nvae', 'diva', 'dann', 'dann_augmented', 'irm']

    configs = []

    if 'nvae' in models:
        # Just test none and medium sparsity with medium classifier/kl
        for sparsity in ['none', 'medium']:
            preset_names = {
                'sparsity': sparsity,
                'classifier': 'medium',
                'kl': 'medium',
            }
            params = merge_configs(
                copy.deepcopy(FIXED_PARAMS),
                SPARSITY_PRESETS[sparsity],
                CLASSIFIER_PRESETS['medium'],
                KL_PRESETS['medium'],
            )
            params['diva'] = False
            configs.append({
                'name': get_config_name('nvae', preset_names),
                'model_type': 'nvae',
                'preset_names': preset_names,
                'params': params,
            })

    if 'diva' in models:
        # Just test none and medium sparsity with medium classifier/kl
        for sparsity in ['none', 'medium']:
            preset_names = {
                'sparsity': sparsity,
                'classifier': 'medium',
                'kl': 'medium',
            }
            params = merge_configs(
                copy.deepcopy(FIXED_PARAMS),
                SPARSITY_PRESETS[sparsity],
                CLASSIFIER_PRESETS['medium'],
                KL_PRESETS['medium'],
            )
            params['diva'] = True
            configs.append({
                'name': get_config_name('diva', preset_names),
                'model_type': 'diva',
                'preset_names': preset_names,
                'params': params,
            })

    if 'dann' in models:
        # Just test baseline config for quick screening
        params = merge_configs(
            copy.deepcopy(FIXED_PARAMS),
            DANN_PRESETS['baseline'],
        )
        configs.append({
            'name': get_config_name('dann', {'config': 'baseline'}),
            'model_type': 'dann',
            'preset_names': {'config': 'baseline'},
            'params': params,
        })

    if 'dann_augmented' in models:
        # Test 2 sparsity levels × medium adversarial × medium classifier = 2 configs
        for sparsity in ['none', 'medium']:
            preset_names = {
                'sparsity': sparsity,
                'adversarial': 'medium',
                'classifier': 'medium',
            }
            params = merge_configs(
                copy.deepcopy(FIXED_PARAMS),
                DANN_AUG_SPARSITY_PRESETS[sparsity],
                DANN_AUG_ADVERSARIAL_PRESETS['medium'],
                CLASSIFIER_PRESETS['medium'],
            )
            configs.append({
                'name': get_config_name('dann_aug', preset_names),
                'model_type': 'dann_augmented',
                'preset_names': preset_names,
                'params': params,
            })

    if 'irm' in models:
        # Just test baseline config for quick screening
        preset_params = IRM_PRESETS['baseline']
        params = merge_configs(
            copy.deepcopy(FIXED_PARAMS),
            preset_params,
        )
        configs.append({
            'name': get_config_name('irm', {'penalty': 'baseline'}),
            'model_type': 'irm',
            'preset_names': {'penalty': 'baseline'},
            'params': params,
        })

    return configs


def print_config_summary():
    """Print a summary of all available configurations."""
    print("=" * 60)
    print("CRMNIST Grid Search Configuration Summary")
    print("=" * 60)

    print("\n1. SPARSITY PRESETS (L1 penalties):")
    for name, params in SPARSITY_PRESETS.items():
        print(f"   {name}: zy={params['l1_lambda_zy']}, zx={params['l1_lambda_zx']}, "
              f"zdy={params['l1_lambda_zdy']}, zd={params['l1_lambda_zd']}")

    print("\n2. CLASSIFIER PRESETS (Alpha weights):")
    for name, params in CLASSIFIER_PRESETS.items():
        print(f"   {name}: alpha_y={params['alpha_y']}, alpha_d={params['alpha_d']}")

    print("\n3. KL PRESETS (Beta weights):")
    for name, params in KL_PRESETS.items():
        print(f"   {name}: beta_zy={params['beta_zy']}, beta_zx={params['beta_zx']}, "
              f"beta_zdy={params['beta_zdy']}, beta_zd={params['beta_zd']}")

    print("\n4. AUGMENTED DANN SPARSITY PRESETS:")
    print("   [Symmetric presets: zy = zd | Asymmetric presets: independent zy, zd]")
    for name, params in DANN_AUG_SPARSITY_PRESETS.items():
        print(f"   {name}: zdy={params['sparsity_weight_zdy']}, "
              f"zy={params['sparsity_weight_zy']}, zd={params['sparsity_weight_zd']}")

    print("\n5. AUGMENTED DANN ADVERSARIAL PRESETS:")
    for name, params in DANN_AUG_ADVERSARIAL_PRESETS.items():
        print(f"   {name}: beta_adv={params['beta_adv']}, "
              f"lambda_schedule_gamma={params['lambda_schedule_gamma']}")

    print("\n6. DANN PRESETS:")
    for name, params in DANN_PRESETS.items():
        print(f"   {name}: domain_weight={params['dann_domain_weight']}, "
              f"lambda_gamma={params['dann_lambda_gamma']}")

    print("\n7. IRM PRESETS:")
    for name, params in IRM_PRESETS.items():
        print(f"   {name}: penalty={params['irm_penalty_weight']}, "
              f"anneal={params['irm_anneal_iters']}")

    print("\n" + "-" * 60)
    print("Configuration counts:")
    print(f"  NVAE:           {len(list(get_nvae_configs()))} configs")
    print(f"  DIVA:           {len(list(get_diva_configs()))} configs")
    print(f"  DANN:           {len(list(get_dann_configs()))} configs")
    print(f"  AugmentedDANN:  {len(list(get_dann_aug_configs()))} configs")
    print(f"  IRM:            {len(list(get_irm_configs()))} configs")
    print(f"  TOTAL:          {len(get_all_configs())} configs")
    print(f"  Quick screen:   {len(get_quick_configs())} configs")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()

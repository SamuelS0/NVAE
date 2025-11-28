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
SPARSITY_PRESETS = {
    'none': {
        'l1_lambda_zy': 0.0,
        'l1_lambda_zx': 0.0,
        'l1_lambda_zdy': 0.0,
        'l1_lambda_zd': 0.0,
    },
    'low': {
        'l1_lambda_zy': 5.0,
        'l1_lambda_zx': 2.0,
        'l1_lambda_zdy': 10.0,
        'l1_lambda_zd': 5.0,
    },
    'medium': {
        'l1_lambda_zy': 15.0,
        'l1_lambda_zx': 5.0,
        'l1_lambda_zdy': 25.0,
        'l1_lambda_zd': 15.0,
    },
    'high': {
        'l1_lambda_zy': 25.0,
        'l1_lambda_zx': 10.0,
        'l1_lambda_zdy': 40.0,
        'l1_lambda_zd': 25.0,
    },
}

# =============================================================================
# CLASSIFIER PRESETS (Alpha weights for auxiliary classifiers)
# =============================================================================
CLASSIFIER_PRESETS = {
    'low': {
        'alpha_y': 25.0,  # class prediction weight
        'alpha_d': 25.0,  # domain prediction weight
    },
    'medium': {
        'alpha_y': 75.0,
        'alpha_d': 75.0,
    },
    'high': {
        'alpha_y': 150.0,
        'alpha_d': 150.0,
    },
}

# =============================================================================
# KL DIVERGENCE PRESETS (Beta weights for VAE regularization)
# =============================================================================
KL_PRESETS = {
    'low': {
        'beta_zy': 1.0,   # zy KL weight (class-specific)
        'beta_zx': 1.0,   # zx KL weight (residual)
        'beta_zdy': 1.0,  # zdy KL weight (domain-class interaction)
        'beta_zd': 1.0,   # zd KL weight (domain-specific)
    },
    'medium': {
        'beta_zy': 10.0,
        'beta_zx': 10.0,
        'beta_zdy': 10.0,
        'beta_zd': 10.0,
    },
    'high': {
        'beta_zy': 50.0,
        'beta_zx': 50.0,
        'beta_zdy': 50.0,
        'beta_zd': 50.0,
    },
}

# =============================================================================
# AUGMENTED DANN SPARSITY PRESETS (L1 penalties for latent spaces)
# =============================================================================
# Includes both symmetric (legacy) and asymmetric (independent) sparsity configs.
# The model accepts: sparsity_weight_zdy, sparsity_weight_zy, sparsity_weight_zd
# For backward compat, sparsity_weight_zy_zd sets both zy and zd to same value.
DANN_AUG_SPARSITY_PRESETS = {
    # --- Symmetric presets (legacy behavior: zy = zd) ---
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
        'sparsity_weight_zdy': 5.0,
        'sparsity_weight_zy': 2.0,
        'sparsity_weight_zd': 2.0,
    },
    # --- Asymmetric presets (independent zy vs zd) ---
    # Hypothesis: High z_d sparsity forces domain-specific features
    'zd_high': {
        'sparsity_weight_zdy': 1.0,
        'sparsity_weight_zy': 0.5,     # Low - preserve class info
        'sparsity_weight_zd': 5.0,     # HIGH - force sparse domain features
    },
    # Hypothesis: Very high z_d, no z_y constraint
    'zd_extreme': {
        'sparsity_weight_zdy': 1.0,
        'sparsity_weight_zy': 0.0,     # None - let class be rich
        'sparsity_weight_zd': 10.0,    # EXTREME - force very sparse domain
    },
    # Hypothesis: High z_y sparsity, low z_d
    'zy_high': {
        'sparsity_weight_zdy': 1.0,
        'sparsity_weight_zy': 5.0,     # HIGH - force compact class
        'sparsity_weight_zd': 0.5,     # Low
    },
    # Hypothesis: Domain focus - no zdy, high zd
    'domain_focus': {
        'sparsity_weight_zdy': 0.0,    # None - no interaction penalty
        'sparsity_weight_zy': 0.0,     # None - let class be rich
        'sparsity_weight_zd': 8.0,     # HIGH - force domain sparsity
    },
}

# =============================================================================
# AUGMENTED DANN ADVERSARIAL PRESETS (adversarial training dynamics)
# =============================================================================
DANN_AUG_ADVERSARIAL_PRESETS = {
    'low': {
        'beta_adv': 0.2,               # adversarial loss weight
        'lambda_schedule_gamma': 3.0,  # ramp-up speed
    },
    'medium': {
        'beta_adv': 0.5,
        'lambda_schedule_gamma': 5.0,
    },
    'high': {
        'beta_adv': 1.0,
        'lambda_schedule_gamma': 10.0,
    },
}

# =============================================================================
# IRM PRESETS (invariance penalty settings)
# =============================================================================
IRM_PRESETS = {
    'low': {
        'irm_penalty_weight': 1.0,
        'irm_anneal_iters': 500,
    },
    'medium': {
        'irm_penalty_weight': 5.0,
        'irm_anneal_iters': 1000,
    },
    'high': {
        'irm_penalty_weight': 20.0,
        'irm_anneal_iters': 2000,
    },
}

# =============================================================================
# FIXED PARAMETERS (not varied in grid search)
# =============================================================================
FIXED_PARAMS = {
    # Latent dimensions
    'zy_dim': 8,
    'zx_dim': 8,
    'zdy_dim': 8,
    'zd_dim': 8,
    # Training
    'epochs': 5,
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
    # Backward compatible parameter names (some trainers/code may expect these)
    # Alpha weights (fallbacks for new names alpha_y, alpha_d)
    'alpha_1': 75.0,
    'alpha_2': 75.0,
    # Beta weights (fallbacks for new names beta_zy, beta_zx, beta_zdy, beta_zd)
    # These will be overwritten by KL_PRESETS but provide defaults
    'beta_1': 10.0,  # default fallback for beta_zy
    'beta_2': 10.0,  # default fallback for beta_zx
    'beta_3': 10.0,  # default fallback for beta_zdy
    'beta_4': 10.0,  # default fallback for beta_zd
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


def get_dann_config() -> Dict[str, Any]:
    """
    Get DANN baseline configuration (single config, no hyperparameter variations).
    """
    params = copy.deepcopy(FIXED_PARAMS)

    return {
        'name': 'dann_baseline',
        'model_type': 'dann',
        'preset_names': {'config': 'baseline'},
        'params': params,
    }


def get_dann_aug_configs() -> Iterator[Dict[str, Any]]:
    """
    Generate all AugmentedDANN configurations.

    Iterates over 3 independent dimensions:
    - Sparsity: 8 options (none, low, medium, high + zd_high, zd_extreme, zy_high, domain_focus)
    - Adversarial: 3 options (low, medium, high)
    - Classifier: 3 options (low, medium, high)

    Total: 8 × 3 × 3 = 72 configurations

    Note: Sparsity presets now include both symmetric (zy=zd) and asymmetric
    (independent zy, zd) options. The model handles both via sparsity_weight_zy
    and sparsity_weight_zd parameters.
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
        configs.append(get_dann_config())

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
        configs.append(get_dann_config())

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
        # Just test medium penalty
        preset_params = IRM_PRESETS['medium']
        params = merge_configs(
            copy.deepcopy(FIXED_PARAMS),
            preset_params,
        )
        configs.append({
            'name': get_config_name('irm', {'penalty': 'medium'}),
            'model_type': 'irm',
            'preset_names': {'penalty': 'medium'},
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

    print("\n6. IRM PRESETS:")
    for name, params in IRM_PRESETS.items():
        print(f"   {name}: penalty={params['irm_penalty_weight']}, "
              f"anneal={params['irm_anneal_iters']}")

    print("\n" + "-" * 60)
    print("Configuration counts:")
    print(f"  NVAE:           {len(list(get_nvae_configs()))} configs")
    print(f"  DIVA:           {len(list(get_diva_configs()))} configs")
    print(f"  DANN:           1 config (baseline)")
    print(f"  AugmentedDANN:  {len(list(get_dann_aug_configs()))} configs")
    print(f"  IRM:            {len(list(get_irm_configs()))} configs")
    print(f"  TOTAL:          {len(get_all_configs())} configs")
    print(f"  Quick screen:   {len(get_quick_configs())} configs")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()

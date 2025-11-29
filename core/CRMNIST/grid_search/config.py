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
    # Recommended default: light L1 on z_dy reduces I(z_y;z_dy) entanglement
    # 20-epoch IT analysis showed NVAE had I(z_y;z_dy)=1.98 with no L1
    'zdy_light': {
        'l1_lambda_zy': 0.0,
        'l1_lambda_zx': 0.0,
        'l1_lambda_zdy': 5.0,  # Light penalty to encourage z_dy selectivity
        'l1_lambda_zd': 0.0,
    },
    'low': {
        'l1_lambda_zy': 5.0,
        'l1_lambda_zx': 2.5,
        'l1_lambda_zdy': 10.0,
        'l1_lambda_zd': 5.0,
    },
    'medium': {
        'l1_lambda_zy': 15.0,
        'l1_lambda_zx': 5.0,
        'l1_lambda_zdy': 25.0,
        'l1_lambda_zd': 15.0,
    },
    # Interpolation between medium and high - added based on grid search analysis
    'medium_high': {
        'l1_lambda_zy': 20.0,
        'l1_lambda_zx': 7.5,
        'l1_lambda_zdy': 32.5,
        'l1_lambda_zd': 20.0,
    },
    'high': {
        'l1_lambda_zy': 25.0,
        'l1_lambda_zx': 10.0,
        'l1_lambda_zdy': 40.0,
        'l1_lambda_zd': 25.0,
    },
    # Higher than 'high' - safe for NVAE (0% collapse in grid search)
    # ~40% increase from 'high' preset
    'very_high': {
        'l1_lambda_zy': 35.0,
        'l1_lambda_zx': 15.0,
        'l1_lambda_zdy': 57.5,
        'l1_lambda_zd': 35.0,
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
    # Interpolation between low and medium - grid search showed medium was optimal
    # This explores the transition region
    'medium_low': {
        'beta_zy': 5.0,
        'beta_zx': 5.0,
        'beta_zdy': 5.0,
        'beta_zd': 5.0,
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
    # Interpolation between medium and high - safe based on grid search
    'medium_high': {
        'sparsity_weight_zdy': 3.5,
        'sparsity_weight_zy': 1.5,
        'sparsity_weight_zd': 1.5,
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
    # Balanced preset based on 20-epoch IT analysis - prevents dimension collapse
    # while still encouraging clean latent separation
    'balanced': {
        'sparsity_weight_zdy': 1.5,    # Moderate - prevents z_dy from becoming catch-all
        'sparsity_weight_zy': 0.5,     # Light - preserve rich class representation
        'sparsity_weight_zd': 2.5,     # Moderate - enough to encourage compactness without collapse
    },
    # Focus on z_dy sparsity with conservative zy/zd - based on grid search analysis
    # High zdy encourages selectivity in interaction latent without collapsing main latents
    'zdy_focus': {
        'sparsity_weight_zdy': 7.5,    # High - force z_dy selectivity
        'sparsity_weight_zy': 0.75,    # Between low and medium - preserve class
        'sparsity_weight_zd': 1.0,     # Same as medium - safe for domain
    },
    # ==========================================================================
    # REMOVED PRESETS - These caused total latent collapse in grid search:
    # ==========================================================================
    # 'zd_extreme' - REMOVED: Caused total collapse (DQS < 0.01) with high adversarial
    #   - adv=high + cls=high + zd_extreme: capture_y=0.0, capture_d=0.0002, DQS=0.0002
    #   - adv=high + cls=low + zd_extreme: capture_y=0.18, capture_d=0.0, DQS=0.15
    #
    # 'zy_high' - REMOVED: Caused z_y collapse (DQS < 0.01) with high adversarial
    #   - adv=high + cls=medium + zy_high: capture_y=0.0, capture_d=0.001, DQS=0.001
    #
    # 'domain_focus' - REMOVED: Caused z_d collapse (100% of models) + some total collapse
    #   - All domain_focus configs had capture_d < 0.01
    #   - adv=low + cls=high/low + domain_focus: total collapse (DQS < 0.25)
    # ==========================================================================
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
    # Same strength as medium but faster ramp-up schedule
    # Based on grid search: medium adversarial worked best with high sparsity
    'medium_fast': {
        'beta_adv': 0.5,               # Same as medium
        'lambda_schedule_gamma': 7.5,  # Between medium (5.0) and high (10.0)
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
    - Sparsity: 8 options (none, low, medium, medium_high, high, zd_high, balanced, zdy_focus)
    - Adversarial: 4 options (low, medium, medium_fast, high)
    - Classifier: 3 options (low, medium, high)

    Total: 8 × 4 × 3 = 96 configurations

    Note: Removed 3 presets that caused total latent collapse in grid search:
    - zd_extreme: caused both z_y and z_d collapse with high adversarial
    - zy_high: caused z_y collapse with high adversarial
    - domain_focus: caused 100% z_d collapse across all configs
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

# CRMNIST Model Comparison Script Usage Guide

## Overview
The updated `run_crmnist.py` script can train and visualize **all four models** (NVAE, DIVA, DANN, IRM) with CRMNIST data using the unified balanced sampling function.

## Basic Usage

### Run All Models (Default)
```bash
python -m core.CRMNIST.run_crmnist --out results/crmnist_comparison --epochs 50
```

### Run Specific Models
```bash
# Run only NVAE and DIVA
python -m core.CRMNIST.run_crmnist --models nvae diva --out results/generative_models

# Run only discriminative models
python -m core.CRMNIST.run_crmnist --models dann irm --out results/discriminative_models

# Run single model
python -m core.CRMNIST.run_crmnist --models nvae --out results/nvae_only
```

## Key Arguments

### Model Selection
- `--models`: Choose which models to run
  - Options: `nvae`, `diva`, `dann`, `irm`
  - Default: All models
  - Example: `--models nvae diva`

### Dataset Caching
- `--use_cache`: Use cached datasets (default: True)
- `--no_cache`: Force dataset regeneration

### Training Control
- `--epochs`: Number of training epochs (default: 1)
- `--skip_training`: Skip training, only do visualization

### Model Parameters
- `--zy_dim`, `--zx_dim`, `--za_dim`, `--zay_dim`: Latent dimensions (default: 32)
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size

## Output Structure

```
results/
├── models/                    # Trained model checkpoints
│   ├── nvae_model_epoch_X.pt
│   ├── diva_model_epoch_X.pt
│   ├── dann_model_epoch_X.pt
│   └── irm_model_epoch_X.pt
├── latent_space/             # Visualization files
│   ├── nvae_latent_spaces.png
│   ├── diva_latent_spaces.png
│   ├── dann_latent_spaces.png
│   └── irm_latent_spaces.png
├── reconstructions/          # (Future: reconstruction visualizations)
├── nvae_model_params.json    # Model parameters for each model
├── diva_model_params.json
├── dann_model_params.json
└── irm_model_params.json
```

## Visualization Details

### NVAE Latent Spaces (4 spaces)
- **zy**: Label-specific features (should cluster by digit)
- **za**: Domain-specific features (should cluster by rotation)
- **zay**: Domain-label interaction features
- **zx**: Residual features

### DIVA Latent Spaces (3 spaces)
- **zy**: Label-specific features
- **za**: Domain-specific features  
- **zx**: Residual features
- **No zay**: DIVA doesn't use domain-label interaction

### DANN/IRM Latent Spaces (1 shared space)
- **Features**: Single learned representation
- Used for both classification and domain adaptation

## Example Commands

### Quick Test (1 epoch)
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae diva \
    --epochs 1 \
    --out results/quick_test
```

### Full Training
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae diva dann irm \
    --epochs 100 \
    --learning_rate 1e-3 \
    --batch_size 64 \
    --out results/full_comparison
```

### Custom Latent Dimensions
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae \
    --zy_dim 64 \
    --za_dim 32 \
    --zx_dim 32 \
    --zay_dim 16 \
    --out results/custom_dims
```

### Force Dataset Regeneration
```bash
python -m core.CRMNIST.run_crmnist \
    --no_cache \
    --models dann irm \
    --out results/fresh_data
```

## Expected Output

The script will show:

1. **Dataset Loading**: Cached or generated datasets
2. **Model Training**: Progress for each selected model
3. **Model Saving**: Checkpoint locations
4. **Visualization**: Latent space plots
5. **Summary**: Training metrics and file locations

Example output:
```
🎯 Running comparison experiments...
Selected models to run: ['nvae', 'diva', 'dann', 'irm']

============================================================
🔥 TRAINING NVAE MODEL
============================================================
Starting balanced sampling for nvae model...
🎨 Generating NVAE latent space visualization...

============================================================
🔥 TRAINING DIVA MODEL  
============================================================
Model is in DIVA mode (no zay latent space)
🎨 Generating DIVA latent space visualization...

============================================================
📊 EXPERIMENT SUMMARY
============================================================
Models trained and saved:
  ✅ NVAE: VAE
  ✅ DIVA: VAE  
  ✅ DANN: DANN
  ✅ IRM: IRM

🎉 All experiments completed!
```

## Benefits

✅ **Unified Interface**: Single script for all models  
✅ **Consistent Sampling**: Same balanced sampling across all models  
✅ **Flexible Selection**: Choose which models to run  
✅ **Automatic Caching**: Fast dataset loading on subsequent runs  
✅ **Rich Visualizations**: Model-specific latent space plots  
✅ **Easy Comparison**: All results in organized output structure 
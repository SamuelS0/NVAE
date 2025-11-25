# CRMNIST Model Comparison Script - Complete Usage Guide

## Overview
The `run_crmnist.py` script is a comprehensive framework for training and evaluating **five different models** (NVAE, DIVA, DANN, AugmentedDANN, IRM) on the CRMNIST dataset. It includes extensive visualization capabilities, latent expressiveness analysis, disentanglement evaluation, and information-theoretic testing.

## Table of Contents
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Command-Line Arguments](#command-line-arguments)
- [Usage Examples](#usage-examples)
- [Output Structure](#output-structure)
- [Visualizations](#visualizations)
- [Analysis & Evaluation](#analysis--evaluation)
- [Advanced Features](#advanced-features)

---

## Quick Start

### Run All 5 Models (Default)
```bash
python -m core.CRMNIST.run_crmnist --out results/crmnist_comparison --epochs 50
```

This runs **all 5 models** by default: NVAE, DIVA, DANN, AugmentedDANN, and IRM.

### Run Specific Models Only
```bash
python -m core.CRMNIST.run_crmnist --models nvae diva --out results/vae_only --epochs 50
```

### Quick Test (1 Epoch)
```bash
python -m core.CRMNIST.run_crmnist --epochs 1 --out results/quick_test
```

---

## Available Models

### 1. NVAE (Novel VAE with 4 Latent Subspaces)
- **Latent Spaces**: `zy`, `za`, `zay`, `zx`
- **Features**: Full latent decomposition with domain-label interaction
- **Use Case**: Testing the novel zay component hypothesis

### 2. DIVA (Domain-Invariant Variational Autoencoder)
- **Latent Spaces**: `zy`, `za`, `zx` (no zay)
- **Features**: Standard disentangled VAE without interaction term
- **Use Case**: Baseline comparison for NVAE

### 3. DANN (Domain-Adversarial Neural Network)
- **Latent Spaces**: Single shared feature space
- **Features**: Gradient reversal for domain-invariant features
- **Use Case**: Discriminative baseline, domain adaptation

### 4. AugmentedDANN (Enhanced DANN with Latent Decomposition)
- **Latent Spaces**: `zy`, `zd`, `zdy` (3 subspaces)
- **Features**: DANN with explicit latent partitioning
- **Special Parameters**: `--lambda_reversal`, `--sparsity_weight`, `--beta_adv`
- **Use Case**: Bridging generative and discriminative approaches

### 5. IRM (Invariant Risk Minimization)
- **Latent Spaces**: Single shared feature space
- **Features**: Learns invariant predictors across domains using gradient-based penalty
- **Special Parameters**: `--irm_penalty_weight`, `--irm_anneal_iters`
- **Use Case**: Causal invariance baseline
- **Training Recommendation**: IRM requires **50+ epochs** for proper convergence (warm-up period needed)

---

## Command-Line Arguments

### Model Selection
```bash
--models MODELS [MODELS ...]
```
**Options**: `nvae`, `diva`, `dann`, `dann_augmented`, `irm`
**Default**: `nvae diva dann dann_augmented irm` (all 5 models)
**Example**: `--models nvae diva` (run only NVAE and DIVA)

**✅ All models run by default** - no need to specify unless you want to run a subset.

### Dataset & Caching
```bash
--config PATH              # Config file (default: conf/crmnist.json)
--use_cache                # Use cached datasets (default: True)
--no_cache                 # Force dataset regeneration
--intensity FLOAT          # Transform intensity (default: 1.5)
--intensity_decay FLOAT    # Intensity decay (default: 1.0)
```

### Training Control
```bash
--epochs INT               # Number of training epochs (default: 1)
--batch_size INT           # Batch size (default: 64)
--learning_rate FLOAT      # Learning rate (default: 0.001)
--patience INT             # Early stopping patience (default: 5)
--skip_training            # Skip training, only do visualization (requires pre-trained models)
```

### Latent Dimensions (NVAE/DIVA)
```bash
--zy_dim INT               # Label-specific latent dimension (default: 32)
--za_dim INT               # Domain-specific latent dimension (default: 32)
--zay_dim INT              # Domain-label interaction latent (default: 32)
--zx_dim INT               # Residual latent dimension (default: 32)
```

### Loss Weights (NVAE/DIVA)
```bash
--beta_1 FLOAT             # KL weight for zy (default: 1.0)
--beta_2 FLOAT             # KL weight for za (default: 1.0)
--beta_3 FLOAT             # KL weight for zay (default: 1.0)
--beta_4 FLOAT             # KL weight for zx (default: 1.0)
--alpha_1 FLOAT            # Classification weight (default: 100.0)
--alpha_2 FLOAT            # Domain classification weight (default: 100.0)
--beta_annealing           # Enable beta annealing for KL divergence
--beta_scale FLOAT         # Beta annealing scale factor (default: 1.0)
```

### L1 Sparsity Penalties (NVAE/DIVA)
```bash
--l1_lambda_zy FLOAT       # L1 penalty for zy (default: 0.0 = disabled)
--l1_lambda_za FLOAT       # L1 penalty for za (default: 0.0)
--l1_lambda_zay FLOAT      # L1 penalty for zay (default: 0.0)
--l1_lambda_zx FLOAT       # L1 penalty for zx (default: 0.0)
```

### AugmentedDANN-Specific Parameters
```bash
--lambda_reversal FLOAT    # Gradient reversal strength (default: 1.0)
--sparsity_weight FLOAT    # Sparsity penalty on zdy (default: 0.01)
--beta_adv FLOAT           # Adversarial loss weight (default: 0.1)
```

### IRM-Specific Parameters
```bash
--irm_penalty_weight FLOAT # Weight for IRM invariance penalty (default: 1e4)
--irm_anneal_iters INT     # Iterations before applying penalty (default: 500)
```
**Note**: The penalty annealing provides a warm-up period where the model first learns basic features before enforcing invariance. With 360,000 training samples and batch size 64, one epoch = ~5,625 batches. The default 500 iteration warm-up represents ~9% of the first epoch.

### Output & Organization
```bash
--out PATH                 # Output directory (required)
--setting STR              # Experimental setting name (default: standard)
--cuda                     # Enable CUDA (default: True if available)
```

---

## Usage Examples

### 1. Quick Test (1 Epoch, All Default Models)
```bash
python -m core.CRMNIST.run_crmnist \
    --epochs 1 \
    --out results/quick_test
```

### 2. Run All 5 Models (Default Behavior)
```bash
python -m core.CRMNIST.run_crmnist \
    --epochs 50 \
    --out results/all_five_models
```
All 5 models run by default - no need to specify `--models`

### 3. Run Only Generative Models (NVAE + DIVA)
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae diva \
    --epochs 100 \
    --out results/generative_only
```

### 4. Run Only Discriminative Models
```bash
python -m core.CRMNIST.run_crmnist \
    --models dann dann_augmented irm \
    --epochs 50 \
    --out results/discriminative_only
```

### 5. Full Training with Custom Dimensions
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae diva \
    --zy_dim 64 \
    --za_dim 32 \
    --zx_dim 32 \
    --zay_dim 16 \
    --epochs 100 \
    --out results/custom_dims
```

### 6. Training with L1 Sparsity Regularization
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae \
    --l1_lambda_zy 0.01 \
    --l1_lambda_zay 0.02 \
    --epochs 100 \
    --out results/sparse_latents
```

### 7. Training with Beta Annealing
```bash
python -m core.CRMNIST.run_crmnist \
    --models nvae diva \
    --beta_annealing \
    --beta_scale 0.5 \
    --epochs 100 \
    --out results/beta_annealing
```

### 8. AugmentedDANN with Custom Parameters
```bash
python -m core.CRMNIST.run_crmnist \
    --models dann_augmented \
    --lambda_reversal 0.5 \
    --sparsity_weight 0.05 \
    --beta_adv 0.2 \
    --epochs 100 \
    --out results/augmented_dann_custom
```

### 9. Force Dataset Regeneration (No Cache)
```bash
python -m core.CRMNIST.run_crmnist \
    --no_cache \
    --models nvae diva \
    --out results/fresh_data
```

### 10. Visualization Only (Skip Training)
```bash
python -m core.CRMNIST.run_crmnist \
    --skip_training \
    --models nvae diva dann irm \
    --out results/existing_experiment
```
**Note**: Requires pre-trained models in `results/existing_experiment/models/`

---

## Output Structure

```
results/
├── comparison_models/                   # Saved model checkpoints with metadata
│   ├── alpha1-100.0_alpha2-100.0_..._final.pt
│   └── ...
├── models/                              # Model checkpoints by epoch
│   ├── nvae_model_epoch_X.pt
│   ├── diva_model_epoch_X.pt
│   ├── dann_model_epoch_X.pt
│   ├── dann_augmented_model_epoch_X.pt
│   └── irm_model_epoch_X.pt
├── latent_space/                        # Main latent space visualizations
│   ├── nvae_latent_spaces.png          # t-SNE plots of 4 latent subspaces
│   ├── diva_latent_spaces.png          # t-SNE plots of 3 latent subspaces
│   ├── dann_latent_spaces.png          # t-SNE plot of single feature space
│   ├── dann_augmented_latent_spaces.png # t-SNE plots of 3 latent subspaces
│   └── irm_latent_spaces.png           # t-SNE plot of single feature space
├── latent_epoch_viz/                    # Per-epoch latent visualizations
│   └── crmnist_latent_epoch_*.png
├── nvae_expressiveness/                 # NVAE latent expressiveness analysis
│   ├── latent_expressiveness_comparison.png
│   └── latent_expressiveness_results.json
├── diva_expressiveness/                 # DIVA latent expressiveness analysis
│   └── latent_expressiveness_results.json
├── nvae_disentanglement/                # NVAE disentanglement visualizations
│   ├── disentanglement.png             # Factor disentanglement
│   ├── interpolation.png               # Latent space interpolations
│   └── traversal.png                   # Factor traversal
├── diva_disentanglement/                # DIVA disentanglement visualizations
│   ├── disentanglement.png
│   ├── interpolation.png
│   └── traversal.png
├── information_theoretic_analysis/      # MIP theorem adherence testing
│   ├── nvae/
│   │   └── it_results.json
│   ├── diva/
│   │   └── it_results.json
│   ├── model_comparison.json
│   └── [various IT visualization plots]
├── reconstructions/                     # Image reconstructions
├── domain_samples/                      # Per-domain sample visualizations
├── training_history.json                # Complete training metrics
├── training_history.csv                 # Training metrics in CSV format
├── training_curves.png                  # Loss and accuracy curves
├── comprehensive_expressiveness_comparison.png
├── expressiveness_analysis_report.txt   # Summary report
└── {model}_model_params.json            # Model hyperparameters
```

---

## Visualizations

### 1. Latent Space Visualizations (All Models)
**Files**: `latent_space/{model}_latent_spaces.png`

#### NVAE (4 Subspaces)
- **zy**: Label-specific features (should cluster by digit 0-9)
- **za**: Domain-specific features (should cluster by rotation angle)
- **zay**: Domain-label interaction features (captures shared information)
- **zx**: Residual features

#### DIVA (3 Subspaces)
- **zy**: Label-specific features
- **za**: Domain-specific features
- **zx**: Residual features
- **No zay**: DIVA doesn't model domain-label interaction

#### AugmentedDANN (3 Subspaces)
- **zy**: Label-specific features
- **zd**: Domain-specific features
- **zdy**: Domain-label interaction features

#### DANN/IRM (1 Shared Space)
- **Features**: Single learned representation
- Used for both classification and domain adaptation

### 2. Latent Expressiveness Analysis (NVAE/DIVA)
**Files**: `{model}_expressiveness/`

Tests how well each latent subspace encodes the intended information:
- **Domain Classification**: Can za (+ zay) predict the domain?
- **Label Classification**: Can zy (+ zay) predict the label?
- **Comparison Plots**: Bar charts showing accuracy improvements
- **JSON Results**: Detailed metrics with train/val/test splits

**Key Finding** (NVAE): Adding zay improves:
- Domain classification by ~22%
- Label classification by ~20%

### 3. Disentanglement Visualizations (NVAE/DIVA)
**Files**: `{model}_disentanglement/`

#### a. Factor Disentanglement (`disentanglement.png`)
- Shows how varying each latent dimension affects reconstructed images
- Demonstrates which dimensions control which factors
- 3 example images × 4 latent spaces × 7 variations each

#### b. Latent Interpolation (`interpolation.png`)
- Smooth interpolations between pairs of images in each latent subspace
- Shows what each latent space learns to encode
- 7 interpolation steps per latent space

#### c. Factor Traversal (`traversal.png`)
- Systematic traversal of latent dimensions
- Shows independent control of factors
- Separate plots for zy, za, zay, zx (NVAE) or zy, za, zx (DIVA)

### 4. Training Curves
**Files**: `training_curves.png`, `training_history.json/csv`

- Loss curves (train & validation)
- Accuracy curves (label & domain)
- Per-epoch metrics for all models
- Available in JSON, CSV, and plot formats

### 5. Comprehensive Expressiveness Comparison
**Files**: `comprehensive_expressiveness_comparison.png`, `expressiveness_analysis_report.txt`

- Side-by-side comparison of all trained models
- Highlights relative performance on domain/label classification
- Summary report with key findings

---

## Analysis & Evaluation

### 1. Latent Expressiveness Evaluation
**What it does**: Trains linear classifiers on latent representations to measure information content

**For NVAE**:
- Tests if za alone can predict domain (baseline)
- Tests if za+zay can predict domain (with interaction)
- Tests if zay alone can predict domain
- Same experiments for label prediction with zy

**Outputs**:
- Accuracy metrics (train/val/test) for each experiment
- Comparison plot showing improvements
- JSON file with complete results

**When it runs**: Automatically after training NVAE or DIVA models

### 2. Disentanglement Analysis
**What it does**: Visualizes how latent dimensions control generative factors

**Visualizations**:
1. **Disentanglement**: Vary one latent dimension at a time
2. **Interpolation**: Smooth transitions in latent space
3. **Traversal**: Systematic exploration of latent dimensions

**When it runs**: Automatically after training NVAE or DIVA models

### 3. Information-Theoretic Evaluation
**What it does**: Tests adherence to the Minimal Information Partition (MIP) theorem

**Metrics Computed**:
- I(zy; y): Mutual information between label latent and labels
- I(za; d): Mutual information between domain latent and domains
- I(zy; d): Unwanted label-domain information leakage
- I(za; y): Unwanted domain-label information leakage
- Conditional independence tests
- Bootstrap confidence intervals (10 iterations, optimized for speed)

**Performance Optimizations**:
- Sample size: ~3,200 samples (50 batches)
- Bootstrap iterations: 10 (provides statistically valid confidence intervals)
- Evaluation time: ~30-60 seconds per model (40x faster than original settings)

**Supported Models**: NVAE, DIVA (requires VAE-style latent decomposition)

**Outputs**:
- Per-model IT results in JSON format
- Model comparison with partition quality scores
- Multiple visualization plots
- Rankings by MIP adherence

**When it runs**: Automatically after training (if NPEET is installed)

**Requirements**:
```bash
pip install git+https://github.com/gregversteeg/NPEET.git
```

### 4. Comprehensive Model Comparison
**What it does**: Aggregates expressiveness results across all models

**Outputs**:
- Comparative bar charts
- Statistical summary report
- Best model identification

**When it runs**: Automatically after all models finish training

---

## Advanced Features

### 1. Beta Annealing
Gradually increase KL divergence weight during training to improve optimization:

```bash
--beta_annealing --beta_scale 0.5
```

**Use Case**: Preventing posterior collapse in VAE training

### 2. L1 Sparsity Regularization
Add L1 penalties to promote sparse latent representations:

```bash
--l1_lambda_zy 0.01 --l1_lambda_zay 0.02
```

**Use Case**: Encouraging interpretability and reducing redundancy

### 3. Dataset Caching
**Default**: Enabled (`--use_cache`)
- First run generates and caches dataset
- Subsequent runs load from cache (much faster)

**Disable Caching**:
```bash
--no_cache
```

**Cache Location**: `./data/crmnist/cache/`

**Cache Key**: MD5 hash of dataset configuration
- Different configs automatically create separate caches
- Changing rotation, color, or domain settings creates new cache

### 4. Skip Training Mode
Run visualization and analysis on pre-trained models:

```bash
--skip_training --out path/to/existing/experiment
```

**Requirements**:
- Pre-trained model checkpoints in `path/to/existing/experiment/models/`
- Model files must match expected naming: `{model}_model_epoch_*.pt`

**Use Case**: Re-generate visualizations or test new analysis methods

### 5. Custom Experimental Settings
Organize multiple experiments:

```bash
--setting ablation_study_1
```

**Use Case**: Systematic hyperparameter sweeps, ablation studies

---

## Model-Specific Details

### NVAE vs DIVA
| Feature | NVAE | DIVA |
|---------|------|------|
| Latent Subspaces | 4 (zy, za, zay, zx) | 3 (zy, za, zx) |
| Domain-Label Interaction | Yes (zay) | No |
| Total Latent Dims (default) | 128 | 128 |
| Disentanglement | Tested | Tested |
| IT Evaluation | Supported | Supported |

### DANN vs AugmentedDANN
| Feature | DANN | AugmentedDANN |
|---------|------|---------------|
| Architecture | Discriminative | Discriminative |
| Latent Decomposition | No (single space) | Yes (3 subspaces) |
| Gradient Reversal | Yes | Yes |
| Sparsity Penalty | No | Yes (on zdy) |
| Special Parameters | None | lambda_reversal, sparsity_weight, beta_adv |

### Dimension Redistribution (AugmentedDANN)
When using AugmentedDANN with other models:
- Other models: zy(32) + za(32) + zay(32) + zx(32) = 128 total
- AugmentedDANN redistributes across 3 spaces:
  - zy: 43 dims
  - zd: 43 dims
  - zdy: 42 dims
  - Total: 128 dims (fair comparison)

---

## Expected Output & Results

### Training Progress
```
🎯 Running comparison experiments...
Selected models to run: ['nvae', 'diva', 'dann', 'irm']

============================================================
🔥 TRAINING NVAE MODEL
============================================================
Epoch 1/50:
  Train Loss: 8681.13
  Val Loss: 5023.98
  Val y_accuracy: 0.9761
  Val a_accuracy: 0.9915

🎨 Generating NVAE latent space visualization...
🧪 Evaluating NVAE latent expressiveness...
🎨 Generating NVAE disentanglement visualizations...
   ✅ Disentanglement visualization saved
   ✅ Interpolation visualization saved
   ✅ Factor traversal visualization saved

============================================================
🔥 TRAINING DIVA MODEL
============================================================
[Similar output for DIVA...]

============================================================
📊 COMPREHENSIVE EXPRESSIVENESS ANALYSIS
============================================================
[Comparison across models...]

============================================================
🧮 INFORMATION-THEORETIC EVALUATION
============================================================
[MIP theorem testing...]

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

### Typical Performance (After Training)
**NVAE**:
- Validation accuracy (labels): ~97-98%
- Validation accuracy (domains): ~99%
- zay improves za domain classification by ~22%
- zay improves zy label classification by ~20%

**DIVA**:
- Validation accuracy (labels): ~97-98%
- Validation accuracy (domains): ~99%
- Strong baseline without interaction term

**DANN**:
- Validation accuracy (labels): ~98%
- Domain discriminator accuracy: ~50-70% (desired: ~50% for invariance)

**IRM**:
- Variable performance depending on penalty weight
- Learns domain-invariant predictors

---

## Frequently Asked Questions

### Q: What are the color labels in CRMNIST? Are they used for training?
**A**: CRMNIST datasets contain 4 types of information per sample:
- **Images (x)**: 28×28 RGB images of colored, rotated digits
- **Digit labels (y)**: The digit class (0-9) - this is what models predict
- **Color labels (c)**: The color applied to the digit (7 colors total)
- **Domain/Rotation labels (d)**: The rotation angle (0°, 10°, 20°, 30°, 40°, 50°)

**Color labels are NOT used for model training**. They exist for:
- Dataset generation (controlling which colors are applied)
- Visualization and analysis (checking if models rely on spurious color correlations)
- Quality control (verifying dataset composition)

**What models actually use**:
- All models receive: `(x, y, d)` = (images, digit labels, domain labels)
- Color information is embedded in the images themselves
- Models learn to handle color through the pixel values, not as a separate supervision signal
- The domain label `d` represents rotation angle, which is the primary domain identifier

**Relationship between color and rotation**:
- Each rotation domain is assigned a primary color (e.g., 0° → Blue, 10° → Green)
- Some digits get colored "red" across all domains (spurious correlation)
- This creates a rich multi-domain structure for testing invariance

### Q: Why does IRM need so much more training than other models?
**A**: IRM uses a two-phase learning strategy:
1. **Warm-up phase** (first 500 iterations): Learns basic features without penalty
2. **Invariance phase** (remaining training): Applies gradient-based invariance penalty

The penalty forces the model to find representations that work equally well across all domains. This optimization is more challenging than standard ERM (Empirical Risk Minimization) and requires:
- More epochs to converge (50+ recommended vs. 20-30 for other models)
- Careful hyperparameter tuning (`--irm_penalty_weight`, `--irm_anneal_iters`)
- Sufficient data per domain to compute meaningful penalties

For quick testing, use DANN instead - it achieves similar domain invariance with faster convergence.

---

## Troubleshooting

### Issue: "IRM model has low accuracy (<20%)"
**Solution**:
- IRM requires significantly more training than other models (50+ epochs recommended)
- Try reducing penalty weight: `--irm_penalty_weight 1e3` or `1e2`
- Adjust warm-up period: `--irm_anneal_iters 100` for faster penalty application
- Single-epoch runs are insufficient for IRM convergence

### Issue: "No model checkpoint found"
**Solution**:
- When using `--skip_training`, ensure models exist in `{output}/models/`
- Check model naming: `nvae_model_epoch_*.pt` or `{model}_model.pt`

### Issue: Information-theoretic evaluation skipped
**Solution**:
```bash
pip install git+https://github.com/gregversteeg/NPEET.git
```

### Issue: Information-theoretic evaluation too slow
**Current Settings** (optimized):
- Sample size: ~3,200 samples (50 batches)
- Bootstrap: 10 iterations
- Time: ~30-60 seconds per model

**If still too slow**, modify `core/CRMNIST/run_crmnist.py` line 747-748:
```python
max_batches=20,      # Even faster: ~1,280 samples
n_bootstrap=5        # Minimal CI: 5 iterations
```

**If you need publication-quality results**, use:
```python
max_batches=100,     # ~6,400 samples
n_bootstrap=20       # More robust CIs
```

### Issue: CUDA out of memory
**Solutions**:
- Reduce `--batch_size` (try 32 or 16)
- Reduce latent dimensions: `--zy_dim 16 --za_dim 16 --zx_dim 16 --zay_dim 16`
- Use CPU: Remove `--cuda` flag

### Issue: Slow dataset generation
**Solution**:
- First run generates dataset (slow)
- Subsequent runs use cache (fast)
- Use `--use_cache` (enabled by default)

### Issue: Want to regenerate dataset with same config
**Solution**:
```bash
--no_cache
```

---

## Complete Example: Full Research Workflow

```bash
# 1. Quick test to verify setup (1 epoch)
python -m core.CRMNIST.run_crmnist \
    --epochs 1 \
    --out results/test_run

# 2. Full training of all 5 models with custom hyperparameters
python -m core.CRMNIST.run_crmnist \
    --models nvae diva dann dann_augmented irm \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --zy_dim 64 --za_dim 32 --zx_dim 32 --zay_dim 16 \
    --beta_annealing \
    --l1_lambda_zay 0.01 \
    --out results/full_experiment

# 3. Ablation study: NVAE without zay component (equivalent to DIVA)
python -m core.CRMNIST.run_crmnist \
    --models diva \
    --epochs 100 \
    --out results/ablation_no_zay

# 4. Re-generate visualizations from trained models
python -m core.CRMNIST.run_crmnist \
    --skip_training \
    --models nvae diva dann dann_augmented irm \
    --out results/full_experiment

# Results structure:
# results/
# ├── test_run/                    # Quick verification
# ├── full_experiment/              # Main results with all analyses
# └── ablation_no_zay/              # Ablation study
```

---

## Summary of Key Points

### Models
✅ **5 models available**: NVAE, DIVA, DANN, AugmentedDANN, IRM
✅ **All 5 run by default**: `nvae diva dann dann_augmented irm`
🔧 **To run specific models**: Use `--models nvae diva` (example: only VAE models)

### Visualizations
✅ Latent space t-SNE plots (all models)
✅ Latent expressiveness analysis (NVAE, DIVA)
✅ Disentanglement visualizations (NVAE, DIVA)
✅ Interpolation & traversal (NVAE, DIVA)
✅ Training curves & metrics (all models)
✅ Comprehensive model comparison
✅ Information-theoretic evaluation (NVAE, DIVA)

### Features
✅ Dataset caching for speed
✅ Skip training mode for visualization-only
✅ L1 sparsity regularization
✅ Beta annealing for VAEs
✅ Extensive hyperparameter control
✅ Automatic analysis pipeline
✅ MIP theorem testing with NPEET

---

## Citation & References

If using this framework, please cite the relevant papers:
- **NVAE**: Novel architecture with zay component
- **DIVA**: Ilse et al., "DIVA: Domain Invariant Variational Autoencoders"
- **DANN**: Ganin et al., "Domain-Adversarial Training of Neural Networks"
- **IRM**: Arjovsky et al., "Invariant Risk Minimization"

---

**Last Updated**: 2025-11-14
**Framework Version**: Complete implementation with 5 models + full analysis pipeline

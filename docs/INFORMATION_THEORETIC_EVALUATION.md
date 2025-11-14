# Information-Theoretic Evaluation Framework

## Overview

This framework evaluates whether learned representations from NVAE and Augmented DANN models adhere to the **Minimal Information Partition (MIP) theorem**. It uses rigorous information-theoretic metrics to measure how well models partition information into class-specific, domain-specific, interaction, and residual components.

## Theoretical Background

### Minimal Information Partition Theorem

The entropy of input X decomposes as:
```
H(X) = I_Y + I_D + I_YD + I_X
```

Where:
- **I_Y = I(X;Y|D)**: Class-specific information
- **I_D = I(X;D|Y)**: Domain-specific information
- **I_YD = I(X;Y;D)**: Shared/interaction information
- **I_X = H(X) - I(X;Y,D)**: Residual information

### Minimally Partitioned Representation

A representation **Z = (Z_y, Z_dy, Z_d, Z_x)** is minimally partitioned if:

1. **Z_y** captures class-specific info: `I(Z_y;Y|D) = I_Y` and `I(Z_y;D|Y) = 0`
2. **Z_d** captures domain-specific info: `I(Z_d;D|Y) = I_D` and `I(Z_d;Y|D) = 0`
3. **Z_dy** captures shared info: `I(Z_dy;Y;D) = I_YD`
4. **Z_x** is residual: `I(Z_x;Y,D) = 0`

### Model Mapping

- **NVAE**: Full 4-component decomposition (z_y, z_ay, z_d, z_x)
- **DIVA**: 3-component decomposition (z_y, z_d, z_x) - missing z_ay
- **Augmented DANN**: Has partitioned latents but different extraction interface
- **Baseline DANN/IRM**: Monolithic feature representations without explicit partitioning

**Hypothesis**: NVAE should better adhere to MIP than DIVA because it can isolate shared information in z_ay, preventing leakage into z_y and z_d.

## Compatible Models

### Currently Supported: NVAE and DIVA

The information-theoretic evaluation framework requires models with explicit VAE-style latent decomposition. **Only NVAE and DIVA** are currently supported for evaluation.

**Requirements:**
- `qz()` method for extracting latent distributions
- Index range attributes (`zy_index_range`, `zx_index_range`, `za_index_range`, `zay_index_range`)
- `diva` flag indicating model variant

**Model Compatibility:**

| Model | Supported | Reason |
|-------|-----------|--------|
| **NVAE** | ✅ Yes | Full VAE implementation with 4-space decomposition |
| **DIVA** | ✅ Yes | VAE variant with 3-space decomposition (no z_ay) |
| **DANN (baseline)** | ❌ No | Uses monolithic feature extractor, no latent partitioning |
| **Augmented DANN** | ⚠️ Future | Has partitioned latents but requires custom extraction logic |
| **IRM** | ❌ No | Uses monolithic feature extractor, no latent partitioning |

**Note**: When running experiments with all 5 models, the IT evaluation will automatically skip non-compatible models (DANN, IRM, AugmentedDANN) and only evaluate NVAE and DIVA. This is expected behavior.

## Installation

```bash
# Install the NPEET library for information-theoretic estimation
pip install git+https://github.com/gregversteeg/NPEET.git

# Or use the requirements file
pip install -r requirements_it.txt
```

## Usage

### Option 1: Automatic Evaluation During Training

The IT evaluation runs automatically after training in both CRMNIST and WILD experiments:

```bash
# CRMNIST experiments with IT evaluation
# Note: Even though all 5 models are trained, only NVAE and DIVA will be evaluated
python -m core.CRMNIST.run_crmnist \
    --out results/crmnist_20epochs \
    --config conf/crmnist.json \
    --models nvae diva dann dann_augmented irm \
    --epochs 20 \
    --cuda

# WILD experiments with IT evaluation
python -m core.WILD.run_wild \
    --out results/wild_20epochs \
    --models nvae diva \
    --epochs 20 \
    --cuda
```

**Important**: The IT evaluation will automatically run only on compatible models (NVAE and DIVA). Other models (DANN, IRM, AugmentedDANN) will be skipped with informative messages explaining why they're not compatible.

Results will be saved to: `{output_dir}/information_theoretic_analysis/`

### Option 2: Standalone Analysis on Pre-trained Models

Use the standalone script to analyze already-trained models:

```bash
# Recommended: Only specify compatible models
python scripts/analyze_information_partition.py \
    --dataset crmnist \
    --model_dir results/crmnist_20epochs \
    --output_dir results/crmnist_20epochs/it_analysis \
    --models nvae diva \
    --max_batches 200 \
    --bootstrap 100 \
    --cuda
```

**Arguments:**
- `--dataset`: Dataset name (`crmnist` or `wild`)
- `--model_dir`: Directory containing trained model checkpoints
- `--output_dir`: Where to save IT analysis results
- `--models`: Models to evaluate - **recommend `nvae diva` only** (compatible models)
- `--max_batches`: Number of batches (~20k samples with batch_size=64)
- `--bootstrap`: Bootstrap iterations for confidence intervals (default: 100)
- `--cuda`: Use GPU if available

**Note**: The standalone script will attempt to load all specified models. If you specify incompatible models (dann, irm, augmented_dann), it will skip them with warnings.

### Option 3: Programmatic Usage

```python
from core.information_theoretic_evaluation import evaluate_model, MinimalInformationPartitionEvaluator
from core.visualization.plot_information_theoretic import visualize_all

# Evaluate a single model
results = evaluate_model(
    model=nvae_model,
    dataloader=val_loader,
    device='cuda',
    max_batches=200,
    n_bootstrap=100
)

# Compare multiple models
evaluator = MinimalInformationPartitionEvaluator()
comparison = evaluator.compare_models({
    'NVAE': nvae_results,
    'DIVA': diva_results,
    'DANN': dann_results
})

# Generate visualizations
visualize_all(comparison, output_dir='results/it_analysis')
```

## Output Files

After running IT evaluation, the following files are generated:

```
{output_dir}/information_theoretic_analysis/
├── model_comparison.json                     # Comparison across all models
├── it_model_comparison.png                   # Multi-panel comparison plot
├── it_heatmap.png                            # Heatmap of all metrics
├── it_comparison_table.csv                   # Metric table (CSV format)
├── it_comparison_table.tex                   # Metric table (LaTeX format)
├── it_summary_report.txt                     # Text summary and rankings
└── {model_name}/
    └── it_results.json                       # Individual model results
```

## Interpreting Results

### Key Metrics

| Metric | Meaning | Desired Value |
|--------|---------|--------------|
| **I(z_y;Y\|D)** | Class info in z_y | **HIGH** (>1.5 nats) |
| **I(z_y;D\|Y)** | Domain leakage into z_y | **LOW** (<0.2 nats) |
| **I(z_d;D\|Y)** | Domain info in z_d | **HIGH** (>1.0 nats) |
| **I(z_d;Y\|D)** | Class leakage into z_d | **LOW** (<0.2 nats) |
| **I(z_dy;Y;D)** | Interaction information | **POSITIVE** (>0.3 nats) for NVAE/AugDANN |
| **I(z_x;Y,D)** | Label info in residual | **LOW** (<0.3 nats) |
| **Partition Quality** | Overall adherence to MIP | **HIGH** (>0.7 excellent, >0.5 good) |

### Expected Results

**For NVAE/Augmented DANN (with z_ay):**
```
✅ I(z_y;Y|D) = 2.1-2.5 nats    (HIGH - good class specificity)
✅ I(z_y;D|Y) = 0.05-0.15 nats  (LOW - minimal domain leakage)
✅ I(z_d;D|Y) = 1.5-2.0 nats    (HIGH - good domain specificity)
✅ I(z_d;Y|D) = 0.05-0.15 nats  (LOW - minimal class leakage)
✅ I(z_ay;Y;D) = 0.3-0.8 nats   (POSITIVE - captures interaction)
✅ I(z_x;Y,D) = 0.1-0.3 nats    (LOW - clean residual)
✅ Partition Quality = 0.7-0.9  (EXCELLENT)
```

**For DIVA/Baseline DANN (without z_ay):**
```
⚠️ I(z_y;D|Y) = 0.3-0.6 nats    (HIGHER - more leakage)
⚠️ I(z_d;Y|D) = 0.3-0.6 nats    (HIGHER - more leakage)
⚠️ I(z_ay;Y;D) = 0.0 nats       (ZERO - no interaction latent)
⚠️ I(z_x;Y,D) = 0.4-0.7 nats    (HIGHER - label info in residual)
⚠️ Partition Quality = 0.4-0.6  (FAIR/GOOD)
```

### Statistical Significance

Bootstrap confidence intervals are computed for all metrics:
- If CIs don't overlap, the difference is significant (p < 0.05)
- Look for NVAE scores consistently higher than DIVA across metrics

## Implementation Details

### Estimation Method

- **KNN-based estimators** from NPEET library (Kraskov et al. 2003)
- Default k=5 neighbors for robustness
- **PCA preprocessing** for latents with >30 dimensions
- **Bootstrap resampling** (n=100) for confidence intervals

### Computational Requirements

**Per model evaluation:**
- **Samples**: ~20k (200 batches × 64 batch_size)
- **Time**: 8-12 minutes on CPU, 3-5 minutes on GPU
- **Memory**: ~2GB RAM

**Full comparison (5 models):**
- **Total time**: 40-60 minutes
- **Total memory**: ~4GB RAM

### Validation

All estimators have been validated on synthetic data:
- Independent variables → MI ≈ 0 ✓
- Deterministic relationships → MI ≈ H(Y) ✓
- Conditional independence → CMI ≈ 0 ✓
- Joint MI decomposition consistency ✓

Run validation tests:
```bash
python scripts/test_it_estimators.py
```

## Troubleshooting

### ImportError: No module named 'npeet'

```bash
pip install git+https://github.com/gregversteeg/NPEET.git
```

### High estimation variance

- Increase sample size: `--max_batches 300`
- Increase k-neighbors: Modify `n_neighbors=10` in code
- Ensure balanced class/domain distribution

### PCA warning messages

- Normal for high-dimensional latents (>30 dims)
- PCA preserves 95% variance by default
- Can adjust: `pca_variance=0.99` in code

### Out of memory errors

- Reduce batch size: `--batch_size 32`
- Reduce samples: `--max_batches 100`
- Disable bootstrap: `--bootstrap 0`

## References

1. **Kraskov et al. (2004)**: "Estimating mutual information" - KNN MI estimation
2. **NPEET**: https://github.com/gregversteeg/NPEET
3. **Minimal Information Partition Theorem**: See paper Section X.X

## Citation

If you use this IT evaluation framework in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Your Paper Title},
  author={Your Names},
  journal={Conference/Journal},
  year={2024}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your_email@domain.com

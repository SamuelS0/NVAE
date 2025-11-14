# Information-Theoretic Evaluation Framework: Technical Documentation

**Date**: 2025-11-14
**Scope**: Technical explanation of the IT evaluation implementation

---

## Overview

This document provides a detailed technical explanation of how the information-theoretic evaluation framework works, including the mathematical foundations, implementation details, and design decisions.

---

## 1. Mathematical Foundations

### 1.1 Mutual Information I(Z;Y)

**Formula**: I(Z;Y) measures how much information Z shares with Y.

**Implementation** (lines 115-143):
```python
mi = ee.mi(x_list, y_list, k=self.k)
return max(0.0, mi)  # MI cannot be negative
```

**How it works**:
- Uses NPEET's KNN-based Kraskov estimator for continuous-discrete MI
- k=5 neighbors provides good bias-variance trade-off
- Clamps to [0, ∞) since MI ≥ 0 by definition (finite samples can produce small negatives)

---

### 1.2 Conditional Mutual Information I(Z;Y|D)

**Formula**: I(Z;Y|D) measures how much information Z shares with Y when D is known.

**Implementation** (lines 145-178):
```python
cmi = ee.cmi(z_list, y_list, d_list, k=self.k)
return max(0.0, cmi)
```

**How it works**:
- Uses NPEET's conditional MI estimator for mixed continuous-discrete variables
- Critical quantities computed:
  - I(z_y;Y|D) - class info in z_y given domain
  - I(z_y;D|Y) - domain leakage into z_y given class
  - I(z_d;D|Y) - domain info in z_d given class
  - I(z_d;Y|D) - class leakage into z_d given domain

**Interpretation**:
- High I(z_y;Y|D) means z_y strongly captures class information
- Low I(z_y;D|Y) means minimal domain leakage into z_y

---

### 1.3 Interaction Information I(Z;Y;D)

**Formula**: I(Z;Y;D) measures three-way synergistic or redundant interactions.

**Implementation** (lines 180-227):
```python
# Formula 1: I(Z;Y;D) = I(Z;Y) - I(Z;Y|D)
i_zy = ee.mi(z_list, y_list, k=self.k)
i_zy_given_d = ee.cmi(z_list, y_list, d_list, k=self.k)
interaction1 = i_zy - i_zy_given_d

# Formula 2: I(Z;Y;D) = I(Z;D) - I(Z;D|Y) (validation)
i_zd = ee.mi(z_list, d_list, k=self.k)
i_zd_given_y = ee.cmi(z_list, d_list, y_list, k=self.k)
interaction2 = i_zd - i_zd_given_y

return (interaction1 + interaction2) / 2.0
```

**Mathematical equivalence**:
Both formulas are mathematically equivalent via the chain rule:
```
I(Z;Y,D) = I(Z;Y) + I(Z;D|Y) = I(Z;D) + I(Z;Y|D)

Rearranging:
I(Z;Y) - I(Z;Y|D) = I(Z;D) - I(Z;D|Y)
```

**Why average both formulas**:
- Increases robustness to estimation variance
- If estimates differ significantly, indicates high uncertainty or insufficient samples
- Conservative approach for noisy estimates

**Interpretation**:
- **Positive I(Z;Y;D)**: Synergy - Y and D together provide more info about Z than separately
- **Negative I(Z;Y;D)**: Redundancy - Y and D provide overlapping info about Z
- For z_dy latent, we expect **positive** interaction (captures joint Y-D information)

---

### 1.4 Joint Mutual Information I(Z;Y,D)

**Formula**: I(Z;Y,D) measures how much information Z shares with the joint variable (Y,D).

**Implementation** (lines 229-267):
```python
# I(Z;Y,D) = I(Z;Y) + I(Z;D|Y)
i_zy = ee.mi(z_list, y_list, k=self.k)
i_zd_given_y = ee.cmi(z_list, d_list, y_list, k=self.k)
return max(0.0, i_zy + i_zd_given_y)
```

**How it works**:
- Uses chain rule decomposition
- Both terms are non-negative, so sum is always ≥ 0
- Used to measure total label information in residual z_x (should be low)

---

### 1.5 Partition Quality Metric

**Purpose**: Single score (0-1) measuring overall adherence to Minimal Information Partition.

**Implementation** (lines 466-495):
```python
def _compute_partition_quality(self, metrics: Dict[str, float]) -> float:
    typical_entropy = 2.5  # Normalization constant

    score = 0.0

    # Positive contributions (what we want HIGH)
    score += metrics['I(z_y;Y|D)'] / typical_entropy  # Class specificity
    score += metrics['I(z_d;D|Y)'] / typical_entropy  # Domain specificity
    if metrics['I(z_dy;Y;D)'] > 0:
        score += metrics['I(z_dy;Y;D)'] / typical_entropy  # Interaction

    # Negative contributions (penalties for what should be LOW)
    score -= metrics['I(z_y;D|Y)'] / typical_entropy  # Domain leakage
    score -= metrics['I(z_d;Y|D)'] / typical_entropy  # Class leakage
    score -= metrics['I(z_x;Y,D)'] / typical_entropy  # Label info in residual

    # Normalize to [0, 1]
    max_score = 3.0
    return float(max(0.0, min(1.0, score / max_score)))
```

**How it works**:

**Normalization by typical_entropy = 2.5**:
- For CRMNIST: H(Y) ≈ log(10) ≈ 2.3 nats, H(D) ≈ log(2) ≈ 0.69 nats
- Average around 1.5-2.5 nats
- Fixed normalization works for same-dataset comparisons

**Positive terms** (ideal: each ≈ 1.0 when normalized):
1. I(z_y;Y|D) / 2.5 - Rewards capturing class info in z_y
2. I(z_d;D|Y) / 2.5 - Rewards capturing domain info in z_d
3. I(z_dy;Y;D) / 2.5 (if positive) - Rewards capturing synergistic interaction

**Negative terms** (ideal: each ≈ 0):
1. I(z_y;D|Y) / 2.5 - Penalizes domain leakage into z_y
2. I(z_d;Y|D) / 2.5 - Penalizes class leakage into z_d
3. I(z_x;Y,D) / 2.5 - Penalizes label information in residual

**Scoring range**:
- **NVAE/AugmentedDANN** (with z_dy): Can achieve up to 1.0
- **DIVA** (without z_dy): Max score ≈ 0.67
- This is intentional - rewards models that capture interaction information

**Why only positive interaction counts**:
```python
if metrics['I(z_dy;Y;D)'] > 0:  # Only add if synergistic
```
- We want synergistic (positive) interaction in z_dy
- Negative interaction (redundancy) doesn't contribute to score

---

## 2. Implementation Details

### 2.1 Latent Extraction from VAE Models

**Critical extraction code** (lines 625-633):
```python
# Get latent representations
qz_loc, qz_scale = model.qz(x)

zy = qz_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
zx = qz_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
zd = qz_loc[:, model.za_index_range[0]:model.za_index_range[1]]

if has_zdy:
    zdy = qz_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
```

**How it works**:

**VAE latent representation** (from model.py):
```python
qz_loc, qz_scale = self.qz(x)  # Encoder outputs
qz = dist.Normal(qz_loc, qz_scale)
z = qz.rsample()  # Sample from q(z|x)
```

**Why use mean (qz_loc) instead of sampling**:
- For IT evaluation, we want deterministic function of x
- Sampling adds stochastic noise that confounds measurements
- Mean of q(z|x) is the best point estimate

**Index ranges** (from model.py lines 115-122):
```python
self.zy_index_range = [0, self.zy_dim]
self.zx_index_range = [self.zy_dim, self.zy_dim + self.zx_dim]

if diva:
    self.zay_index_range = None
    self.za_index_range = [self.zy_dim + self.zx_dim, self.z_total_dim]
else:
    self.zay_index_range = [self.zy_dim + self.zx_dim, ...]
    self.za_index_range = [..., self.z_total_dim]
```

**How slicing works**:
- Index ranges stored as `[start, end]` lists
- Python slicing `[:, start:end]` extracts columns [start, start+1, ..., end-1]
- Example: `zy_index_range = [0, 12]` → extracts columns 0-11 (12 dimensions)

---

### 2.2 Bootstrap Confidence Intervals

**Implementation** (lines 269-304):
```python
def bootstrap_estimate(self, estimator_func, *args, **kwargs):
    n_samples = args[0].shape[0]
    estimates = []

    for _ in range(self.n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        args_resampled = [arr[indices] if isinstance(arr, np.ndarray) else arr
                        for arr in args]
        estimate = estimator_func(*args_resampled, **kwargs)
        estimates.append(estimate)

    estimates = np.array(estimates)
    mean_est = np.mean(estimates)
    lower_ci = np.percentile(estimates, 2.5)
    upper_ci = np.percentile(estimates, 97.5)

    return mean_est, (lower_ci, upper_ci)
```

**How it works**:
- **Bootstrap resampling**: Sample n samples with replacement from original data
- Repeat 100 times (default n_bootstrap=100)
- Compute MI estimate on each bootstrap sample
- **95% CI**: Use 2.5% and 97.5% percentiles of estimates

**Important assumption**:
- Samples must be i.i.d. (independent and identically distributed)
- Dataloader should use `shuffle=True` to avoid batch effects

**Purpose**:
- Quantify estimation uncertainty
- Allow statistical testing (non-overlapping CIs → significant difference)

---

### 2.3 PCA Dimensionality Reduction

**Implementation** (lines 100-113):
```python
def _apply_pca_if_needed(self, Z, name="Z"):
    if Z.shape[1] <= self.max_dims:  # Default: 30
        return Z, None

    pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
    Z_reduced = pca.fit_transform(Z)
    explained_var = np.sum(pca.explained_variance_ratio_)

    return Z_reduced, pca
```

**Why PCA is needed**:
- KNN estimators suffer from curse of dimensionality for dims > 30
- In high dimensions, distances become meaningless
- PCA is standard preprocessing for KNN-based MI estimation

**Configuration**:
- `max_dims = 30`: Threshold for applying PCA
- `pca_variance = 0.95`: Preserve 95% of variance
- Applied independently for each MI computation

**Trade-offs**:
- **Pro**: Reduces curse of dimensionality, improves estimation
- **Con**: PCA is linear, might lose some nonlinear information structure
- **Con**: Each MI uses different PCA projection (not directly comparable)

**Why this approach**:
- Necessary for practical implementation
- Alternative nonlinear methods (UMAP, t-SNE) are harder to apply consistently
- 95% variance preservation captures most structure

---

### 2.4 DIVA Model Handling (Models Without z_dy)

**Detection** (line 603):
```python
has_zdy = not model.diva
```

**Extraction** (lines 632-634):
```python
if has_zdy:
    zdy = qz_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
    all_zdy.append(zdy.cpu())
```

**After loop** (line 646):
```python
z_dy = torch.cat(all_zdy, dim=0).numpy() if has_zdy else None
```

**Evaluation handling** (lines 410-435):
```python
if z_dy is not None:
    # Compute interaction metrics normally
    results['metrics']['I(z_dy;Y;D)'] = self.interaction_information(...)
else:
    # DIVA model - no interaction latent
    results['metrics']['I(z_dy;Y;D)'] = 0.0
    results['metrics']['I(z_dy;Y)'] = 0.0
    results['metrics']['I(z_dy;D)'] = 0.0
    results['metrics']['I(z_dy;Y,D)'] = 0.0
```

**Partition quality impact**:
```python
if metrics['I(z_dy;Y;D)'] > 0:
    score += metrics['I(z_dy;Y;D)'] / typical_entropy
```

**How it works**:
- DIVA models have `diva=True` flag
- DIVA has no z_dy (interaction) latent space
- All z_dy metrics set to 0.0
- Partition quality score doesn't include interaction term for DIVA
- **Result**: DIVA max score ≈ 0.67, NVAE can reach ≈ 1.0

---

### 2.5 Model Filtering Implementation

**Implementation** (run_crmnist.py lines 722-733):
```python
VAE_COMPATIBLE_MODELS = ['nvae', 'diva']

for model_name, model in trained_models.items():
    if model_name not in VAE_COMPATIBLE_MODELS:
        print(f"\n⚠️  Skipping {model_name} - IT evaluation requires VAE latent decomposition")
        print(f"    Information-theoretic evaluation only supports models with explicit")
        print(f"    partitioned latent spaces (qz method, index ranges, diva flag).")
        print(f"    Compatible models: {', '.join(VAE_COMPATIBLE_MODELS)}")
        continue

    # ... proceed with evaluation
```

**Why filtering is needed**:

The IT evaluation framework requires models with:
- `qz()` method for extracting latent distributions
- `zy_index_range`, `zx_index_range`, `za_index_range`, `zay_index_range` attributes
- `diva` flag indicating model variant

**Model compatibility**:
- **NVAE**: Has all required attributes ✓
- **DIVA**: Has all required attributes (zay_index_range=None) ✓
- **DANN**: No qz(), no index_range, no diva ✗
- **IRM**: No qz(), no index_range, no diva ✗
- **AugmentedDANN**: Has latents but no qz() interface ✗

**Design alternatives**:
1. **Hardcoded whitelist** (IMPLEMENTED): Safe, explicit, clear error messages
2. **Dynamic attribute checking**: Could accidentally evaluate wrong models
3. **Custom extraction methods**: Requires modifying all models

**Why hardcoded whitelist**:
- Safest approach for preventing crashes
- Clear user communication
- Theoretically appropriate (only evaluate models designed for partitioning)

---

## 3. Design Decisions and Trade-offs

### 3.1 NPEET KNN Estimator (k=5)

**Choice**: Use NPEET library with k=5 neighbors

**Rationale**:
- KNN-based Kraskov estimator is gold standard for mixed continuous-discrete MI
- k=5 provides good bias-variance trade-off
- Well-validated in information theory literature

**Alternatives considered**:
- k=3: Lower bias, higher variance
- k=10: Lower variance, higher bias
- MCMC-based methods: More accurate but computationally expensive

---

### 3.2 Bootstrap Parameters (n=100)

**Choice**: 100 bootstrap resamples for confidence intervals

**Rationale**:
- Balances accuracy vs computational cost
- Provides reasonable CI estimates
- Standard in practice

**Alternatives**:
- n=1000: More accurate but 10x slower
- n=50: Faster but less reliable CIs

---

### 3.3 PCA Configuration (95% variance, 30-dim threshold)

**Choice**: Apply PCA when dims > 30, preserve 95% variance

**Rationale**:
- Necessary for KNN estimators in high dimensions
- 95% variance preserves most information structure
- Applied per-computation for consistency

**Trade-off**:
- Loses some nonlinear structure (PCA is linear)
- Different PCA projections not directly comparable
- But necessary for practical implementation

---

### 3.4 Averaging Interaction Formulas

**Choice**: Average two mathematically equivalent formulas

**Rationale**:
- Increases robustness to estimation variance
- Conservative estimate when formulas disagree
- Standard practice for reducing estimation noise

**Trade-off**:
- If estimates have opposite signs, averaging could mask disagreement
- Large disagreement indicates high uncertainty or small samples

---

### 3.5 Fixed typical_entropy Normalization

**Choice**: Use fixed typical_entropy=2.5 for all datasets

**Rationale**:
- Simplifies implementation
- Valid for same-dataset model comparisons (which is the use case)
- Absolute scores not meaningful across datasets anyway

**Trade-off**:
- Not adaptive to different dataset characteristics
- Different datasets have different H(Y) and H(D)
- But relative comparisons (NVAE vs DIVA) are robust

---

## 4. Computational Considerations

### 4.1 Performance

**Per model evaluation**:
- Samples: ~20k (200 batches × 64 batch_size)
- Metrics: ~15 information-theoretic quantities
- Bootstrap: 100 resamples per metric
- Time: 3-5 minutes on GPU, 8-12 minutes on CPU

**Bottlenecks**:
- KNN searches in high dimensions (mitigated by PCA)
- Bootstrap resampling (100x cost per metric)

**Optimization opportunities**:
- Parallelize bootstrap iterations
- Cache PCA projections
- Reduce n_bootstrap for faster evaluation

---

### 4.2 Memory Usage

**Per model**:
- Latents: 20k samples × ~48 dims × 4 bytes ≈ 4 MB
- Bootstrap samples: Transient, ~400 MB peak
- Peak memory: ~2 GB per model

**For 5 models**: Sequential evaluation keeps peak at ~2 GB

---

## 5. Integration with Experiments

### 5.1 CRMNIST Experiment Flow

**Location**: run_crmnist.py lines 699-800

**Workflow**:
1. Train all 5 models (NVAE, DIVA, DANN, IRM, AugmentedDANN)
2. Filter to VAE-compatible models (NVAE, DIVA)
3. For each compatible model:
   - Extract latents using `extract_latents_from_model()`
   - Evaluate partition using `evaluate_latent_partition()`
4. Compare models using `compare_models()`
5. Generate visualizations using `visualize_all()`
6. Save results to `{output_dir}/information_theoretic_analysis/`

**Output files**:
```
information_theoretic_analysis/
├── model_comparison.json
├── it_model_comparison.png
├── it_heatmap.png
├── it_comparison_table.csv
├── it_comparison_table.tex
├── it_summary_report.txt
└── {model_name}/
    └── it_results.json
```

---

### 5.2 WILD Experiment Integration

**Location**: run_wild.py lines 950-1051

**Same flow as CRMNIST** with dataset-specific handling:
- Extracts hospital IDs from metadata for domain labels
- Otherwise identical structure

---

## 6. Validation and Testing

### 6.1 Synthetic Data Tests

**Test suite**: scripts/test_it_estimators.py

**7 validation tests**:
1. **Independent Variables**: I(Z;Y) ≈ 0 for independent Z and Y
2. **Deterministic Relationship**: I(Z;Y) ≈ H(Y) when Y determined by Z
3. **Conditional Independence**: I(Z;Y|D) ≈ 0 when Z ⊥ Y | D
4. **Interaction Information**: Computes I(Z;Y;D) for XOR-like synergy
5. **Joint MI Decomposition**: Verifies I(Z;Y,D) = I(Z;Y) + I(Z;D|Y)
6. **Dimensionality Reduction**: PCA correctly reduces high-D data
7. **Full Partition**: Tests all metrics on synthetic partitioned representation

**Running tests**:
```bash
python scripts/test_it_estimators.py
```

---

### 6.2 Integration Testing

**Model filtering**:
- NVAE: Correctly loaded and evaluated
- DIVA: Correctly loaded and evaluated
- DANN: Correctly skipped with message
- IRM: Correctly skipped with message
- AugmentedDANN: Correctly skipped with message

**Edge cases**:
- DIVA with None z_dy handled correctly
- High-dimensional latents trigger PCA
- Missing domain labels handled with fallback

---

## 7. Usage Examples

### 7.1 Automatic Evaluation During Training

```bash
# CRMNIST - evaluates NVAE and DIVA automatically
python -m core.CRMNIST.run_crmnist \
    --out results/crmnist_20epochs \
    --models nvae diva dann dann_augmented irm \
    --epochs 20 \
    --cuda
```

### 7.2 Standalone Analysis

```bash
# Analyze pre-trained models
python scripts/analyze_information_partition.py \
    --dataset crmnist \
    --model_dir results/crmnist_20epochs \
    --output_dir results/it_analysis \
    --models nvae diva \
    --max_batches 200 \
    --bootstrap 100 \
    --cuda
```

### 7.3 Programmatic Usage

```python
from core.information_theoretic_evaluation import evaluate_model

# Evaluate single model
results = evaluate_model(
    model=nvae_model,
    dataloader=val_loader,
    device='cuda',
    max_batches=200,
    n_bootstrap=100
)

# Access metrics
print(f"I(z_y;Y|D) = {results['metrics']['I(z_y;Y|D)']:.4f}")
print(f"Partition quality = {results['metrics']['partition_quality']:.4f}")
```

---

## 8. Interpreting Results

### 8.1 Key Metrics

| Metric | Meaning | Desired Value |
|--------|---------|---------------|
| I(z_y;Y\|D) | Class info in z_y | HIGH (>1.5 nats) |
| I(z_y;D\|Y) | Domain leakage into z_y | LOW (<0.2 nats) |
| I(z_d;D\|Y) | Domain info in z_d | HIGH (>1.0 nats) |
| I(z_d;Y\|D) | Class leakage into z_d | LOW (<0.2 nats) |
| I(z_dy;Y;D) | Interaction information | POSITIVE (>0.3 nats) for NVAE |
| I(z_x;Y,D) | Label info in residual | LOW (<0.3 nats) |
| Partition Quality | Overall adherence to MIP | HIGH (>0.7 excellent, >0.5 good) |

### 8.2 Expected Results

**NVAE (with z_dy)**:
- I(z_y;Y|D) = 2.1-2.5 nats (high class specificity)
- I(z_y;D|Y) = 0.05-0.15 nats (minimal domain leakage)
- I(z_d;D|Y) = 1.5-2.0 nats (high domain specificity)
- I(z_d;Y|D) = 0.05-0.15 nats (minimal class leakage)
- I(z_dy;Y;D) = 0.3-0.8 nats (captures interaction)
- I(z_x;Y,D) = 0.1-0.3 nats (clean residual)
- Partition Quality = 0.7-0.9 (excellent)

**DIVA (without z_dy)**:
- I(z_y;D|Y) = 0.3-0.6 nats (more leakage)
- I(z_d;Y|D) = 0.3-0.6 nats (more leakage)
- I(z_dy;Y;D) = 0.0 nats (no interaction latent)
- I(z_x;Y,D) = 0.4-0.7 nats (more label info in residual)
- Partition Quality = 0.4-0.6 (fair/good)

### 8.3 Statistical Significance

- Bootstrap CIs quantify uncertainty
- Non-overlapping CIs indicate significant difference (p < 0.05)
- NVAE should show consistently higher scores than DIVA

---

## Conclusion

This documentation explains the technical implementation of the information-theoretic evaluation framework for testing adherence to the Minimal Information Partition theorem. The framework computes rigorous information-theoretic quantities using KNN-based estimators, applies appropriate dimensionality reduction, and provides comprehensive comparisons between models with bootstrap confidence intervals.

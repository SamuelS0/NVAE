# Staged Training for Better Disentanglement

## Problem

In standard VAE training with multiple latent variables (`za`, `zy`, `zay`), the `zay` variable often captures redundant information from both `za` and `zy` instead of learning only their synergistic interaction. This leads to poor disentanglement where:

- `zay` contains information that should be in `za` or `zy`
- The latent variables are not properly specialized
- Downstream tasks suffer from entangled representations

## Solution: Staged Training with Capacity Annealing

We implement a three-stage training approach:

### Stage 1: Foundation Training (za, zy only)
- **Duration**: First 1/3 of total epochs (default)
- **Objective**: Force `za` and `zy` to encode their respective factors
- **Method**: Set `zay` capacity to near-zero (0.01) to suppress its learning
- **Goal**: Ensure `za` learns domain information and `zy` learns class information

### Stage 2: Gradual Introduction (zay with optional independence penalty)
- **Duration**: Second 1/3 of total epochs (default)
- **Objective**: Introduce `zay` while preventing it from capturing `za`/`zy` information
- **Method**: 
  - Gradually increase `zay` capacity from 0.0 to 1.0
  - Optionally add independence penalty to minimize correlation between `zay` and `za`/`zy`
- **Goal**: `zay` learns only synergistic class-domain interactions

### Stage 3: Full Training
- **Duration**: Final 1/3 of total epochs (default)
- **Objective**: Fine-tune all latent variables together
- **Method**: Standard VAE training with all variables active
- **Goal**: Optimize overall model performance while maintaining disentanglement

## Key Components

### 1. Capacity Annealing
```python
if self.current_stage == 1:
    zay_capacity = 0.01  # Suppress zay
elif self.current_stage == 2:
    progress = self.stage_epoch / max(1, self.stage2_epochs - 1)
    zay_capacity = progress  # Gradually increase from 0 to 1
else:
    zay_capacity = 1.0  # Full capacity
```

### 2. Independence Penalty
The independence penalty prevents `zay` from capturing information that should be in `za` or `zy`:

```python
def _compute_independence_penalty(self, y, x, a):
    # Compute correlations between zay and za/zy
    corr_zay_zy = correlation(zay, zy)
    corr_zay_za = correlation(zay, za)
    return corr_zay_zy + corr_zay_za
```

### 3. Stage-Specific Loss
```python
if stage == 1:
    loss = recon_loss + β * (KL_zy + KL_zx + 0.01 * KL_zay + KL_za) + α * class_losses
elif stage == 2:
    loss = recon_loss + β * (KL_zy + KL_zx + capacity * KL_zay + KL_za) + α * class_losses
    if use_independence_penalty:
        loss += λ * independence_penalty
else:
    loss = recon_loss + β * (KL_zy + KL_zx + KL_zay + KL_za) + α * class_losses
```

## Usage

### Basic Staged Training
```bash
python -B core/WILD/run_wild.py \
    --staged_training \
    --epochs 60 \
    --out results_staged/ \
    --independence_penalty 10.0
```

### Custom Stage Lengths
```bash
python -B core/WILD/run_wild.py \
    --staged_training \
    --epochs 60 \
    --stage1_epochs 30 \
    --stage2_epochs 15 \
    --out results_staged_custom/ \
    --independence_penalty 15.0
```

### Staged Training Without Independence Penalty
```bash
python -B core/WILD/run_wild.py \
    --staged_training \
    --no_independence_penalty \
    --epochs 60 \
    --out results_staged_no_penalty/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--staged_training` | False | Enable staged training |
| `--stage1_epochs` | epochs//3 | Epochs for Stage 1 (za, zy only) |
| `--stage2_epochs` | epochs//3 | Epochs for Stage 2 (gradual zay) |
| `--use_independence_penalty` | True | Use independence penalty in Stage 2 |
| `--no_independence_penalty` | - | Disable independence penalty |
| `--independence_penalty` | 10.0 | Weight for independence penalty |

## Evaluation Metrics

The training automatically generates disentanglement analysis with key metrics:

### 1. Independence Scores
- **zay_independence_score**: Higher is better (closer to 1.0)
- **zy_zay_correlation**: Lower is better (closer to 0.0)
- **za_zay_correlation**: Lower is better (closer to 0.0)

### 2. Specialization Scores
- **zy_y_specificity**: Higher is better (zy should predict class y, not domain a)
- **za_a_specificity**: Higher is better (za should predict domain a, not class y)

### 3. Visualizations
- `correlation_heatmap.png`: Shows correlations between all latent variables
- `label_prediction_accuracy.png`: Shows how well each latent variable predicts labels

## Expected Results

With proper staged training, you should see:

1. **Lower correlations** between `zay` and `za`/`zy`
2. **Higher independence score** for `zay`
3. **Better specialization**: `zy` predicts class, `za` predicts domain
4. **Cleaner latent space** visualizations
5. **Better downstream task performance**

## Troubleshooting

### Problem: za/zy not learning their factors in Stage 1
**Solution**: Increase `--stage1_epochs` to give them more time

### Problem: zay still captures za/zy information
**Solution**: 
- Increase `--independence_penalty` weight, or
- Try `--no_independence_penalty` and rely on capacity annealing alone

### Problem: Poor reconstruction quality
**Solution**: Use `--beta_annealing` to balance reconstruction and KL terms

### Problem: Training instability
**Solution**: Reduce learning rate or increase patience

## Theory

This approach is based on the principle that:

1. **Specialization first**: Force each variable to learn its intended factor
2. **Gradual complexity**: Introduce interactions only after basics are learned
3. **Explicit constraints**: Use penalties to enforce desired properties
4. **Capacity control**: Limit learning capacity to prevent information leakage

The result is better disentangled representations where each latent variable has a clear, interpretable role. 
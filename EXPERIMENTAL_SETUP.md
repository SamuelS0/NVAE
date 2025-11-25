# Experimental Setup: Domain Generalization with Disentangled Representations

## 1. Overview

This document provides a comprehensive technical description of our experimental framework for evaluating disentangled representation learning models on domain generalization tasks. We compare multiple architectures across two distinct datasets using both traditional performance metrics and novel information-theoretic evaluation criteria based on the Minimal Information Partition theorem.

**Key Components:**
- Two datasets: CRMNIST (synthetic) and WILD/Camelyon17 (real-world medical imaging)
- Five model architectures: NVAE, DIVA, DANN, AugmentedDANN, and IRM
- Latent expressiveness evaluation via classifier probing
- Information-theoretic evaluation of representation quality

---

## 2. Datasets

### 2.1 CRMNIST (Colored Rotated MNIST)

**Description:** A synthetic benchmark dataset extending MNIST with controlled color and rotation transformations to study spurious correlations and domain shifts.

**Specifications:**
- **Input dimensions:** 28×28×3 (RGB)
- **Task variable (Y):** 10 digit classes (0-9)
- **Spurious variable (C):** 7 color transformations
- **Domain variable (R):** 6 rotation angles (0°, 10°, 20°, 30°, 40°, 50°)
- **Total samples:** ~60,000 training, ~10,000 test

**Data Generation Process:**
1. Load grayscale MNIST digits (28×28×1)
2. Apply color transformation: Multiply by RGB color vector with intensity ∈ [1.0, 1.5]
3. Apply rotation: Rotate image by discrete angle from R = {0°, 10°, 20°, 30°, 40°, 50°}
4. Normalize pixel values to [0, 1]

**Domain Configuration:**
- Each domain corresponds to a specific rotation angle
- Color-digit correlations are configurable (spurious correlations can be introduced)
- Training/validation split: 80/20 stratified by domain

**Dataset Splits:**
- **Training:** All 6 rotation domains, balanced across digits and colors
- **Validation:** 20% held-out from training data (in-distribution)
- **Test (ID):** In-distribution test set with same domain distribution
- **Test (OOD):** Optional out-of-distribution set with excluded domains

**Cache System:** Generated datasets are cached using MD5 hashing of configuration parameters for reproducibility and efficiency.

---

### 2.2 WILD (Camelyon17)

**Description:** A real-world medical imaging dataset from the WILDS benchmark, containing histopathology image patches for tumor detection across multiple hospitals (domains).

**Specifications:**
- **Input dimensions:** 96×96×3 (RGB histopathology patches)
- **Task variable (Y):** 2 classes (0=Normal tissue, 1=Tumor tissue)
- **Domain variable (A):** 5 hospitals (centers 0-4)
- **Total samples:** ~302,436 patches

**Data Source:**
- Dataset: Camelyon17-WILDS v1.0
- Downloaded via WILDS package
- Cached locally at `~/data/wilds/camelyon17_v1.0/`
- Size: ~10GB download

**Domain Distribution:**
```
Hospital 0: ~50,000 samples
Hospital 1: ~50,000 samples
Hospital 2: ~60,000 samples
Hospital 3: ~70,000 samples
Hospital 4: ~72,000 samples
```

**Dataset Splits (Official WILDS Protocol):**
- **Training hospitals:** {0, 3, 4} - Used for model training
- **Validation hospital (OOD):** {1} - Out-of-distribution validation
- **ID Validation hospital:** {0, 3, 4} - In-distribution validation subset
- **Test hospital (OOD):** {2} - Final out-of-distribution evaluation

**Key Challenges:**
- Hospital-specific staining variations (domain shift)
- Class imbalance (~60% normal, ~40% tumor)
- High-resolution medical imaging requiring specialized architectures
- Real-world domain shift (not synthetic)

**Data Loading:**
- Uses WILDS `get_dataset()` and `get_eval_loader()` utilities
- Standard transforms: `ToTensor()` normalization
- Batch loading with metadata (hospital ID, slide ID, coordinates)

---

## 3. Model Architectures

All models are designed to learn disentangled representations that separate task-relevant information from domain-specific information. We compare three paradigms: generative models (NVAE, DIVA), adversarial models (DANN, AugmentedDANN), and invariance-based models (IRM).

### 3.1 NVAE (Novel Variational Autoencoder)

**Architecture Type:** Variational Autoencoder with 4-space latent decomposition

**Latent Space Structure:**
- **z_y (class-specific):** Captures task-relevant features
  - Prior: p(z_y|y) - Gaussian conditioned on class label
  - Auxiliary classifier: q(y|z_y, z_ay) predicts class from z_y (+z_ay)
- **z_x (residual):** Captures task-irrelevant, domain-invariant features
  - Prior: p(z_x) - Standard Gaussian N(0, I)
  - No auxiliary prediction
- **z_ay (interaction):** Captures class-domain interactions
  - Prior: p(z_ay|y,a) - Gaussian conditioned on both class and domain
  - Used by both auxiliary classifiers
- **z_a (domain-specific):** Captures domain features
  - Prior: p(z_a|a) - Gaussian conditioned on domain label
  - Auxiliary classifier: q(a|z_a, z_ay) predicts domain

**Default Latent Dimensions:**
- CRMNIST: z_y=32, z_x=64, z_ay=16, z_a=32 (total: 144)
- WILD: z_y=128, z_x=128, z_ay=128, z_a=128 (total: 512)

**Encoder Architecture (qz):**

*CRMNIST (28×28×3 input):*
```
Block 1: Conv2d(3→96, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 2: MaxPool2d(2×2)              → 14×14×96
Block 3: Conv2d(96→192, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 4: MaxPool2d(2×2)              → 7×7×192
Block 5: Flatten                     → 9408
Block 6: Linear(9408 → z_total_dim)
Block 7: Softplus → scale + 1e-7 (for numerical stability)
```

*WILD (96×96×3 input):*
```
Block 1: Conv2d(3→64, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
         MaxPool2d(2×2)              → 48×48×64
Block 2: Conv2d(64→128, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
         MaxPool2d(2×2)              → 24×24×128
Block 3: Conv2d(128→256, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
         MaxPool2d(2×2)              → 12×12×256
Block 4: Conv2d(256→512, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
         MaxPool2d(2×2)              → 6×6×512
Block 5: Flatten                     → 18432
Block 6: Linear(18432 → z_total_dim)
         Scale: Linear + Softplus + 1e-7
```

**Auxiliary Classifiers:**

*q(y) - Class Predictor:*
```
Input: Concat(z_y, z_ay)  [or z_y only for DIVA]
FC1: Linear(z_combined → 64) + ReLU
FC2: Linear(64 → 64) + ReLU
FC3: Linear(64 → num_classes)
Output: Logits for CrossEntropy loss
Initialization: Xavier uniform, zero bias
```

*q(a) - Domain Predictor:*
```
Input: Concat(z_a, z_ay)  [or z_a only for DIVA]
FC1: Linear(z_combined → 64) + ReLU
FC2: Linear(64 → 64) + ReLU
FC3: Linear(64 → num_domains)
Output: Logits for CrossEntropy loss
Initialization: Xavier uniform, zero bias
```

**Prior Networks:**

*p(z_y|y):*
```
Embedding(num_classes → 64)
FC: Linear(64 → 2*z_y_dim)
Split into μ and log(σ)
```

*p(z_a|a):*
```
Embedding(num_domains → 64)
FC: Linear(64 → 2*z_a_dim)
Split into μ and log(σ)
```

*p(z_ay|y,a):*
```
y_embed: Embedding(num_classes → 32)
a_embed: Embedding(num_domains → 32)
Concat(y_embed, a_embed)
FC: Linear(64 → 2*z_ay_dim)
Split into μ and log(σ)
```

**Decoder Architecture (px):**

*CRMNIST (output 28×28×3):*
```
Block 1: Linear(z_total → 64×7×7) + BatchNorm1d + ReLU
Block 2: Reshape to (64, 7, 7)
Block 3: Upsample(×2) → 14×14
Block 4: ConvTranspose2d(64→128, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 5: Upsample(×2) → 28×28
Block 6: ConvTranspose2d(128→256, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 7: Conv2d(256→3, k=1) + Sigmoid
```

*WILD (output 96×96×3):*
```
Block 1: Linear(z_total → 512×6×6) + BatchNorm1d + ReLU
Block 2: Reshape to (512, 6, 6)
Block 3: ConvTranspose2d(512→256, k=4, s=2, p=1) + BatchNorm2d + ReLU → 12×12
Block 4: ConvTranspose2d(256→128, k=4, s=2, p=1) + BatchNorm2d + ReLU → 24×24
Block 5: ConvTranspose2d(128→64, k=4, s=2, p=1) + BatchNorm2d + ReLU → 48×48
Block 6: ConvTranspose2d(64→3, k=4, s=2, p=1) + Sigmoid → 96×96
```

**Loss Function:**
```
L_total = w_recon * L_recon
        + β * (β₁*KL[q(z_y)||p(z_y|y)] + β₂*KL[q(z_x)||p(z_x)]
             + β₃*KL[q(z_ay)||p(z_ay|y,a)] + β₄*KL[q(z_a)||p(z_a|a)])
        + α₁*CE(q(y|z_y,z_ay), y)
        + α₂*CE(q(a|z_a,z_ay), a)
        + λ_sparsity * (||z_y||₁ + ||z_x||₁ + ||z_ay||₁ + ||z_a||₁)

where:
  L_recon = MSE(x_recon, x)
  KL[·||·] = KL divergence between posterior and prior
  CE(·,·) = Cross-entropy loss
  β = annealing coefficient (increases from 0 to β_scale during training)
```

**Hyperparameters (WILD default):**
- w_recon = 1.0
- β₁ = β₂ = β₃ = β₄ = 1.0
- β_scale = 1.0 (with annealing schedule)
- α₁ = 150.0 (class prediction weight)
- α₂ = 40.0 (domain prediction weight)
- λ_sparsity: zy=10.0, zx=10.0, zay=100.0, za=10.0

---

### 3.2 DIVA (Domain Invariant Variational Autoencoder)

**Architecture Type:** Variational Autoencoder with 3-space latent decomposition (no interaction term)

**Latent Space Structure:**
- **z_y (class-specific):** Increased dimensionality to compensate for missing z_ay
- **z_x (residual):** Same as NVAE
- **z_a (domain-specific):** Increased dimensionality to compensate for missing z_ay
- **z_ay:** Explicitly removed (set to None)

**Dimension Redistribution:**
When converting from NVAE configuration with z_ay_dim, DIVA redistributes:
```python
base_dim = z_ay_dim // 3
remainder = z_ay_dim % 3

z_y_dim_diva = z_y_dim + base_dim + (1 if remainder >= 1 else 0)
z_x_dim_diva = z_x_dim + base_dim + (1 if remainder >= 2 else 0)
z_a_dim_diva = z_a_dim + base_dim
z_ay_dim_diva = 0
```

Example: NVAE (32, 64, 16, 32) → DIVA (37, 69, 0, 37) with total=144

**Encoder/Decoder:** Identical architecture to NVAE, but outputs 3 latent spaces

**Auxiliary Classifiers:** Simplified (no z_ay concatenation):
```
q(y): Input=z_y only → FC(z_y, 64) → FC(64, 64) → FC(64, num_classes)
q(a): Input=z_a only → FC(z_a, 64) → FC(64, 64) → FC(64, num_domains)
```

**Loss Function:**
```
L_total = w_recon * L_recon
        + β * (β₁*KL[q(z_y)||p(z_y|y)] + β₂*KL[q(z_x)||p(z_x)] + β₄*KL[q(z_a)||p(z_a|a)])
        + α₁*CE(q(y|z_y), y)
        + α₂*CE(q(a|z_a), a)
        + λ_sparsity * (||z_y||₁ + ||z_x||₁ + ||z_a||₁)

Note: β₃ term and z_ay terms are removed
```

**Key Difference from NVAE:** DIVA assumes class-domain interactions can be captured implicitly in z_y and z_a, rather than requiring an explicit z_ay space. This is the original DIVA paper formulation.

---

### 3.3 DANN (Domain-Adversarial Neural Network)

**Architecture Type:** Discriminative model with gradient reversal layer

**Latent Space:** Single unified feature space (no explicit decomposition)
- Feature dimension: z_dim = z_y + z_x + z_ay + z_a (matches total VAE capacity)
- Example: CRMNIST z_dim = 144, WILD z_dim = 512

**Feature Extractor:**

*CRMNIST:*
```
Block 1: Conv2d(3→96, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 2: MaxPool2d(2×2)              → 14×14×96
Block 3: Conv2d(96→192, k=5, s=1, p=2) + BatchNorm2d + ReLU
Block 4: MaxPool2d(2×2)              → 7×7×192
Block 5: Flatten                     → 9408
Block 6: Linear(9408 → z_dim)
```

*WILD:*
```
Block 1: Conv2d(3→64, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d → 48×48×64
Block 2: Conv2d(64→128, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d → 24×24×128
Block 3: Conv2d(128→256, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d → 12×12×256
Block 4: Conv2d(256→512, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d → 6×6×512
Block 5: Flatten + Linear(18432 → z_dim)
```

**Label Classifier:**
```
Linear(z_dim → 64) + ReLU
Linear(64 → 64) + ReLU
Linear(64 → num_classes)
Initialization: Xavier uniform, zero bias
```

**Domain Discriminator (with Gradient Reversal Layer):**
```
Input: features → GradientReversalLayer(λ) → reversed_features

Network:
  Linear(z_dim → 100) + ReLU
  Linear(100 → 100) + ReLU
  Linear(100 → num_domains)

GRL backward pass: grad_output → -λ * grad_output
```

**Loss Function:**
```
L_total = L_class + L_domain

where:
  L_class = CrossEntropy(classifier(features), y)
  L_domain = CrossEntropy(discriminator(GRL(features, λ)), r)

Note: λ is NOT applied to L_domain (GRL handles it via gradient modification)
```

**Lambda Schedule (Adaptive):**
```python
λ(epoch) = 2/(1 + exp(-10 * epoch/max_epochs)) - 1

λ starts at 0 (no adversarial training)
λ → 1 as training progresses (full adversarial)
```

**Training Dynamics:**
- Forward pass: Classifier predicts class, discriminator predicts domain
- Backward pass (classifier): Standard gradients to minimize L_class
- Backward pass (discriminator): Standard gradients to minimize L_domain
- Backward pass (feature extractor):
  - Positive gradients from L_class (improve classification)
  - Negative gradients from L_domain (confuse discriminator via GRL)
  - Net effect: Learn domain-invariant features

---

### 3.4 AugmentedDANN

**Architecture Type:** Adversarial model with explicit 3-space latent decomposition

**Latent Space Structure:**
- **z_y (class-specific):** Task-relevant features
- **z_d (domain-specific):** Domain features (not adversarially trained)
- **z_dy (interaction):** Class-domain interaction features (adversarially trained)

**Dimension Redistribution:** Same as DIVA (distribute total across 3 spaces)

**Feature Extractor:** Similar to DANN, but outputs 3 separate feature vectors

*CRMNIST:*
```
Shared Encoder: (same as DANN)
  → Flatten → 9408

Split into 3 branches:
  FC_y: Linear(9408 → z_y_dim)
  FC_d: Linear(9408 → z_d_dim)
  FC_dy: Linear(9408 → z_dy_dim)
```

**Classifiers:**
```
Label Classifier:
  Input: z_y only
  Linear(z_y_dim → 64) + ReLU
  Linear(64 → num_classes)

Domain Classifier:
  Input: z_d only
  Linear(z_d_dim → 64) + ReLU
  Linear(64 → num_domains)
```

**Domain Discriminator (adversarial on z_dy):**
```
Input: z_dy → GRL(λ) → reversed_z_dy
Linear(z_dy_dim → 64) + ReLU
Linear(64 → num_domains)
```

**Loss Function:**
```
L_total = α_y * L_class
        + α_d * L_domain
        + β_adv * L_adversarial
        + λ_sparsity * ||z_dy||₁

where:
  L_class = CrossEntropy(classifier(z_y), y)
  L_domain = CrossEntropy(domain_clf(z_d), d)
  L_adversarial = CrossEntropy(discriminator(GRL(z_dy)), d)

Sparsity encourages z_dy to be minimal (interaction should be small)
```

**Hyperparameters (default):**
- α_y = 150.0 (class weight)
- α_d = 40.0 (domain weight)
- β_adv = 0.1 (adversarial weight)
- λ_reversal = 1.0 (GRL strength)
- λ_sparsity = 0.01 (interaction sparsity)

**Key Innovation:** Separates domain information (z_d) from domain-invariant interaction (z_dy), allowing explicit modeling of what should be invariant vs. what should capture domain.

---

### 3.5 IRM (Invariant Risk Minimization)

**Architecture Type:** Discriminative model with invariance penalty

**Latent Space:** Single unified feature space (like DANN)
- Feature dimension: z_dim = z_y + z_x + z_ay + z_a

**Feature Extractor:** Identical to DANN architecture

**Classifier:** Identical to DANN classifier

**Key Innovation - IRM Penalty:**
The IRM penalty encourages the optimal classifier to be the same across all environments (domains). This is achieved by penalizing the gradient of the loss w.r.t. a dummy classifier.

**IRM Penalty Computation:**
```python
def compute_irm_penalty(logits, y):
    # Create dummy classifier weight (all ones)
    dummy_w = ones_like(logits).requires_grad_()

    # Compute loss with dummy classifier
    dummy_logits = logits * dummy_w
    dummy_loss = CrossEntropy(dummy_logits, y)

    # Compute gradient of loss w.r.t. dummy classifier
    dummy_grads = autograd.grad(dummy_loss, dummy_w, create_graph=True)[0]

    # IRM penalty is squared norm of these gradients
    penalty = sum(dummy_grads ** 2)

    return penalty
```

**Loss Function (per environment):**
```
For each domain d in unique_domains:
    # Get samples from this domain
    mask_d = (domain_labels == d)
    logits_d = logits[mask_d]
    y_d = y[mask_d]

    # Classification loss for this domain
    L_class_d = CrossEntropy(logits_d, y_d)

    # IRM penalty for this domain
    penalty_d = compute_irm_penalty(logits_d, y_d)

L_total = Σ_d (L_class_d + w_penalty * penalty_d)

where w_penalty = 0 for first N iterations (annealing)
                = penalty_weight after N iterations
```

**Hyperparameters:**
- penalty_weight = 1e4 (WILD), 1e3 (CRMNIST)
- penalty_anneal_iters: Controls when penalty activates
  - WILD default: 500 iterations (~6 epochs with batch_size=64)
  - CRMNIST default: 100 iterations

**Training Dynamics:**
- Early training: Only minimize classification loss (penalty_weight=0)
- After annealing: Penalize environments where gradient of loss w.r.t. classifier weights is non-zero
- Result: Learn features where a simple linear classifier works equally well across all domains

**Key Difference from DANN:**
- DANN: Explicitly removes domain information via adversarial training
- IRM: Learns representations where optimal classifier is domain-invariant

---

## 4. Training Process

### 4.1 Optimization

**Optimizer:** Adam
- Learning rate: 1e-4 (WILD), 1e-3 (CRMNIST)
- Beta parameters: (0.9, 0.999)
- Weight decay: 0 (no L2 regularization)
- Epsilon: 1e-8

**Batch Sizes:**
- WILD: 32-64 samples/batch
- CRMNIST: 64-128 samples/batch

**Training Duration:**
- WILD: 10-50 epochs (~3,000-15,000 iterations)
- CRMNIST: 5-20 epochs (~500-2,000 iterations)

### 4.2 Beta Annealing (VAE models only)

VAE models (NVAE, DIVA) use KL annealing to prevent posterior collapse:

```python
def get_current_beta(epoch, max_epochs, beta_scale):
    # Linear annealing from 0 to beta_scale
    progress = min(epoch / (max_epochs * 0.5), 1.0)  # Reach full β at 50% of training
    return progress * beta_scale
```

Applies to all KL terms: β(epoch) * (β₁*KL_zy + β₂*KL_zx + β₃*KL_zay + β₄*KL_za)

### 4.3 Early Stopping

**Mechanism:**
- Monitor validation loss (or accuracy for discriminative models)
- Patience: 100 epochs (WILD), 50 epochs (CRMNIST)
- Save best model checkpoint based on validation metric
- Restore best model after training for final evaluation

**Validation Frequency:** After every epoch

### 4.4 Checkpointing

**Saved Information:**
```python
checkpoint = {
    'state_dict': model.state_dict(),
    'epoch': current_epoch,
    'training_metrics': {
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'total_epochs_trained': total_epochs,
        'best_model_epoch': best_epoch
    }
}
```

**Checkpoint Naming:** `{model_name}_model_epoch_{best_epoch}.pt`

### 4.5 Training Visualization

**Per-Epoch Visualizations:**
- Latent space t-SNE projections (colored by class, domain, spurious)
- Reconstruction quality samples
- Training curves (loss, accuracy, KL terms, penalties)

**Saved to:** `{output_dir}/{model_name}_latent_viz/`

---

## 5. Latent Expressiveness Evaluation

### 5.1 Methodology

**Goal:** Measure how well each latent subspace captures its intended information by training simple classifiers to predict labels from latent representations.

**Evaluation Protocol:**
1. Extract latent representations from trained model
2. Train separate linear classifiers (Logistic Regression) on each latent subspace
3. Evaluate classifier accuracy on held-out test set
4. Compare accuracy across subspaces to validate disentanglement

### 5.2 Latent Extraction

```python
def extract_latent_representations(model, dataloader, device):
    """Extract all latent subspaces from trained model."""

    all_zy, all_zx, all_zay, all_za = [], [], [], []
    all_y, all_a = [], []

    for x, y, metadata in dataloader:
        x = x.to(device)
        a = metadata[:, 0].long()  # Hospital/domain ID

        # For VAE models (NVAE/DIVA):
        _, z, qz, pzy, pzx, pza, pzay, _, _, zy, zx, zay, za = model(y, x, a)

        # For AugmentedDANN:
        zy, zd, zdy = model.extract_features(x)
        # Map: za←zd, zay←zdy, zx←empty

        all_zy.append(zy.cpu())
        all_za.append(za.cpu())
        all_zay.append(zay.cpu() if zay else zeros(...))
        all_zx.append(zx.cpu())
        all_y.append(y.cpu())
        all_a.append(a.cpu())

    return {
        'zy': concat(all_zy), 'zx': concat(all_zx),
        'zay': concat(all_zay), 'za': concat(all_za),
        'y': concat(all_y), 'a': concat(all_a)
    }
```

**Note:** Baseline DANN and IRM models have monolithic feature representations and cannot be evaluated with this method.

### 5.3 Classifier Training

**Classifier:** Sklearn Logistic Regression
- Solver: lbfgs
- Max iterations: 1000
- Multi-class: multinomial (CRMNIST), binary (WILD)
- Regularization: L2 (C=1.0)
- Random state: 42 (reproducibility)

**Training Data Split:**
- Training set: 80% of extracted representations
- Validation set: 20% of extracted representations
- Test set: Separate held-out data from model training

**Evaluation Metrics:**
- Accuracy: Primary metric (% correctly classified)
- Per-class precision, recall, F1-score
- Confusion matrices

### 5.4 Evaluation Scenarios

**Individual Subspace Tests:**
1. **zy → y:** How well does class-specific latent predict class?
   - Expected: High accuracy (zy should capture class info)
2. **zy → a:** How well does class-specific latent predict domain?
   - Expected: Low accuracy (zy should be domain-invariant)
3. **za → a:** How well does domain-specific latent predict domain?
   - Expected: High accuracy (za should capture domain info)
4. **za → y:** How well does domain-specific latent predict class?
   - Expected: Low accuracy (za should be class-invariant)
5. **zay → (y,a):** How well does interaction latent predict both?
   - Expected: Moderate accuracy for both (captures interactions)
6. **zx → y/a:** How well does residual latent predict anything?
   - Expected: Low accuracy for both (should be uninformative)

**Combined Subspace Tests:**
7. **Concat(zy, zay) → y:** Full class prediction (as in model)
8. **Concat(za, zay) → a:** Full domain prediction (as in model)
9. **All → y:** Upper bound on class prediction
10. **All → a:** Upper bound on domain prediction

### 5.5 Results Interpretation

**Ideal Disentanglement Profile:**
```
zy → y:  ✓ High (>85%)
zy → a:  ✗ Low  (<25%)
za → a:  ✓ High (>85%)
za → y:  ✗ Low  (<25%)
zx → *:  ✗ Low  (<25%)
zay → y: ~ Mid  (40-60%)
zay → a: ~ Mid  (40-60%)
```

**Failure Modes:**
- **Leakage:** zy → a is high (class features contain domain info)
- **Incomplete:** zy → y is low (class features don't capture class)
- **Non-disentangled:** za → y is high (domain features needed for classification)

### 5.6 Output Files

Saved to `{output_dir}/{model_name}_expressiveness/`:
```
- latent_expressiveness_results.json  # All accuracy metrics
- zy_to_y_classification_report.txt   # Detailed per-class metrics
- za_to_a_classification_report.txt
- confusion_matrix_zy_to_y.png        # Visualization
- expressiveness_summary.png          # Bar plot of all accuracies
```

---

## 6. Information-Theoretic Evaluation

### 6.1 Theoretical Foundation

**Minimal Information Partition Theorem:**

The entropy H(X) of input X decomposes as:
```
H(X) = I_Y + I_D + I_YD + I_X

where:
  I_Y = I(X;Y|D)      - Class-specific information (unique to class, independent of domain)
  I_D = I(X;D|Y)      - Domain-specific information (unique to domain, independent of class)
  I_YD = I(X;Y;D)     - Shared/interaction information (depends on both)
  I_X = H(X) - I(X;Y,D) - Residual information (independent of both)
```

**Ideal Partition:** A representation Z = (z_y, z_d, z_dy, z_x) is minimally partitioned if:
1. **z_y captures I_Y:** I(z_y;Y|D) = I_Y and I(z_y;D|Y) = 0
2. **z_d captures I_D:** I(z_d;D|Y) = I_D and I(z_d;Y|D) = 0
3. **z_dy captures I_YD:** I(z_dy;Y;D) = I_YD
4. **z_x captures I_X:** I(z_x;Y,D) = 0

### 6.2 Mutual Information Estimation

**Method:** KNN-based estimation using Kraskov et al. (2003) algorithm via NPEET library

**Estimator:** `npeet.entropy_estimators`
- **mi(X, Y):** Mutual information I(X;Y) for continuous X, discrete Y
- **cmi(X, Y, Z):** Conditional mutual information I(X;Y|Z)
- k-neighbors: 7 (default, balances bias-variance)

**High-Dimensional Handling:**
- If latent dimension > 30, apply PCA to preserve 99% variance
- Standardize features before MI estimation
- Log transform if needed for numerical stability

### 6.3 Information-Theoretic Quantities

**Measured Quantities (per model):**

1. **Class-specific information:**
   - I(z_y; Y|D) - z_y should maximize this
   - I(z_y; D|Y) - z_y should minimize this (leakage)

2. **Domain-specific information:**
   - I(z_d; D|Y) - z_d should maximize this
   - I(z_d; Y|D) - z_d should minimize this (leakage)

3. **Interaction information:**
   - I(z_dy; Y; D) - z_dy should capture class-domain synergy
   - Computed as: I(z_dy; Y; D) = I(z_dy; Y) - I(z_dy; Y|D)
   - Validated with: I(z_dy; D) - I(z_dy; D|Y)

4. **Residual information:**
   - I(z_x; Y,D) - z_x should minimize this (be uninformative)
   - Computed as: I(z_x; Y) + I(z_x; D|Y)

5. **Joint mutual information:**
   - I(z_*; Y,D) for each subspace (informativeness bound)

### 6.4 Evaluation Metrics

**Disentanglement Scores:**

1. **Specificity (S_y):** How class-specific is z_y?
   ```
   S_y = I(z_y; Y|D) / [I(z_y; Y|D) + I(z_y; D|Y)]
   Range: [0, 1], higher is better (1 = perfect class-specificity)
   ```

2. **Specificity (S_d):** How domain-specific is z_d?
   ```
   S_d = I(z_d; D|Y) / [I(z_d; D|Y) + I(z_d; Y|D)]
   Range: [0, 1], higher is better
   ```

3. **Interaction Capture (IC):** How well does z_dy capture synergy?
   ```
   IC = |I(z_dy; Y; D)| / max(I(X; Y; D), ε)
   Range: [0, 1], higher is better
   ```

4. **Residual Minimality (RM):** How minimal is z_x information?
   ```
   RM = 1 - I(z_x; Y,D) / H(X)
   Range: [0, 1], higher is better (1 = completely uninformative)
   ```

5. **Overall Partition Quality (PQ):**
   ```
   PQ = (S_y + S_d + IC + RM) / 4
   Range: [0, 1], higher is better
   Final ranking uses this composite score
   ```

### 6.5 Bootstrap Confidence Intervals

**Method:** Nonparametric bootstrap with 100 iterations (default)

**Procedure:**
1. For each bootstrap iteration i:
   - Resample n samples with replacement from dataset
   - Compute all MI quantities on bootstrap sample
2. Aggregate: Mean, 2.5th percentile, 97.5th percentile (95% CI)

**Reported Format:**
```
I(z_y; Y|D) = 2.45 [2.31, 2.59] nats
              ↑     ↑     ↑
            mean   lower  upper (95% CI)
```

### 6.6 Compatible Models

**Information-theoretic evaluation requires explicit latent decomposition:**
- ✓ **NVAE:** 4-space (z_y, z_d, z_dy, z_x) - Full evaluation
- ✓ **DIVA:** 3-space (z_y, z_d, z_x) - Evaluation without interaction term
- ✓ **AugmentedDANN:** 3-space (z_y, z_d, z_dy) - No z_x (discriminative model)
- ✗ **Baseline DANN:** Monolithic features (no decomposition)
- ✗ **IRM:** Monolithic features (no decomposition)

### 6.7 Evaluation Pipeline

**Step 1: Extract Latent Representations** (~10 mins for 12,800 samples)
```python
evaluator = MinimalInformationPartitionEvaluator(n_neighbors=7, n_bootstrap=100)
latent_data = extract_latent_representations(model, val_loader, device)

z_y = latent_data['zy'].numpy()      # (n_samples, dim_zy)
z_d = latent_data['za'].numpy()      # (n_samples, dim_zd)
z_dy = latent_data['zay'].numpy()    # (n_samples, dim_zdy) or None
z_x = latent_data['zx'].numpy()      # (n_samples, dim_zx)
y_labels = latent_data['y'].argmax(1).numpy()  # (n_samples,)
d_labels = latent_data['a'].numpy()  # (n_samples,)
```

**Step 2: Compute MI Quantities** (~30 mins with bootstrap)
```python
results = evaluator.evaluate_latent_partition(
    z_y, z_d, z_dy, z_x, y_labels, d_labels,
    compute_bootstrap=True
)
```

**Step 3: Model Comparison**
```python
# Compare multiple models
comparison = evaluator.compare_models({
    'NVAE': nvae_results,
    'DIVA': diva_results,
    'AugmentedDANN': dann_aug_results
})

# Generate visualization
visualize_all(comparison, output_dir)
```

### 6.8 Output Files

Saved to `{output_dir}/information_theoretic_analysis/`:
```
model_comparison.json               # All IT metrics for all models
nvae/it_results.json               # Detailed NVAE results
diva/it_results.json               # Detailed DIVA results
dann_augmented/it_results.json     # Detailed AugmentedDANN results

Visualizations:
  partition_quality_comparison.png      # Bar plot of PQ scores
  specificity_scores.png               # S_y and S_d comparison
  information_capture.png              # IC and RM comparison
  conditional_mi_heatmap.png           # Heatmap of all CMI values
```

**JSON Structure Example:**
```json
{
  "config": {
    "n_samples": 12800,
    "n_neighbors": 7,
    "n_bootstrap": 100,
    "dims": {"z_y": 128, "z_d": 128, "z_dy": 128, "z_x": 128}
  },
  "class_specific": {
    "I_zy_Y_given_D": {"mean": 2.45, "ci": [2.31, 2.59]},
    "I_zy_D_given_Y": {"mean": 0.12, "ci": [0.08, 0.16]}
  },
  "domain_specific": {
    "I_zd_D_given_Y": {"mean": 1.87, "ci": [1.73, 2.01]},
    "I_zd_Y_given_D": {"mean": 0.23, "ci": [0.18, 0.28]}
  },
  "interaction": {
    "I_zdy_Y_D": {"mean": 0.45, "ci": [0.38, 0.52]}
  },
  "residual": {
    "I_zx_Y_D": {"mean": 0.08, "ci": [0.05, 0.11]}
  },
  "scores": {
    "specificity_y": 0.953,
    "specificity_d": 0.891,
    "interaction_capture": 0.678,
    "residual_minimality": 0.984,
    "partition_quality": 0.877
  }
}
```

---

## 7. Summary

This experimental framework provides a comprehensive evaluation of disentangled representation learning for domain generalization. By combining traditional performance metrics, latent expressiveness probing, and rigorous information-theoretic evaluation, we can assess not only whether models generalize to new domains, but also whether they learn interpretable, theoretically grounded representations that decompose information according to the Minimal Information Partition theorem.

**Key Evaluation Axes:**
1. **Performance:** Classification accuracy on ID and OOD test sets
2. **Disentanglement:** Latent expressiveness via classifier probing
3. **Theoretical Adherence:** Information-theoretic partition quality
4. **Interpretability:** Visualization of learned representations

**Expected Model Rankings** (hypothesis):
1. **NVAE:** Best partition quality (explicit 4-space decomposition)
2. **AugmentedDANN:** Good specificity (3-space with adversarial training)
3. **DIVA:** Moderate quality (3-space without interaction term)
4. **DANN/IRM:** Not evaluated (monolithic representations)

**Computational Requirements:**
- **Training:** 1-4 hours per model (WILD on GPU)
- **Expressiveness:** ~10 minutes per model
- **IT Evaluation:** ~30 minutes per model (with bootstrap)
- **Total per model:** ~2-5 hours end-to-end

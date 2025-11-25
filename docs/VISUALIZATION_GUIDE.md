# NVAE Visualization Guide

## Executive Summary

This guide provides comprehensive documentation for all visualization functionality in the NVAE (Nouveau VAE) codebase. The project implements multiple visualization types to assess disentangled representation learning across two main datasets:

- **CRMNIST**: Colored Rotated MNIST dataset with controlled factors (digit identity, color, rotation)
- **WILD**: Histopathology dataset with medical imaging data from multiple hospitals

The visualizations help evaluate whether the model successfully separates task-relevant factors (digit identity, pathology) from domain-specific factors (rotation, hospital characteristics).

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Visualization Types](#visualization-types)
   - [Latent Space Visualizations (t-SNE)](#latent-space-visualizations-tsne)
   - [Disentanglement Analysis](#disentanglement-analysis)
   - [Reconstruction Quality Assessment](#reconstruction-quality-assessment)
   - [Conditional Generation](#conditional-generation)
   - [Statistical Analysis](#statistical-analysis)
3. [Dataset-Specific Visualizations](#dataset-specific-visualizations)
   - [CRMNIST Visualizations](#crmnist-visualizations)
   - [WILD Visualizations](#wild-visualizations)
4. [Model-Specific Visualizations](#model-specific-visualizations)
   - [NVAE (Full VAE)](#nvae-full-vae)
   - [DIVA](#diva)
   - [DANN](#dann)
   - [IRM](#irm)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Technical Appendix](#technical-appendix)

---

## Quick Reference

| Visualization Type | Location | Purpose | Output Files |
|-------------------|----------|---------|--------------|
| **t-SNE Latent Spaces** | `core/utils.py:visualize_latent_spaces` | Assess disentanglement quality via dimensionality reduction | `{model}_latent_spaces.png` |
| **Disentanglement Analysis** | `core/{DATASET}/disentanglement_visualization.py` | Show how varying each latent space affects generation | `{name}_example_{n}.png`, `{name}_summary.png` |
| **Latent Interpolation** | `core/{DATASET}/disentanglement_visualization.py:visualize_latent_interpolation` | Demonstrate smoothness of latent representations | `{name}_interpolation.png` |
| **Factor Traversal** | `core/{DATASET}/disentanglement_visualization.py:visualize_factor_traversal` | Identify which dimensions control which factors | `{name}_traversal_{space}.png` |
| **Reconstructions** | `core/{DATASET}/utils_{dataset}.py:visualize_reconstructions` | Assess reconstruction quality across domains | `{model}_epoch_{n}.png` |
| **Conditional Generation** | `core/{DATASET}/utils_{dataset}.py:visualize_conditional_generation` | Test conditional generation capabilities | `conditional_generations.png` |
| **DANN Latent Spaces** | `core/CRMNIST/dann_model.py:visualize_latent_spaces` | DANN-specific t-SNE with RGB color mapping | `dann_latent_spaces.png` |
| **IRM Latent Features** | `core/comparison/irm.py:visualize_latent_space` | IRM invariant feature visualization | `irm_latent_spaces.png` |
| **Domain Samples** | `core/{DATASET}/utils_{dataset}.py:save_domain_samples_visualization` | Verify balanced sampling across domains | `domain_samples_epoch_{n}.png` |
| **Latent Ablation** | `core/WILD/utils_wild.py:generate_images_latent` | Compare individual latent space contributions | `latent_reconstructions_{type}_{mode}.png` |

---

## Visualization Types

### Latent Space Visualizations (t-SNE)

#### Overview

t-SNE (t-Distributed Stochastic Neighbor Embedding) visualizations project high-dimensional latent spaces into 2D for visual inspection. These visualizations are crucial for assessing disentanglement quality.

#### What to Look For

**Good Disentanglement Indicators:**
- **Label-specific space (zy)**: Strong clustering by task labels (digits or pathology)
- **Domain-specific space (za)**: Strong clustering by domain (rotation or hospital)
- **Residual space (zx)**: Uniform distribution (no strong patterns)
- **Interaction space (zay)**: Organized structure showing label-domain interactions

**Poor Disentanglement Indicators:**
- Task labels clustering in domain-specific spaces (domain leakage)
- Domain labels clustering in task-specific spaces (task leakage)
- No clear separation in any space (poor disentanglement overall)

#### Implementation Details

**Location**: `core/utils.py:visualize_latent_spaces()`

**Key Features:**
- Balanced sampling ensures equal representation across all combinations
- Supports all model types (NVAE, DIVA, DANN, WILD, CRMNIST)
- Automatically detects DIVA mode (no zay space)
- Creates grid layout: `(1 + num_domains) rows × num_latent_spaces columns`

**Technical Parameters:**
```python
max_samples = 5000  # Maximum samples for visualization
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
```

#### Interpreting the Grid

Each visualization shows:
- **Row 1**: Latent spaces colored by task labels (digits/pathology)
- **Rows 2+**: Latent spaces colored by domain variables (rotation/hospital)

The arrangement allows direct comparison of how each latent space separates task vs. domain information.

---

### Disentanglement Analysis

#### Overview

Disentanglement visualizations show how varying individual latent spaces affects image generation while keeping other spaces fixed. This directly demonstrates the independence and interpretability of learned representations.

#### What Each Row Shows

1. **Original Image Row**: The base image used for analysis
2. **zy (Label-Specific) Row**: Varying zy changes task features
   - CRMNIST: Should change digit identity
   - WILD: Should change tumor/normal classification
3. **za (Domain-Specific) Row**: Varying za changes domain features
   - CRMNIST: Should change rotation angle
   - WILD: Should change hospital-specific characteristics
4. **zay (Interaction) Row** (NVAE only): Shows label-domain interactions
5. **zx (Residual) Row**: Shows style/texture variations

#### Column Structure

- **Center column**: Original latent value (0σ)
- **Other columns**: Perturbations from -1.5σ to +1.5σ

#### Location

**CRMNIST**: `core/CRMNIST/disentanglement_visualization.py`
**WILD**: `core/WILD/disentanglement_visualization.py`

**Functions:**
- `visualize_disentanglement()`: Creates per-example visualizations
- `_create_summary_visualization()`: Combines multiple examples
- `visualize_latent_interpolation()`: Shows smooth transitions
- `visualize_factor_traversal()`: Systematic dimension traversal

#### Interpretation

**Good Disentanglement:**
- Each row shows independent control of specific factors
- Varying zy changes only task-relevant features
- Varying za changes only domain-relevant features
- Changes are smooth and semantically meaningful

**Poor Disentanglement:**
- Rows show mixed or coupled changes
- Varying one space affects factors it shouldn't control
- Changes are abrupt or nonsensical

---

### Reconstruction Quality Assessment

#### Overview

Reconstruction visualizations compare original images with model reconstructions, organized by domain. This assesses how well the model preserves information through its encoding-decoding process.

#### Organization

**CRMNIST:**
- 6 rotation domains (0°, 10°, 20°, 30°, 40°, 50°)
- 10 samples per domain
- 2 rows per domain: original (top), reconstruction (bottom)

**WILD:**
- 5 hospital domains
- 10 samples per hospital
- 2 rows per domain: original (top), reconstruction (bottom)

#### What to Assess

1. **Task Preservation**: Do reconstructions maintain digit/pathology identity?
2. **Domain Preservation**: Are rotation/hospital characteristics preserved?
3. **Image Quality**: Are reconstructions visually coherent?
4. **Cross-Domain Consistency**: Is quality consistent across all domains?

#### Location

**CRMNIST**: `core/CRMNIST/utils_crmnist.py:visualize_reconstructions()`
**WILD**: `core/WILD/utils_wild.py:visualize_reconstructions()`

**Output Pattern**: `{model_name}_epoch_{epoch}.png`

---

### Conditional Generation

#### Overview

Conditional generation tests the model's ability to generate samples given specific label and domain conditions. This evaluates control over the generative process.

#### Structure

**CRMNIST:**
- 10 rows (digits 0-9)
- 5 samples per row
- All generated with same label

**WILD:**
- 2 rows (Normal, Tumor)
- 5 columns (Hospitals 1-5)
- Each cell conditioned on specific label-hospital pair

#### Evaluation Criteria

1. **Label Consistency**: Do all samples with same label show correct class?
2. **Diversity**: Do samples show appropriate within-class variation?
3. **Domain Control**: Can model generate for all domain values?
4. **Quality**: Are generated images realistic and coherent?

#### Location

**CRMNIST**: `core/CRMNIST/utils_crmnist.py:visualize_conditional_generation()`
**WILD**: `core/WILD/utils_wild.py:visualize_conditional_generation()`

**Output**: `conditional_generations.png`

---

### Statistical Analysis

#### Overview

Quantitative metrics complement visual assessment of disentanglement quality.

#### Available Analyses

1. **Correlation Heatmaps**: Show correlations between latent spaces
2. **Mutual Information**: Quantify information sharing between spaces
3. **Label Prediction Accuracy**: Test if latent spaces encode expected information
4. **Expressiveness Metrics**: Quantify how well each space captures factors

#### Location

**WILD**: `core/WILD/disentanglement_analysis.py:analyze_disentanglement()`
**CRMNIST**: `core/CRMNIST/latent_expressiveness.py`

**Outputs:**
- `correlation_heatmap.png`
- `label_prediction_accuracy.png`
- `builtin_predictor_accuracy.png`
- `comprehensive_expressiveness_comparison.png`

---

## Dataset-Specific Visualizations

### CRMNIST Visualizations

#### Dataset Characteristics

- **Input**: 28×28 colored MNIST digits
- **Factors of Variation**:
  - **Task Factor (y)**: Digit identity (0-9)
  - **Domain Factor (r)**: Rotation angle (0°, 10°, 20°, 30°, 40°, 50°)
  - **Spurious Factor (c)**: Color (Blue, Green, Yellow, Cyan, Magenta, Orange, Red)

#### Latent Spaces

- **zy**: Should encode digit identity
- **za**: Should encode rotation/domain
- **zay** (NVAE only): Digit-rotation interactions
- **zx**: Residual style/texture

#### Key Visualizations

1. **t-SNE Latent Spaces** (`core/utils.py`)
   - Grid shows 3-4 latent spaces
   - Row 1: Colored by digit
   - Row 2: Colored by rotation
   - Row 3: Colored by color

2. **Disentanglement Analysis** (`core/CRMNIST/disentanglement_visualization.py`)
   - Shows how varying each space affects digit, rotation, and color
   - Multiple examples for robustness assessment

3. **DANN-Specific Visualization** (`core/CRMNIST/dann_model.py`)
   - 3×3 grid for DANN architecture
   - Special RGB color mapping in bottom row
   - Shows class-specific (zy), domain-specific (zd), and interaction (zdy) spaces

4. **Expressiveness Comparison** (`core/CRMNIST/compare_all_expressiveness.py`)
   - Quantitative comparison across models
   - Shows which models best separate factors

#### Domain Sample Visualization

**Location**: `core/CRMNIST/utils_crmnist.py:save_domain_samples_visualization()`

Shows:
- Top row: Rare red images
- Following 6 rows: Rotation domains (0°-50°)
- Confirms balanced sampling

---

### WILD Visualizations

#### Dataset Characteristics

- **Input**: 96×96 histopathology tissue images
- **Factors of Variation**:
  - **Task Factor (y)**: Pathology (0=Normal, 1=Tumor)
  - **Domain Factor (hospital_id)**: Hospital (1-5)
  - **Implicit Factors**: Staining protocol, scanner, imaging conditions

#### Latent Spaces

- **zy**: Should encode tumor/normal pathology
- **za**: Should encode hospital-specific characteristics
- **zay** (NVAE only): Pathology-hospital interactions
- **zx**: Texture and fine-grained details

#### Medical Context

The WILD visualizations are designed with medical imaging in mind:
- **Pathology Preservation**: Critical for diagnostic utility
- **Domain Invariance**: Ensures models generalize across hospitals
- **Staining Variations**: Hospital-specific staining protocols captured in za
- **Interpretability**: Important for clinical trust and validation

#### Key Visualizations

1. **t-SNE Latent Spaces** (`core/utils.py`)
   - Row 1: Colored by pathology (Blue=Normal, Red=Tumor)
   - Row 2: Colored by hospital
   - Colormap 'RdYlBu' chosen for medical interpretability

2. **Histopathology Disentanglement** (`core/WILD/disentanglement_visualization.py`)
   - Shows tissue-specific and hospital-specific variations
   - Titles explicitly mention medical context

3. **Latent Ablation Study** (`core/WILD/utils_wild.py:generate_images_latent()`)
   - Shows reconstructions with/without specific latent spaces
   - Two modes: "only" (use only this space) and "without" (use all except this)
   - Helps identify which spaces are critical for pathology vs. style

4. **Statistical Disentanglement Analysis** (`core/WILD/disentanglement_analysis.py`)
   - Correlation heatmaps show independence of spaces
   - Mutual information quantifies information sharing
   - Label prediction tests if spaces encode expected factors

#### Hospital Domain Visualization

**Location**: `core/WILD/utils_wild.py:save_domain_samples_visualization()`

Shows:
- 5 rows (one per hospital)
- 10 samples per hospital
- Confirms balanced hospital representation

---

## Model-Specific Visualizations

### NVAE (Full VAE)

#### Architecture

4 latent spaces: zy, za, zay, zx

#### Visualization Grid

- **4 columns**: One per latent space
- **Multiple rows**: One for task labels, one per domain variable

#### Interpretation

The full VAE with zay space can capture complex label-domain interactions. Look for:
- **zy**: Pure task clustering
- **za**: Pure domain clustering
- **zay**: Organized interaction patterns
- **zx**: Residual variance

### DIVA

#### Architecture

3 latent spaces: zy, za, zx (no zay)

#### Visualization Grid

- **3 columns**: zy, za, zx
- **Simpler architecture**: Forces cleaner separation

#### Interpretation

DIVA enforces stronger independence between zy and za. Expect:
- **More distinct separation** between task and domain
- **Stronger clustering** in zy and za
- **Less interaction effects** (no zay)

### DANN

#### Architecture

Domain Adversarial Neural Network with:
- **zy**: Class-specific features
- **zd**: Domain-specific features (learned adversarially)
- **zdy**: Domain-class interaction

#### Unique Visualization Features

**Location**: `core/CRMNIST/dann_model.py:visualize_latent_spaces()`

- **3×3 grid structure**:
  - Row 1: Colored by digit
  - Row 2: Colored by rotation
  - Row 3: Colored by RGB (actual image colors)
- **RGB color mapping**: Bottom row uses actual RGB values instead of categorical colors
- **Balanced sampling**: Ensures 200 samples per digit×rotation×color combination

#### Interpretation

DANN uses adversarial training to make zy domain-invariant. Look for:
- **zy**: Task clustering, uniform domain distribution
- **zd**: Domain clustering, potentially mixed task distribution
- **zdy**: Interaction patterns between task and domain

**Adversarial Success Indicators:**
- zy shows strong digit clustering but no rotation/color clustering
- zd shows rotation/color patterns
- Clear separation between what zy and zd capture

### IRM

#### Architecture

Invariant Risk Minimization learns features that enable domain-invariant prediction by minimizing gradient variance across environments.

#### Visualization Structure

**Location**: `core/comparison/irm.py:visualize_latent_space()`

- **1×3 layout**: Simpler than other models
- **Three plots**:
  1. Colored by task classes (digits)
  2. Colored by colors (spurious feature)
  3. Colored by domains (rotation angles)

#### Interpretation

IRM directly learns invariant features. Success indicators:
- **Strong task clustering**: Digits well-separated
- **Uniform color distribution**: No color clustering (good invariance)
- **Uniform domain distribution**: No rotation clustering (primary goal)

**Failure indicators:**
- Clustering by color or rotation indicates IRM penalty is insufficient
- Weak task clustering indicates features aren't task-relevant

---

## Interpretation Guidelines

### General Principles

1. **Disentanglement ≠ Reconstruction Quality**
   - A model can reconstruct perfectly but still have entangled representations
   - Focus on separation in latent spaces, not just image quality

2. **Context Matters**
   - CRMNIST: Rotation is domain, digit is task
   - WILD: Hospital is domain, pathology is task

3. **Multiple Views Required**
   - t-SNE alone can be misleading
   - Combine with disentanglement analysis and quantitative metrics

### t-SNE Interpretation Checklist

**For Label-Specific Space (zy):**
- [ ] Clear clustering by task labels (digits/pathology)?
- [ ] Minimal clustering by domain labels (rotation/hospital)?
- [ ] Cluster separation is visually distinct?

**For Domain-Specific Space (za):**
- [ ] Clear clustering by domain labels (rotation/hospital)?
- [ ] Task labels distributed across domain clusters?
- [ ] Domain effects captured (rotation angles ordered, hospital differences visible)?

**For Residual Space (zx):**
- [ ] Relatively uniform distribution?
- [ ] No strong clustering patterns?
- [ ] Minor texture/style variations only?

### Disentanglement Analysis Checklist

**Independence:**
- [ ] Varying zy changes only task features?
- [ ] Varying za changes only domain features?
- [ ] Changes are smooth and interpretable?

**Completeness:**
- [ ] Can zy represent all task classes?
- [ ] Can za represent all domain values?
- [ ] No factors missing from representation?

**Consistency:**
- [ ] Same patterns across multiple examples?
- [ ] Behavior stable across domains?
- [ ] Factor control is predictable?

### Medical Imaging Specific (WILD)

**Pathology Preservation:**
- [ ] Reconstructions maintain tumor/normal identity?
- [ ] zy clusters clearly separate pathology?
- [ ] Pathology features not leaked to za?

**Hospital Invariance:**
- [ ] Model performs across all hospitals?
- [ ] za captures hospital-specific style?
- [ ] Hospital information doesn't dominate zy?

**Clinical Interpretability:**
- [ ] Latent manipulations produce biologically plausible changes?
- [ ] Domain adaptation doesn't compromise diagnostic features?
- [ ] Visualizations aid in understanding model decisions?

---

## Technical Appendix

### t-SNE Configuration

#### Default Parameters

```python
TSNE(
    n_components=2,          # 2D projection
    random_state=42,         # Reproducibility
    n_jobs=-1                # Use all CPU cores
)
```

#### Alternative Configurations

**For large datasets** (>2000 samples):
```python
TSNE(
    n_components=2,
    random_state=42,
    n_iter=500,              # Fewer iterations
    learning_rate='auto',    # Adaptive learning rate
    init='pca',              # PCA initialization
    method='barnes_hut'      # Faster approximation
)
```

**For small datasets** (<500 samples):
```python
TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(30, n_samples//4)  # Adjusted perplexity
)
```

### Figure Sizes and DPI

| Visualization Type | Figure Size | DPI | Notes |
|-------------------|-------------|-----|-------|
| t-SNE Latent Spaces | 5×num_cols × 5×num_rows inches | 100 | Large grid needs space |
| Disentanglement | 2×num_variations × 2×(num_spaces+1) inches | 150 | Medium resolution |
| Reconstruction | 20 × 4×num_domains inches | 100 | Wide layout for samples |
| WILD Conditional | 25 × 8 inches | 300 | High quality for medical |
| IRM Visualization | 18 × 6 inches | 300 | Publication quality |

### Color Schemes

#### Categorical Colormaps

- **tab10**: Digits (0-9), Hospitals (if ≤10)
- **Set1**: Color labels in CRMNIST
- **Set2**: Rotation labels in some contexts
- **RdYlBu**: WILD pathology (Red=Tumor, Blue=Normal)

#### Custom RGB Mapping (DANN)

```python
color_mappings = {
    0: [0, 0, 1],      # Blue
    1: [0, 1, 0],      # Green
    2: [1, 1, 0],      # Yellow
    3: [0, 1, 1],      # Cyan
    4: [1, 0, 1],      # Magenta
    5: [1, 0.5, 0],    # Orange
    6: [1, 0, 0]       # Red
}
```

### Balanced Sampling Strategy

#### Purpose
Ensure equal representation of all factor combinations in visualizations.

#### Implementation

**Location**: `core/utils.py:balanced_sample_for_visualization()`

**For CRMNIST:**
```python
target_samples_per_combination = 50
# Samples 50 instances of each (digit × color × rotation) combination
# Maximum: min(200, max_samples // 30)
```

**For WILD:**
```python
target_samples_per_combination = 50
# Samples 50 instances of each (label × hospital) combination
```

#### Benefits
- Prevents bias toward majority combinations
- Ensures all factors visible in visualization
- Makes comparisons fair across domains

#### Memory Management
- Processes batches with limit (max_batches_per_pass=50)
- Moves tensors to CPU between batches
- Clears GPU cache periodically
- Falls back to PCA if t-SNE fails

### Output Directory Structure

```
results/
├── all_models/
│   └── {experiment_name}/
│       ├── latent_spaces/
│       │   ├── nvae_latent_spaces.png
│       │   ├── diva_latent_spaces.png
│       │   ├── dann_latent_spaces.png
│       │   ├── irm_latent_spaces.png
│       │   ├── crmnist_latent_epoch_001.png
│       │   └── wild_latent_epoch_001.png
│       ├── reconstructions/
│       │   ├── nvae_epoch_1.png
│       │   ├── domain_samples_epoch_1.png
│       │   └── conditional_generations.png
│       ├── disentanglement/
│       │   ├── disentanglement_example_1.png
│       │   ├── disentanglement_summary.png
│       │   ├── interpolation.png
│       │   └── traversal_{space}.png
│       └── analysis/
│           ├── correlation_heatmap.png
│           ├── label_prediction_accuracy.png
│           └── expressiveness_comparison.png
```

### File Naming Conventions

- **Per-epoch visualizations**: `{model}_latent_epoch_{epoch:03d}.png`
- **Reconstructions**: `{model_name}_epoch_{epoch}.png`
- **Conditional generation**: `conditional_generations.png`
- **Domain samples**: `domain_samples_epoch_{epoch}.png`
- **Disentanglement**: `{name}_example_{n}.png`, `{name}_summary.png`
- **Interpolation**: `{name}_interpolation.png`
- **Traversal**: `{name}_traversal_{space_name}.png`
- **Analysis**: Descriptive names like `correlation_heatmap.png`

### Matplotlib Configuration

#### Font Sizes

```python
# Overall title
fontsize=13-16, fontweight='bold'

# Subplot titles
fontsize=10-11

# Axis labels
fontsize=9-10

# Legends
fontsize=8-9
```

#### Layout Adjustment

```python
# For figures with suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96-0.98])

# For figures without suptitle
plt.tight_layout()
```

### Common Issues and Solutions

#### Issue: t-SNE takes too long

**Solutions:**
1. Reduce max_samples (5000 → 1000)
2. Use barnes_hut method
3. Reduce n_iter (1000 → 500)
4. Fall back to PCA

#### Issue: Visualization titles overlap

**Solutions:**
1. Adjust suptitle y parameter (y=0.98 → y=1.02)
2. Use tight_layout with rect parameter
3. Reduce fontsize
4. Increase figure size

#### Issue: Colors hard to distinguish

**Solutions:**
1. Use colorblind-friendly colormaps
2. Increase alpha for better contrast
3. Add explicit legends with labels
4. Use different marker sizes

#### Issue: Memory errors during visualization

**Solutions:**
1. Process in smaller batches
2. Clear GPU cache (torch.cuda.empty_cache())
3. Move tensors to CPU earlier
4. Reduce max_samples
5. Use PCA instead of t-SNE

---

## Frequently Asked Questions

### General Questions

**Q: Why use t-SNE instead of PCA?**

A: t-SNE preserves local structure better than PCA, making cluster boundaries more visible. However, t-SNE:
- Doesn't preserve global distances
- Can create misleading clusters
- Is computationally expensive

Use both for comprehensive analysis.

**Q: How many samples needed for reliable visualization?**

A: Minimum 1000-2000 samples for t-SNE. For CRMNIST with 10 digits × 6 rotations × 7 colors = 420 combinations, aim for 50+ per combination (21,000+ total samples), but practical limit is ~5000 with balanced sampling.

**Q: Can I compare t-SNE plots across different runs?**

A: No. t-SNE is non-deterministic (despite random_state) and non-parametric. Each run produces different coordinates. Compare cluster structure, not absolute positions.

### Dataset-Specific Questions

**Q: Why are red images shown separately in CRMNIST?**

A: Red is a rare color in the training distribution (7% compared to 15% for others). Showing red images separately confirms the model handles rare cases.

**Q: Why use RdYlBu colormap for WILD pathology?**

A: Medical convention: Red often indicates abnormal/tumor, Blue indicates normal. This colormap is also colorblind-friendly.

**Q: How do I know if hospital characteristics are problematic?**

A: If zy (pathology space) clusters by hospital rather than by tumor/normal, the model has learned hospital-specific diagnostic features rather than generalizable pathology features.

### Model-Specific Questions

**Q: When should I use DIVA vs. full VAE?**

A:
- **DIVA**: Enforces strict independence, better for clean domain adaptation
- **VAE with zay**: Captures interactions, better when task and domain genuinely interact

**Q: Why does DANN have RGB color visualization?**

A: DANN learns to be invariant to color (domain shift). The RGB visualization shows whether color information leaked into class-specific features.

**Q: What IRM penalty weight should I use?**

A: Start with 1e4 (default). If domains still cluster, increase. If task performance degrades, decrease. The visualization helps tune this hyperparameter.

### Technical Questions

**Q: Why use balanced sampling instead of random sampling?**

A: Random sampling may oversample common combinations and undersample rare ones, making visualization biased and some factors invisible.

**Q: Can I use these visualizations for other datasets?**

A: Yes, but you'll need to:
1. Adapt domain definitions
2. Update color schemes
3. Modify label names
4. Adjust interpretation guidelines

**Q: How do I save visualizations in multiple formats?**

A: Modify save lines:
```python
plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
```

---

## Conclusion

This guide provides comprehensive documentation for all visualization functionality in the NVAE codebase. Visualizations are essential tools for:

1. **Assessing disentanglement quality**
2. **Debugging model training**
3. **Comparing model architectures**
4. **Communicating results**
5. **Building trust in model behavior**

For medical imaging applications (WILD), visualizations also serve critical functions in:
- Validating clinical utility
- Ensuring cross-hospital generalization
- Building interpretable diagnostic aids
- Meeting regulatory requirements for explainability

Remember:
- **Use multiple visualization types** for comprehensive assessment
- **Combine qualitative and quantitative** evaluation
- **Interpret in context** of your specific task and dataset
- **Compare across models** to identify best approaches

For questions, issues, or contributions, please refer to the main project README or open an issue on GitHub.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Maintainer**: NVAE Development Team

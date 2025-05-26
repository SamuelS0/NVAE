# Disentanglement Visualization for CRMNIST VAE

This module provides comprehensive visualization tools for analyzing the quality of disentangled representations learned by the CRMNIST VAE model.

## Overview

The disentanglement visualization module includes three main functions:

1. **`visualize_disentanglement`**: Shows how changing individual latent spaces affects image generation
2. **`visualize_latent_interpolation`**: Demonstrates smooth interpolation between different images in latent space
3. **`visualize_factor_traversal`**: Systematically varies individual latent dimensions to identify factor control

## Files

- `disentanglement_visualization.py`: Main visualization functions
- `example_disentanglement_usage.py`: Example usage and evaluation guidelines
- `README_disentanglement.md`: This documentation file

## Quick Start

```python
from disentanglement_visualization import visualize_disentanglement
from torch.utils.data import DataLoader

# After training your model
model.eval()

# Run disentanglement analysis
visualize_disentanglement(
    model=model,
    dataloader=test_dataloader,
    device=device,
    save_path='./results/disentanglement.png',
    num_variations=7,
    num_examples=3
)
```

## Function Details

### `visualize_disentanglement(model, dataloader, device, save_path=None, num_variations=7, num_examples=3)`

**Purpose**: Demonstrates how well the model has learned to disentangle different factors of variation.

**What it does**:
- Selects diverse base examples from the dataset
- For each example, systematically varies each latent space (zy, za, zay, zx) while keeping others fixed
- Shows how each latent space affects different aspects of the generated image
- Creates both individual example visualizations and a summary grid

**Expected Results for Good Disentanglement**:
- **zy (Label-specific)**: Should change digit identity while preserving rotation and color
- **za (Domain-specific)**: Should change rotation/domain while preserving digit and color
- **zay (Domain-Label interaction)**: Should show interaction effects between digit and domain
- **zx (Residual)**: Should change style/noise without affecting semantic content

**Output Files**:
- `{save_path}_example_1.png`, `{save_path}_example_2.png`, etc.: Individual example analyses
- `{save_path}_summary.png`: Compact summary showing all examples

### `visualize_latent_interpolation(model, dataloader, device, save_path=None, num_steps=7)`

**Purpose**: Shows how smoothly the model can interpolate between different images in latent space.

**What it does**:
- Takes two diverse examples from the dataset
- Interpolates between them in each latent space separately
- Shows whether transitions are smooth and meaningful

**Expected Results**:
- Smooth, realistic transitions between source and target images
- Each latent space should show different types of changes during interpolation
- No abrupt jumps or unrealistic intermediate states

**Output Files**:
- `{save_path}_interpolation.png`: Interpolation visualization

### `visualize_factor_traversal(model, device, save_path=None, num_steps=7)`

**Purpose**: Identifies which individual dimensions within each latent space control which factors.

**What it does**:
- Samples a base point from the prior distribution
- Systematically varies individual dimensions within each latent space
- Shows the effect of changing each dimension from -3σ to +3σ

**Expected Results**:
- Individual dimensions should control specific, interpretable factors
- Changes should be smooth and consistent
- Different dimensions should control different aspects

**Output Files**:
- `{save_path}_traversal_zy.png`: Factor traversal for zy space
- `{save_path}_traversal_za.png`: Factor traversal for za space
- `{save_path}_traversal_zay.png`: Factor traversal for zay space (if not DIVA mode)
- `{save_path}_traversal_zx.png`: Factor traversal for zx space

## Integration with Training Pipeline

### Option 1: Add to existing evaluation

```python
# In your training script, after each epoch or at the end
if epoch % 10 == 0:  # Every 10 epochs
    from disentanglement_visualization import visualize_disentanglement
    
    visualize_disentanglement(
        model=model,
        dataloader=val_dataloader,
        device=device,
        save_path=f'./results/epoch_{epoch}_disentanglement.png'
    )
```

### Option 2: Comprehensive analysis script

```python
from example_disentanglement_usage import run_disentanglement_analysis

# After training is complete
run_disentanglement_analysis(
    model=model,
    dataloader=test_dataloader,
    device=device,
    output_dir='./final_disentanglement_analysis'
)
```

## Evaluation Guidelines

### What to Look For

**Good Disentanglement Indicators**:
1. **Factor Isolation**: Each latent space affects only its intended factors
2. **Smooth Transitions**: Changes are gradual and realistic
3. **Semantic Preservation**: Non-target factors remain unchanged
4. **Interpretability**: Effects are visually clear and meaningful

**Poor Disentanglement Indicators**:
1. **Factor Entanglement**: Multiple factors change when varying one latent space
2. **Abrupt Changes**: Unrealistic jumps or artifacts
3. **Mode Collapse**: Limited diversity in generated images
4. **Unclear Effects**: Changes are subtle or hard to interpret

### CRMNIST-Specific Expectations

For the CRMNIST dataset, expect:

- **zy**: Controls digit identity (0-9)
- **za**: Controls rotation angle (0°, 15°, 30°, 45°, 60°, 75°)
- **Color**: Should be controlled by domain-specific factors
- **zx**: Handles residual style variations

## Technical Details

### Dependencies

- PyTorch
- matplotlib
- numpy
- tqdm

### Performance Considerations

- The functions automatically select diverse examples to ensure representative analysis
- Visualizations are saved at 150 DPI for good quality while maintaining reasonable file sizes
- Progress bars show computation status for longer operations

### Customization

All functions accept parameters to customize the analysis:

- `num_variations`: Number of variations to show per latent space
- `num_examples`: Number of base examples to analyze
- `num_steps`: Number of interpolation/traversal steps
- `save_path`: Where to save the visualizations

### Memory Usage

The functions are designed to be memory-efficient:
- Use `torch.no_grad()` to prevent gradient computation
- Process examples one at a time to avoid large batch memory usage
- Automatically handle device placement

## Troubleshooting

**Common Issues**:

1. **"Not enough diverse examples"**: The dataset might be too small or not diverse enough. The function will fall back to using the first available examples.

2. **Memory errors**: Reduce `num_examples` or `num_variations` if running out of memory.

3. **Poor visualizations**: This might indicate the model hasn't learned good disentangled representations. Consider:
   - Training for more epochs
   - Adjusting β parameters in the loss function
   - Checking if the model architecture is appropriate

4. **File not found errors**: Ensure the output directory exists or the function will create it automatically.

## Example Output Interpretation

When you run the visualizations, you should see:

1. **Individual Example Analysis**: Detailed view of how each latent space affects specific examples
2. **Summary Grid**: Compact overview of all examples showing consistent patterns
3. **Interpolation Results**: Smooth transitions demonstrating latent space structure
4. **Factor Traversal**: Systematic exploration of individual latent dimensions

Good results will show clear, interpretable, and consistent effects across all visualizations. 
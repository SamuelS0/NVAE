# DIVA Model Visualization Guide

## Overview
DIVA (Domain Invariant Variational Autoencoders) is implemented as a mode within the VAE model. When `diva=True`, the model doesn't use the `zay` (domain-label interaction) latent space.

## Key Differences from NVAE

| Aspect | NVAE | DIVA |
|--------|------|------|
| **Latent Spaces** | 4 spaces: `zy`, `za`, `zay`, `zx` | 3 spaces: `zy`, `za`, `zx` |
| **Domain-Label Interaction** | ✅ Has `zay` | ❌ No `zay` |
| **Sampling** | (digit, color, rotation) | (digit, color, rotation) |
| **Total Combinations** | ~21,000 | ~21,000 |

## Usage Options

### Option 1: Direct Generic Function
```python
from core.utils import balanced_sample_for_visualization

# Create DIVA model
diva_model = VAE(
    class_map=class_map,
    zy_dim=32, zx_dim=32, za_dim=32,
    y_dim=10, a_dim=6,
    diva=True  # Enable DIVA mode
)

# Use the generic sampling function
features_dict, labels_dict, stats = balanced_sample_for_visualization(
    model=diva_model,
    dataloader=dataloader,
    device=device,
    model_type="diva",  # Specify DIVA type
    max_samples=5000,
    target_samples_per_combination=50
)

print(f"DIVA features:")
print(f"  zy: {features_dict['zy'].shape}")     # Label-specific
print(f"  za: {features_dict['za'].shape}")     # Domain-specific  
print(f"  zay: {features_dict['zay']}")         # None (DIVA mode)
print(f"  zx: {features_dict['zx'].shape}")     # Residual
```

### Option 2: DIVA-Specific Method
```python
# Use the dedicated DIVA visualization method
diva_model.visualize_latent_spaces_diva(
    dataloader=dataloader,
    device=device,
    save_path="diva_latent_spaces.png",
    max_samples=5000
)
```

### Option 3: Auto-Detection
```python
# Automatically detect if model is in DIVA mode
if diva_model.is_diva_mode():
    model_type = "diva"
    print("Using DIVA mode (3 latent spaces)")
else:
    model_type = "nvae" 
    print("Using NVAE mode (4 latent spaces)")

# Use appropriate visualization
features_dict, labels_dict, stats = balanced_sample_for_visualization(
    model=diva_model,
    dataloader=dataloader,
    device=device,
    model_type=model_type,
    max_samples=5000
)
```

### Option 4: Unified VAE Method
```python
# The regular visualize_latent_spaces method now works for both NVAE and DIVA
diva_model.visualize_latent_spaces(
    dataloader=dataloader,
    device=device,
    save_path="latent_spaces.png"
)
# Automatically handles DIVA vs NVAE mode
```

## Expected Output

### DIVA Latent Spaces:
- **zy (Label-specific)**: Should cluster by digit (0-9)
- **za (Domain-specific)**: Should cluster by rotation/domain
- **zx (Residual)**: Should capture remaining variation

### Visualization Layout:
```
┌─────────────────────────────────────────┐
│  zy (Label)    za (Domain)    zx (Residual) │
├─────────────────────────────────────────┤
│ Colored by Digits (Row 1)               │
│ Colored by Rotations (Row 2)            │ 
│ Colored by Colors (Row 3)               │
└─────────────────────────────────────────┘
```

## Sampling Statistics
- **Total samples**: ~21,000 (10 digits × 7 colors × 6 rotations × 50 each)
- **Balanced sampling**: Equal representation across all combinations
- **Memory efficient**: Only processes selected samples

## Example Training Script Integration
```python
# In your training script
def train_diva_model(args, dataloader):
    # Create DIVA model
    model = VAE(
        class_map=spec_data['class_map'],
        zy_dim=args.zy_dim,
        zx_dim=args.zx_dim,
        za_dim=args.za_dim,
        y_dim=spec_data['num_y_classes'],
        a_dim=spec_data['num_r_classes'],
        diva=True  # Enable DIVA mode
    )
    
    # Train model...
    
    # Visualize results
    model.visualize_latent_spaces_diva(
        dataloader=test_loader,
        device=device,
        save_path=f"{args.out}/diva_latent_spaces.png"
    )
```

## Benefits of Unified Function
✅ **Consistent sampling** across NVAE and DIVA  
✅ **Same visualization format** for easy comparison  
✅ **Automatic mode detection** with `is_diva_mode()`  
✅ **Memory efficient** with balanced sampling  
✅ **Configurable parameters** for different use cases 
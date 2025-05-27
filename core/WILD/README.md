# WILD VAE: Domain-Adaptive Variational Autoencoder for Medical Images

A PyTorch implementation of a Variational Autoencoder (VAE) designed for domain adaptation on the **Camelyon17-WILDS** dataset. This system learns disentangled representations to handle domain shifts between different hospitals while performing tumor detection in histopathology images.

## 🎯 Overview

This implementation addresses the challenge of **domain shift** in medical imaging, where models trained on data from one hospital may perform poorly on data from another due to differences in imaging equipment, protocols, and patient populations. Our VAE learns to disentangle:

- **Label-specific features** (tumor vs. normal tissue characteristics)
- **Domain-specific features** (hospital-specific imaging styles)
- **Content features** (image-specific details)
- **Label-domain interactions** (how diseases manifest differently across domains)

## 📊 Dataset: Camelyon17-WILDS

- **Task**: Binary classification (tumor vs. normal tissue)
- **Domains**: 5 different hospitals with varying imaging conditions
- **Images**: Histopathology patches from lymph node sections
- **Domain Split**:
  - Training hospitals: [0, 3, 4]
  - ID validation hospitals: [0, 3, 4]
  - OOD validation hospital: [1]
  - Test hospital: [2]

## 🏗️ Architecture

### Disentangled Latent Spaces

Our VAE learns four distinct latent representations:

```
z = [zy, zx, zay, za]
```

- **`zy`**: Label-specific latent space (captures tumor/normal characteristics)
- **`zx`**: Content latent space (captures image-specific details)
- **`zay`**: Label-domain interaction space (captures class-domain relationships)
- **`za`**: Domain-specific latent space (captures hospital-specific style)

### Loss Function

```
L_total = λ_recon × L_recon + β₁ × KL(zy) + β₂ × KL(zx) + β₃ × KL(zay) + β₄ × KL(za) + α₁ × L_y + α₂ × L_a
```

Where:
- `L_recon`: Reconstruction loss (MSE)
- `KL(·)`: KL divergence terms for each latent space
- `L_y`: Label classification loss
- `L_a`: Domain classification loss

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install torch torchvision wilds matplotlib numpy tqdm pillow
```

### Basic Usage

```bash
# Train VAE on low-resolution images
python run_wild.py --out results_low_res/ --batch_size 128 --model vae --resolution low --epochs 50

# Train VAE on high-resolution images
python run_wild.py --out results_high_res/ --batch_size 64 --model vae --resolution high --epochs 100
```

### Key Arguments

- `--model`: Model type (`vae` or `diva`)
- `--resolution`: Image resolution (`high` for 448×448, `low` for 96×96)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 0.0001)
- `--out`: Output directory for results

### Latent Space Configuration

```bash
# Configure latent dimensions
python run_wild.py \
    --zy_dim 128 \      # Label latent dimension
    --zx_dim 128 \      # Content latent dimension
    --zay_dim 64 \      # Label-domain interaction dimension
    --za_dim 64 \       # Domain latent dimension
    --out results/
```

### Loss Weight Tuning

```bash
# Adjust loss component weights
python run_wild.py \
    --beta_1 1.0 \      # Weight for KL(zy)
    --beta_2 1.0 \      # Weight for KL(zx)
    --beta_3 1.0 \      # Weight for KL(zay)
    --beta_4 1.0 \      # Weight for KL(za)
    --alpha_1 1.0 \     # Weight for label classification
    --alpha_2 1.0 \     # Weight for domain classification
    --recon_weight 1.0  # Weight for reconstruction loss
```

## 📁 File Structure

```
WILD/
├── run_wild.py          # Main training script
├── model_wild.py        # VAE architecture definitions
├── trainer.py           # Training orchestration
├── train.py            # Training function wrapper
├── test.py             # Model evaluation
├── utils_wild.py       # Utility functions
├── data/               # Data directory
└── README.md           # This file
```

### Core Components

- **`run_wild.py`**: Main entry point with argument parsing and experiment orchestration
- **`model_wild.py`**: VAE and VAE_LowRes classes with disentangled architecture
- **`trainer.py`**: WILDTrainer class with early stopping and monitoring
- **`utils_wild.py`**: Data preparation, visualization, and analysis utilities

## 📈 Output Structure

Training generates the following outputs:

```
results/
├── models/
│   ├── model_best.pt                    # Best model checkpoint
│   └── model_checkpoint_epoch_X.pt      # Final model checkpoint
├── reconstructions/
│   ├── train_epoch_X.png                # Training reconstructions
│   ├── val_epoch_X.png                  # Validation reconstructions
│   └── test_reconstructions.png         # Test reconstructions
├── latent_recon/
│   ├── latent_reconstructions_id_val_only.png    # Individual latent contributions
│   └── latent_reconstructions_id_val_without.png # Ablation analysis
└── results.json                         # Training metrics and statistics
```

## 🔍 Analysis Features

### 1. Reconstruction Visualization
- Domain-organized reconstruction grids
- Original vs. reconstructed image comparisons
- Label and domain information overlay

### 2. Latent Space Analysis
- **Individual latent contributions**: See what each latent space captures
- **Ablation studies**: Understand the effect of removing specific latent components
- **Conditional generation**: Generate samples conditioned on class and domain

### 3. Training Monitoring
- Real-time loss tracking
- Early stopping with patience
- Automatic best model saving
- Comprehensive metrics calculation

## 🎛️ Advanced Configuration

### Multi-Resolution Training

```bash
# High-resolution training (448×448)
python run_wild.py --resolution high --batch_size 32

# Low-resolution training (96×96) 
python run_wild.py --resolution low --batch_size 128
```

### Validation Strategy

```bash
# Use in-domain validation
python run_wild.py --val_type id_val

# Use out-of-domain validation
python run_wild.py --val_type val
```

## 📊 Evaluation Metrics

The system tracks multiple metrics:

- **Reconstruction Quality**: MSE between original and reconstructed images
- **Label Classification**: Accuracy on tumor vs. normal classification
- **Domain Classification**: Accuracy on hospital identification
- **Total Loss**: Combined loss with all components

## 🔬 Research Applications

This implementation is valuable for:

- **Medical image analysis** across different institutions
- **Domain adaptation** research in computer vision
- **Disentangled representation learning** studies
- **Fairness in medical AI** by understanding domain biases
- **Transfer learning** between medical imaging domains

## 🛠️ Customization

### Adding New Models

Extend the `NModule` base class in `model_wild.py`:

```python
class CustomVAE(NModule):
    def __init__(self, ...):
        super().__init__()
        # Your architecture here
    
    def forward(self, a, x, y):
        # Your forward pass here
        pass
    
    def loss_function(self, a, x, y):
        # Your loss computation here
        pass
```

### Custom Visualizations

Add new visualization functions to `utils_wild.py`:

```python
def custom_visualization(model, data, output_dir):
    # Your visualization code here
    pass
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- wilds
- matplotlib
- numpy
- tqdm
- Pillow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the NVAE repository. Please refer to the main repository for licensing information.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
    title={Domain-Adaptive Variational Autoencoders for Medical Image Analysis},
    author={Your Name},
    journal={Your Journal},
    year={2024}
}
```

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use low-resolution mode
2. **Slow training**: Use low-resolution mode or reduce latent dimensions
3. **Poor reconstruction**: Adjust reconstruction weight or learning rate
4. **Domain overfitting**: Increase domain classification weight (alpha_2)

### Getting Help

- Check the output logs for detailed error messages
- Verify dataset path and permissions
- Ensure all dependencies are installed correctly
- Review hyperparameter settings for your hardware constraints

---

For more details about the NVAE project, please refer to the main repository documentation. 
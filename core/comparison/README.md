# CRMNIST Model Comparison

This directory contains implementations and evaluation frameworks for comparing different domain adaptation approaches on the CRMNIST dataset.

## Overview

The comparison framework enables systematic evaluation of various domain adaptation models:
- NVAE (New Variational Autoencoder) - our disentangled representation approach
- DIVA (Domain Invariant Variational Autoencoder) - a variant without shared latent space
- DANN (Domain-Adversarial Neural Network) - leveraging adversarial training for domain adaptation

The models are evaluated on their ability to generalize across domains with different rotation and color variations in the CRMNIST dataset.

## Files

- **dann.py**: Implementation of Domain-Adversarial Neural Network with gradient reversal layer
- **dann_trainer.py**: Custom trainer for DANN, extending the base Trainer class
- **train.py**: Training functions for all three model variants (NVAE, DIVA, DANN)
- **evaluation.py**: Utilities for cross-domain and holdout evaluation
- **metrics.py**: Metrics calculation utilities

## Models

### DANN (Domain-Adversarial Neural Network)

DANN uses a gradient reversal approach to encourage domain-invariant features:
- Feature extractor produces representations that should be domain-invariant
- Task classifier predicts digit class from these features
- Domain discriminator tries to identify the domain, while the feature extractor tries to fool it
- Gradient reversal layer enables adversarial optimization

### DIVA (Domain Invariant Variational Autoencoder)

DIVA is a modification of NVAE that:
- Removes the shared latent space between digit and domain attributes (zay)
- Redistributes these dimensions among the remaining latent spaces
- Forces stronger separation between content and style representations

## Evaluation Methods

### Cross-Domain Evaluation

Evaluates models on each domain separately to understand domain-specific performance:

```python
results = run_cross_domain_evaluation(args, nvae, diva, dann, test_loader, spec_data)
```

### Holdout Evaluation

Trains models on all domains except one, then tests on the held-out domain:

```python
results = run_holdout_evaluation(args, train_loader, test_loader, class_map, spec_data)
```

## Running Experiments

To run the full comparison experiment:

```bash
python -m core.CRMNIST.comparison.evaluation --config conf/crmnist.json --out results/ --cuda
```

### Command-Line Arguments

- `--config`: Path to configuration file (default: 'conf/crmnist.json')
- `--out`: Output directory for results (required)
- `--cuda`: Enable CUDA acceleration (default: True)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate for optimizers (default: 0.001)
- `--zy_dim`, `--zx_dim`, `--zay_dim`, `--za_dim`: Latent dimensions for VAE models (default: 8 each, 32 total)
- `--beta_1`, `--beta_2`, `--beta_3`, `--beta_4`: Beta weights for KL terms (default: 1.0 each)

## Output Structure

The evaluation creates:
- Model checkpoints for each approach
- Performance metrics for each domain
- Cross-domain evaluation results
- Visualization of latent spaces

## Visualizations

The models support latent space visualization to analyze feature disentanglement:

- DANN provides `visualize_latent_space()` to plot t-SNE embeddings of features
- NVAE/DIVA provide more extensive visualization tools for analyzing latent spaces

## Example Results

Typical output from the evaluation looks like:

```
--------------------------------
Cross-domain results:
--------------------------------
Domain results for NVAE:

Cross-domain test domain 0 (0°):
Loss: 0.3412
Accuracy: 0.9123

...

--------------------------------
Domain results for DIVA:
...
--------------------------------
Domain results for DANN:
...

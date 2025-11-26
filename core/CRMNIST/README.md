# CRMNIST Variational Autoencoder

## Files

- `run_crmnist.py`: Main training script
- `model.py`: VAE model architecture
- `data_generation.py`: Dataset generation and transformations
- `utils.py`: Utility functions

## Training the Model

To train the model:

```bash
# From the root directory of the repo
python -m core.CRMNIST.run_crmnist --out results/ --config conf/crmnist.json
```

### Required Arguments

- `--out`: Output directory for results (required)
- `--config`: Configuration file (default: `../conf/crmnist.json`)

### Optional Arguments

- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate for the optimizer (default: 1e-3)
- `--epochs`: Number of training epochs (default: 10)
- `--intensity`: Transform intensity (default: 1.5)
- `--intensity_decay`: Transform decay (default: 1.0)
- `--zy_dim`, `--zx_dim`, `--zay_dim`, `--za_dim`: Latent space dimensions (default: 8 each, 32 total)
- `--beta_1`, `--beta_2`, `--beta_3`, `--beta_4`: Beta weights for KL terms (default: 1.0 each)
- `--cuda`: Enable CUDA for GPU training (default: True)

## Output Structure

The training process creates:

- `reconstructions/`: Directory containing image reconstructions during training
- `models/`: Directory with saved model checkpoints
  - `model_best.pt`: The model with the best validation loss
  - `model_checkpoint_epoch_N.pt`: Final saved model
- `results.json`: Training results and metrics

## Early Stopping

The training includes early stopping after 5 epochs without improvement in validation loss. 
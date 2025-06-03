import torch
import os
from typing import Dict, Tuple
from tqdm import tqdm
from core.CRMNIST.trainer import CRMNISTTrainer
from core.WILD.trainer import WILDTrainer


def train(args, model, optimizer, train_loader, val_loader, device, patience, trainer_class=None):
    """
    Train a model with early stopping.
    
    Args:
        args: Arguments object
        model: Model to train
        optimizer: Optimizer to use
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        patience: Number of epochs to wait for improvement before early stopping
        trainer_class: Trainer class to use (default: Trainer)
        model_params: Dictionary of model parameters to save with the model
    """
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize trainer
    
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        device=device,
        args=args,
        patience=patience
    )
    
    # Train model (updates model state in-place with best parameters)
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Save final model state
    trainer.save_final_model(args.epochs)

    # Return training statistics
    return {
        'best_val_accuracy': trainer.best_val_accuracy,
        'total_epochs_trained': trainer.epochs_trained,
        'best_model_epoch': trainer.best_epoch + 1,  
        'best_model_state': trainer.best_model_state,
        'best_batch_metrics': trainer.best_batch_metrics,
        #'model_params': model_params  # Pass model params back to training function
    }

    
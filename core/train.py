import torch
import os
from typing import Dict, Tuple
from tqdm import tqdm
from core.trainer import Trainer

def train(args, model, optimizer, train_loader, val_loader, device, patience, trainer_class=Trainer):
    """
    Train the model in-place, updating its state with the best parameters found during training.
    Also returns training statistics about the training process.
    """
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize trainer
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        device=device,
        args=args,
        patience=patience,
    )
    
    # Train model (updates model state in-place with best parameters)
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Save final model state
    trainer.save_final_model(args.epochs)

    # Return training statistics
    return {
        'best_validation_loss': trainer.best_val_loss,
        'total_epochs_trained': trainer.epochs_trained,
        'best_model_epoch': trainer.best_epoch + 1,  
        'best_model_state': trainer.best_model_state,
        'best_batch_metrics': trainer.best_batch_metrics
    }

    
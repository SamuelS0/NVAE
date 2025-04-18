import torch
import os
from typing import Dict, Tuple
from tqdm import tqdm
from .trainer import Trainer
from .CRMNIST.model import NVAE
from .data import get_dataloaders

def train(args, model, optimizer, train_loader, val_loader, device):
    """
    Train the model in-place and return training metrics.
    The model's state will be updated during training and the best state will be loaded at the end.
    """
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        args=args,
        patience=args.patience,
        min_epochs=args.min_epochs
    )
    
    # Train model (this will update the model in-place)
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Save final model
    trainer.save_final_model(args.epochs)

    # Return training metrics
    return {
        'best_val_loss': trainer.best_val_loss,
        'epochs_trained': trainer.epochs_trained
    }

    
    # # Evaluate on test set
    # test_loss, test_metrics = trainer._validate(test_loader)
    # print("\nTest Results:")
    # print(f"Test Loss: {test_loss:.4f}")
    # for k, v in test_metrics.items():
    #     print(f"Test {k}: {v:.4f}")
    
    # return model, test_loss, test_metrics 
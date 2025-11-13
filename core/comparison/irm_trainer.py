from core.WILD.trainer import WILDTrainer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys
from core.utils import process_batch
import torch.nn.functional as F


class IRMTrainer(WILDTrainer):
    def __init__(self, model, optimizer, device, args, patience=5):
        super().__init__(model, optimizer, device, args, patience)
        self.dataset = args.dataset
        self.optimizer = optimizer

    def _train_epoch(self, train_loader, epoch, current_beta):
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_penalty = 0
       
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, batch in train_pbar:
            x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)
            
            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(d.shape) > 1 and d.shape[1] > 1:
                d = torch.argmax(d, dim=1)

            # Forward pass and loss computation
            irm_loss, class_loss, penalty = self.model.loss_function(x, y, d)
            penalty = torch.tensor(penalty)
            self.optimizer.zero_grad()
            irm_loss.backward()
            self.optimizer.step()

            # Get predictions for accuracy calculation
            logits, _ = self.model.forward(x, y, d)
            y_pred = torch.argmax(logits, dim=1)

            # Update loss totals
            total_loss += irm_loss.item()
            total_class_loss += class_loss.item()
            total_penalty += penalty.item()
            
            # Update accuracy counts
            batch_size = len(y)
            total_samples += batch_size
            correct_y += (y_pred == y).sum().item()

            train_pbar.set_postfix({
                'loss': irm_loss.item(),
                'class_loss': class_loss.item(),
                'penalty': penalty.item(),
                'y_acc': (correct_y / total_samples) * 100
            })

        # Calculate final metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_penalty = total_penalty / len(train_loader)
        y_accuracy = (correct_y / total_samples) * 100

        avg_train_metrics = {
            'y_accuracy': y_accuracy,
            'class_loss': avg_class_loss,
            'penalty': avg_penalty
        }

        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader, epoch, current_beta):
        self.model.eval()
        total_loss = 0
        total_class_loss = 0
        total_penalty = 0
        
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0

        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)

                # Convert one-hot encoded labels to class indices
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                if len(d.shape) > 1 and d.shape[1] > 1:
                    d = torch.argmax(d, dim=1)

                # Forward pass and loss computation
                irm_loss, class_loss, penalty = self.model.loss_function(x, y, d)

                # Get predictions for accuracy calculation
                logits, _ = self.model.forward(x, y, d)
                y_pred = torch.argmax(logits, dim=1)

                # Update loss totals
                total_loss += irm_loss.item()
                total_class_loss += class_loss.item()
                total_penalty += penalty.item()
                
                # Update accuracy counts
                batch_size = len(y)
                total_samples += batch_size
                correct_y += (y_pred == y).sum().item()

                val_pbar.set_postfix({
                    'loss': irm_loss.item(),
                    'class_loss': class_loss.item(),
                    'penalty': penalty.item(),
                    'y_acc': (correct_y / total_samples) * 100
                })

        # Calculate final metrics
        avg_val_loss = total_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        avg_penalty = total_penalty / len(val_loader)
        y_accuracy = (correct_y / total_samples) * 100

        avg_val_metrics = {
            'y_accuracy': y_accuracy,
            'class_loss': avg_class_loss,
            'penalty': avg_penalty
        }

        return avg_val_loss, avg_val_metrics 
from core.CRMNIST.trainer import CRMNISTTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from core.utils import process_batch

class DANNTrainer(CRMNISTTrainer):
    def __init__(self, model, optimizer, device, args, patience=5, model_params=None):
        super().__init__(model, optimizer, device, args, patience, model_params)

    
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_y_loss = 0
        total_domain_loss = 0
       
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0
        correct_domain = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, batch in train_pbar:
            #x, y, c, d = x.to(self.device), y.to(self.device), c.to(self.device), d.to(self.device)
            x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)
            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(d.shape) > 1 and d.shape[1] > 1:
                d = torch.argmax(d, dim=1)

            y_logits, domain_logits = self.model(x, y, d)

            y_pred = torch.argmax(y_logits, dim=1)
            domain_pred = torch.argmax(domain_logits, dim=1)

            loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, d)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update loss totals
            total_loss += loss.item()
            total_y_loss += y_loss.item()
            total_domain_loss += domain_loss.item()
            
            # Update accuracy counts
            batch_size = len(y)
            total_samples += batch_size
            correct_y += (y_pred == y).sum().item()
            correct_domain += (domain_pred == d).sum().item()

            train_pbar.set_postfix({
                'loss': loss.item(),
                'y_loss': y_loss.item(),
                'domain_loss': domain_loss.item(),
                'y_acc': (correct_y / total_samples) * 100,
                'domain_acc': (correct_domain / total_samples) * 100
            })

        # Calculate final metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_y_loss = total_y_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        y_accuracy = (correct_y / total_samples) * 100
        domain_accuracy = (correct_domain / total_samples) * 100

        avg_train_metrics = {
            'y_accuracy': y_accuracy,
            'discriminator_accuracy': domain_accuracy,
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_y_loss = 0
        total_domain_loss = 0
        
        # Initialize counters for accuracy
        total_samples = 0
        correct_y = 0
        correct_domain = 0

        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, d = process_batch(batch, self.device, dataset_type=self.dataset)

                # Convert one-hot encoded labels to class indices
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                if len(d.shape) > 1 and d.shape[1] > 1:
                    d = torch.argmax(d, dim=1)

                y_logits, domain_logits = self.model(x, y, d)

                y_pred = torch.argmax(y_logits, dim=1)
                domain_pred = torch.argmax(domain_logits, dim=1)

                loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, d)

                # Update loss totals
                total_loss += loss.item()
                total_y_loss += y_loss.item()
                total_domain_loss += domain_loss.item()
                
                # Update accuracy counts
                batch_size = len(y)
                total_samples += batch_size
                correct_y += (y_pred == y).sum().item()
                correct_domain += (domain_pred == d).sum().item()

                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'y_loss': y_loss.item(),
                    'domain_loss': domain_loss.item(),
                    'y_acc': (correct_y / total_samples) * 100,
                    'domain_acc': (correct_domain / total_samples) * 100
                })

        # Calculate final metrics
        avg_val_loss = total_loss / len(val_loader)
        avg_y_loss = total_y_loss / len(val_loader)
        avg_domain_loss = total_domain_loss / len(val_loader)
        y_accuracy = (correct_y / total_samples) * 100
        domain_accuracy = (correct_domain / total_samples) * 100

        avg_val_metrics = {
            'y_accuracy': y_accuracy,
            'discriminator_accuracy': domain_accuracy,
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_val_loss, avg_val_metrics
                
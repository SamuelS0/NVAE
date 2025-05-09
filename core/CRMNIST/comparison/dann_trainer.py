from core.trainer import Trainer
import torch
import torch.nn.functional as F
from tqdm import tqdm

class DANNTrainer(Trainer):
    def __init__(self, model, optimizer, device, args, patience=5):
        super().__init__(model, optimizer, device, args, patience)

    
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_y_loss = 0
        total_domain_loss = 0
       
        train_metrics_sum = {'y_accuracy': 0, 'discriminator_accuracy': 0}

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, (x, y, c, r) in train_pbar:
            x, y, c, r = x.to(self.device), y.to(self.device), c.to(self.device), r.to(self.device)

            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(r.shape) > 1 and r.shape[1] > 1:
                r = torch.argmax(r, dim=1)

            y_logits, domain_logits = self.model(x, y, r)

            y_pred = torch.argmax(y_logits, dim=1)
            domain_pred = torch.argmax(domain_logits, dim=1)

            loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, r)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_y_loss += y_loss.item()
            total_domain_loss += domain_loss.item()
            train_metrics_sum['y_accuracy'] += (y_pred == y).sum().item()
            train_metrics_sum['discriminator_accuracy'] += (domain_pred == r).sum().item()

            train_pbar.set_postfix({
                'loss': loss.item(),
                'y_loss': y_loss.item(),
                'domain_loss': domain_loss.item()
            })

        avg_train_loss = total_loss / len(train_loader)
        avg_y_loss = total_y_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        avg_train_metrics = {
            'y_accuracy': train_metrics_sum['y_accuracy'] / len(train_loader),
            'discriminator_accuracy': train_metrics_sum['discriminator_accuracy'] / len(train_loader),
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_train_loss, avg_train_metrics
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_y_loss = 0
        total_domain_loss = 0
        val_metrics_sum = {'y_accuracy': 0, 'discriminator_accuracy': 0}

        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

        with torch.no_grad():
            for batch_idx, (x, y, c, r) in val_pbar:
                x, y, c, r = x.to(self.device), y.to(self.device), c.to(self.device), r.to(self.device)

                # Convert one-hot encoded labels to class indices
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                if len(r.shape) > 1 and r.shape[1] > 1:
                    r = torch.argmax(r, dim=1)

                y_logits, domain_logits = self.model(x, y, r)

                y_pred = torch.argmax(y_logits, dim=1)
                domain_pred = torch.argmax(domain_logits, dim=1)

                loss, y_loss, domain_loss = self.model.loss_function(y_logits, domain_logits, y, r)

                total_loss += loss.item()
                total_y_loss += y_loss.item()
                total_domain_loss += domain_loss.item()
                val_metrics_sum['y_accuracy'] += (y_pred == y).sum().item()
                val_metrics_sum['discriminator_accuracy'] += (domain_pred == r).sum().item()

                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'y_loss': y_loss.item(),
                    'domain_loss': domain_loss.item()
                })

        avg_val_loss = total_loss / len(val_loader)
        avg_y_loss = total_y_loss / len(val_loader)
        avg_domain_loss = total_domain_loss / len(val_loader)
        avg_val_metrics = {
            'y_accuracy': val_metrics_sum['y_accuracy'] / len(val_loader),
            'discriminator_accuracy': val_metrics_sum['discriminator_accuracy'] / len(val_loader),
            'y_loss': avg_y_loss,
            'domain_loss': avg_domain_loss
        }

        return avg_val_loss, avg_val_metrics
                
                

                
    
            
        
        
        
        
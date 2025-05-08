import torch
from core.utils import calculate_metrics
from tqdm import tqdm

def test_nvae(model, test_loader, device):
    model.eval()
    test_loss = 0
    metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")
    
    with torch.no_grad():
        for batch_idx, (x, y, c, r) in test_pbar:
            x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
            
            loss = model.loss_function(y, x, r)
            test_loss += loss.item()

            batch_metrics = calculate_metrics(model, y, x, r)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            
            test_pbar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}

    return test_loss, metrics_avg

def test_dann(model, test_loader, device):
    model.eval()
    test_loss = 0
    metrics_sum = {'y_accuracy': 0, 'discriminator_accuracy': 0}

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")

    with torch.no_grad():
        for batch_idx, (x, y, c, r) in test_pbar:
            x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
            
            y_predictions, domain_predictions = model(x, y, r)

            loss = model.loss_function(y_predictions, domain_predictions, y, r)
            test_loss += loss.item()
            
            y_pred = torch.argmax(y_predictions, dim=1)
            domain_pred = torch.argmax(domain_predictions, dim=1)

            metrics_sum['y_accuracy'] += (y_pred == y).sum().item()
            metrics_sum['discriminator_accuracy'] += (domain_pred == r).sum().item()
            
            test_pbar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}

    return test_loss, metrics_avg
    
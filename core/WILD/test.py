import torch
from utils_wild import calculate_metrics, select_diverse_sample_batch
from tqdm import tqdm

def test(model, device, test_loader, args):
    model.eval()
    test_loss = 0
    metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}
    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                       desc=f"Final [Test]")
    with torch.no_grad():
        for batch_idx, (x, y, metadata) in test_pbar:
            
            hospital_id = metadata[:, 0]
            slide_id = metadata[:, 1]
            x, y, hospital_id = x.to(device), y.to(device), hospital_id.to(device)
            
            loss, _ = model.loss_function(hospital_id, x, y, current_beta = args.beta_scale)
            test_loss += loss.item()
            
            # Calculate additional metrics
            batch_metrics = calculate_metrics(model, y, x, hospital_id, args)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
    
    # Average metrics
    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}
    
    print(f'Test Loss: {test_loss:.4f}')
    for k, v in metrics_avg.items():
        print(f'Test {k}: {v:.4f}')
    
    return test_loss, metrics_avg

import torch
import torch.nn.functional as F
from core.utils import _calculate_metrics
from tqdm import tqdm
from core.utils import process_batch

# Main test function that forwards to test_nvae
def test(model, test_loader, dataset_type, device):
    return test_nvae(model, test_loader, dataset_type, device)

def test_nvae(model, test_loader, dataset_type, device):
    model.eval()
    test_loss = 0
    metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0}

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")
    
    with torch.no_grad():
        for batch_idx, batch in test_pbar:
            #x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
            x, y, domain = process_batch(batch, device, dataset_type=dataset_type)
            loss_result = model.loss_function(y, x, domain)

            # Handle both old (scalar) and new (tuple) return formats
            if isinstance(loss_result, tuple):
                loss, loss_components = loss_result
            else:
                loss = loss_result
                loss_components = None

            test_loss += loss.item()

            batch_metrics = _calculate_metrics(model, y, x, domain, 'test')
            for k, v in batch_metrics.items():
                metrics_sum[k] += v

            # Update progress bar with detailed loss components if available
            if loss_components:
                test_pbar.set_postfix(
                    loss=f"{loss_components['total']:.2f}",
                    recon=f"{loss_components['recon']:.1f}",
                    l1_zy=f"{loss_components['l1_zy']:.3f}",
                    l1_za=f"{loss_components['l1_za']:.3f}",
                    l1_zay=f"{loss_components['l1_zay']:.3f}"
                )
            else:
                test_pbar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}

    return test_loss, metrics_avg

def test_dann(model, test_loader, dataset_type, device):
    model.eval()
    test_loss = 0
    test_y_loss = 0
    test_domain_loss = 0
    metrics_sum = {'y_accuracy': 0, 'discriminator_accuracy': 0}

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")

    with torch.no_grad():
        for batch_idx, batch in test_pbar:
            #x, y, c, r = x.to(device), y.to(device), c.to(device), r.to(device)
            x, y, r = process_batch(batch, device, dataset_type=dataset_type)
            
            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(r.shape) > 1 and r.shape[1] > 1:
                r = torch.argmax(r, dim=1)

            # Handle both basic DANN (2 outputs) and AugmentedDANN (13 outputs)
            output = model(x, y, r)
            if isinstance(output, tuple) and len(output) == 13:
                # AugmentedDANN returns NVAE-style interface
                x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = output
                y_predictions = y_hat
                domain_predictions = a_hat
            else:
                # Basic DANN returns (y_predictions, domain_predictions)
                y_predictions, domain_predictions = output

            loss, y_loss, domain_loss = model.loss_function(y_predictions, domain_predictions, y, r)
            test_loss += loss.item()
            test_y_loss += y_loss.item()
            test_domain_loss += domain_loss.item()
            
            y_pred = torch.argmax(y_predictions, dim=1)
            domain_pred = torch.argmax(domain_predictions, dim=1)

            metrics_sum['y_accuracy'] += (y_pred == y).sum().item()
            metrics_sum['discriminator_accuracy'] += (domain_pred == r).sum().item()
            
            test_pbar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    test_y_loss /= len(test_loader)
    test_domain_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}
    metrics_avg['y_loss'] = test_y_loss
    metrics_avg['domain_loss'] = test_domain_loss

    return test_loss, metrics_avg


def test_irm(model, test_loader, dataset_type, device):
    model.eval()
    test_loss = 0
    metrics_sum = {'y_accuracy': 0}

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Final evaluation")

    with torch.no_grad():
        for batch_idx, batch in test_pbar:
            x, y, r = process_batch(batch, device, dataset_type=dataset_type)

            # Convert one-hot encoded labels to class indices
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if len(r.shape) > 1 and r.shape[1] > 1:
                r = torch.argmax(r, dim=1)

            # IRM forward returns (logits, features)
            logits, features = model(x, y, r)

            # Simple cross-entropy loss for testing (no IRM penalty)
            loss = F.cross_entropy(logits, y)
            test_loss += loss.item()

            y_pred = torch.argmax(logits, dim=1)
            metrics_sum['y_accuracy'] += (y_pred == y).sum().item()

            test_pbar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    metrics_avg = {k: v / len(test_loader) for k, v in metrics_sum.items()}

    return test_loss, metrics_avg


def test_with_ood(model, id_test_loader, ood_test_loader, dataset_type, device, model_type='nvae'):
    """
    Evaluate model on both ID and OOD test sets and compute generalization gap.

    Args:
        model: The model to evaluate
        id_test_loader: DataLoader for in-distribution test data
        ood_test_loader: DataLoader for out-of-distribution test data
        dataset_type: 'crmnist' or 'wild'
        device: torch device
        model_type: Type of model ('nvae', 'diva', 'dann', 'dann_augmented', 'irm')

    Returns:
        dict with keys: 'id_loss', 'id_metrics', 'ood_loss', 'ood_metrics', 'generalization_gap'
    """
    results = {}

    # Choose appropriate test function based on model type
    if model_type in ['dann', 'dann_augmented']:
        test_fn = test_dann
    elif model_type == 'irm':
        # IRM has different interface, needs its own test function
        test_fn = test_irm
    else:
        # NVAE and DIVA use the same test function
        test_fn = test_nvae

    # ID test evaluation
    print("\n📊 Evaluating on In-Distribution (ID) test set...")
    id_loss, id_metrics = test_fn(model, id_test_loader, dataset_type, device)
    results['id_loss'] = id_loss
    results['id_metrics'] = id_metrics

    print(f"\n✅ ID Test Results:")
    print(f"   Loss: {id_loss:.4f}")
    for k, v in id_metrics.items():
        print(f"   {k}: {v:.4f}")

    # OOD test evaluation
    if ood_test_loader is not None:
        print("\n🎯 Evaluating on Out-of-Distribution (OOD) test set...")
        ood_loss, ood_metrics = test_fn(model, ood_test_loader, dataset_type, device)
        results['ood_loss'] = ood_loss
        results['ood_metrics'] = ood_metrics

        print(f"\n🔍 OOD Test Results:")
        print(f"   Loss: {ood_loss:.4f}")
        for k, v in ood_metrics.items():
            print(f"   {k}: {v:.4f}")

        # Compute generalization gap
        results['generalization_gap'] = {
            'loss': ood_loss - id_loss,
        }

        # Compute accuracy gap if available
        if 'y_accuracy' in id_metrics and 'y_accuracy' in ood_metrics:
            results['generalization_gap']['y_accuracy'] = id_metrics['y_accuracy'] - ood_metrics['y_accuracy']

        if 'a_accuracy' in id_metrics and 'a_accuracy' in ood_metrics:
            results['generalization_gap']['a_accuracy'] = id_metrics['a_accuracy'] - ood_metrics['a_accuracy']

        print(f"\n📈 Generalization Gap (ID - OOD):")
        print(f"   Loss increase: {results['generalization_gap']['loss']:.4f}")
        if 'y_accuracy' in results['generalization_gap']:
            print(f"   Y accuracy drop: {results['generalization_gap']['y_accuracy']:.4f}")
        if 'a_accuracy' in results['generalization_gap']:
            print(f"   A accuracy drop: {results['generalization_gap']['a_accuracy']:.4f}")
    else:
        results['ood_loss'] = None
        results['ood_metrics'] = None
        results['generalization_gap'] = None

    return results

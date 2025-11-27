import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


class MLPClassifier(nn.Module):
    """3-layer MLP with 32 hidden units for expressiveness evaluation."""

    def __init__(self, input_dim, num_classes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def extract_latent_representations(model, dataloader, device, num_classes=10):
    """
    Extract latent representations from the trained VAE model.

    Args:
        model: Trained VAE model
        dataloader: DataLoader providing batches
        device: Device to run on (cpu/cuda)
        num_classes: Total number of classes (default 10 for MNIST digits)

    Returns:
        dict: Contains zy, zx, zdy, zd representations and corresponding y, a labels
    """
    model.eval()

    # Validate dataloader is not empty
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty - cannot extract representations")

    # Detect AugmentedDANN model (uses 3-component latent structure)
    is_augmented_dann = (hasattr(model, 'extract_features') and
                         hasattr(model, 'name') and
                         model.name == 'dann_augmented')

    if is_augmented_dann:
        print("Detected AugmentedDANN model - using extract_features() for 3-space latent extraction")
        print("⚠️  Note: AugmentedDANN maps zx=zd for interface compatibility.")
        print("   Expressiveness metrics for zx and zd will be identical.")

    all_zy = []
    all_zx = []
    all_zdy = []
    all_zd = []
    all_y = []  # Will store INTEGER indices for digit labels
    all_a = []  # Will store INTEGER indices for domain labels

    with torch.no_grad():
        for batch_idx, (x, y_raw, color, a_raw) in enumerate(dataloader):
            x = x.to(device)

            # Convert y (digit label) to indices if one-hot, keep as-is if already indices
            if len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
                # One-hot encoded -> convert to indices
                y_indices = y_raw.argmax(dim=1)
            else:
                # Already indices
                y_indices = y_raw.long().squeeze()

            # Convert a (domain label) to indices if one-hot, keep as-is if already indices
            if len(a_raw.shape) > 1 and a_raw.shape[1] > 1:
                # One-hot encoded -> convert to indices
                a_indices = a_raw.argmax(dim=1)
            else:
                # Already indices
                a_indices = a_raw.long().squeeze()

            # For the model forward pass, we need one-hot y and the original a (one-hot)
            y_onehot = F.one_hot(y_indices.long(), num_classes=num_classes).float().to(device)
            a_raw = a_raw.to(device)

            if is_augmented_dann:
                # AugmentedDANN has 3 true latent spaces: zy, zd, zdy
                # Use extract_features() to get them without duplicates
                try:
                    features = model.extract_features(x)
                    assert len(features) == 3, f"Expected 3 latent spaces from extract_features(), got {len(features)}"
                    zy, zd, zdy = features
                except Exception as e:
                    raise RuntimeError(f"Failed to extract AugmentedDANN features: {e}")

                # Map to expected format:
                # - zy stays as zy (class-specific)
                # - zd is domain-specific
                # - zdy is domain-label interaction
                # - zx is empty (no residual space in discriminative models - no decoder)
                zx = torch.zeros(zy.shape[0], 0, device=zy.device)
                # zd and zdy already in correct format
            else:
                # VAE models (NVAE/DIVA) - use standard forward pass
                x_recon, z, qz, pzy, pzx, pzd, pzdy, y_hat, a_hat, zy, zx, zdy, zd = model(y_onehot, x, a_raw)

            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            if zdy is not None and zdy.shape[1] > 0:
                all_zdy.append(zdy.cpu())
            else:
                # For DIVA models or empty zdy, create zeros
                all_zdy.append(torch.zeros(zy.shape[0], 0))
            all_zd.append(zd.cpu())
            # Store INTEGER indices for labels (not one-hot)
            all_y.append(y_indices.cpu())
            all_a.append(a_indices.cpu())
    
    # Concatenate all batches
    latent_data = {
        'zy': torch.cat(all_zy, dim=0),
        'zx': torch.cat(all_zx, dim=0),
        'zdy': torch.cat(all_zdy, dim=0) if all_zdy[0].shape[1] > 0 else None,
        'zd': torch.cat(all_zd, dim=0),
        'y': torch.cat(all_y, dim=0),
        'a': torch.cat(all_a, dim=0)
    }
    
    return latent_data

def train_pytorch_classifier(X_train, y_train, X_val, y_val, X_test=None, y_test=None, num_classes=None,
                             batch_size=256, num_epochs=50, hidden_dim=32):
    """Train a 2-layer MLP classifier using PyTorch with mini-batch training.

    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (can be one-hot or indices)
        X_val: Validation features (numpy array)
        y_val: Validation labels (can be one-hot or indices)
        X_test: Test features (optional, numpy array)
        y_test: Test labels (optional)
        num_classes: Number of classes (optional). If None, inferred from one-hot
                     dimension or max label value. Recommended to pass explicitly.
        batch_size: Mini-batch size for training (default: 256)
        num_epochs: Maximum number of training epochs (default: 50)
        hidden_dim: Hidden layer dimension (default: 32)

    Returns:
        tuple: (train_acc, val_acc, test_acc, trained_classifier)
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine num_classes from one-hot dimension if not provided
    if num_classes is None:
        if len(y_train.shape) > 1:
            # One-hot encoded: use the dimension
            num_classes = y_train.shape[1]
        else:
            # Indices: use max + 1 across all splits to be safe
            all_labels = [y_train]
            if y_val is not None:
                all_labels.append(y_val)
            if y_test is not None:
                all_labels.append(y_test)
            num_classes = int(max(np.max(l) for l in all_labels)) + 1

    # Convert one-hot encoded labels to class indices if needed
    if len(y_train.shape) > 1:
        y_train_indices = np.argmax(y_train, axis=1)
        y_val_indices = np.argmax(y_val, axis=1)
        if X_test is not None and y_test is not None:
            y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_train_indices = y_train.astype(np.int64) if isinstance(y_train, np.ndarray) else y_train
        y_val_indices = y_val.astype(np.int64) if isinstance(y_val, np.ndarray) else y_val
        if X_test is not None and y_test is not None:
            y_test_indices = y_test.astype(np.int64) if isinstance(y_test, np.ndarray) else y_test

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_indices, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_indices, dtype=torch.long, device=device)

    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test_indices, dtype=torch.long, device=device)

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create MLP classifier with correct number of output classes
    input_dim = X_train.shape[1]
    clf = MLPClassifier(input_dim, num_classes, hidden_dim=hidden_dim).to(device)

    # Training setup with LR scheduler
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 10% decay every 5 epochs
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training with mini-batches
        clf.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = clf(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Step LR scheduler at end of epoch
        scheduler.step()

        # Validation (full batch is fine since it's smaller)
        clf.eval()
        with torch.no_grad():
            val_logits = clf(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        clf.load_state_dict(best_state)

    # Evaluate
    clf.eval()
    with torch.no_grad():
        # Train accuracy (evaluate in batches to avoid memory issues)
        all_train_preds = []
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            preds = clf(X_batch).argmax(dim=1).cpu()
            all_train_preds.append(preds)
        train_preds = torch.cat(all_train_preds).numpy()
        train_acc = accuracy_score(y_train_indices, train_preds)

        # Validation accuracy
        val_preds = clf(X_val_t).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val_indices, val_preds)

        # Test accuracy (if provided)
        test_acc = None
        if X_test is not None and y_test is not None:
            test_preds = clf(X_test_t).argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test_indices, test_preds)

    return train_acc, val_acc, test_acc, clf

def evaluate_latent_expressiveness(model, train_loader, val_loader, test_loader, device, save_dir):
    """
    Evaluate the expressiveness of different latent variable combinations.
    
    Compare:
    1. zd alone vs zd+zdy for domain classification
    2. zy alone vs zy+zdy for label classification
    """
    
    print("🔍 Extracting latent representations...")

    # Extract latent representations
    # Use model's y_dim if available, otherwise default to 10 (MNIST digits)
    num_y_classes = model.y_dim if hasattr(model, 'y_dim') else 10
    train_data = extract_latent_representations(model, train_loader, device, num_y_classes)
    val_data = extract_latent_representations(model, val_loader, device, num_y_classes)
    test_data = extract_latent_representations(model, test_loader, device, num_y_classes)
    # Convert to numpy for sklearn compatibility
    train_zy = train_data['zy'].numpy()
    train_zd = train_data['zd'].numpy()
    train_zdy = train_data['zdy'].numpy() if train_data['zdy'] is not None else None
    train_y = train_data['y'].numpy()
    train_a = train_data['a'].numpy()
    #print(f"train data shape: {train_zy.shape}, {train_zd.shape}, {train_zdy.shape}, {train_y.shape}, {train_a.shape}")
    
    val_zy = val_data['zy'].numpy()
    val_zd = val_data['zd'].numpy()
    val_zdy = val_data['zdy'].numpy() if val_data['zdy'] is not None else None
    val_y = val_data['y'].numpy()
    val_a = val_data['a'].numpy()
    
    test_zy = test_data['zy'].numpy()
    test_zd = test_data['zd'].numpy()
    test_zdy = test_data['zdy'].numpy() if test_data['zdy'] is not None else None
    test_y = test_data['y'].numpy()
    test_a = test_data['a'].numpy()

    results = {}
    
    # Skip zdy experiments if model doesn't have zdy (DIVA case)
    has_zdy = train_zdy is not None and train_zdy.shape[1] > 0
    
    print(f"📊 Training classifiers...")
    print(f"   - Training samples: {train_zy.shape[0]}")
    print(f"   - Validation samples: {val_zy.shape[0]}")
    print(f"   - Test samples: {test_zy.shape[0]}")
    print(f"   - zy dim: {train_zy.shape[1]}")
    print(f"   - zd dim: {train_zd.shape[1]}")
    if has_zdy:
        print(f"   - zdy dim: {train_zdy.shape[1]}")
    else:
        print(f"   - zdy: Not available (DIVA model)")

    # Verify label distributions
    print(f"   - Digit labels (y): unique values = {np.unique(train_y)}")
    print(f"   - Domain labels (a): unique values = {np.unique(train_a)}")
    print(f"   - Test digit labels (y): unique values = {np.unique(test_y)}")
    print(f"   - Test domain labels (a): unique values = {np.unique(test_a)}")
    
    # =============================================================================
    # DOMAIN CLASSIFICATION (predict 'a' from latent variables)
    # =============================================================================
    print("\n🎯 Domain Classification Experiments:")
    
    # 1. Domain classification using zd alone
    print("   Training zd → domain classifier...")

    zd_train_acc, zd_val_acc, zd_test_acc, _ = train_pytorch_classifier(
        X_train=train_zd, y_train=train_a, X_val=val_zd, y_val=val_a, X_test=test_zd, y_test=test_a
    )
    results['domain_zd_alone'] = {'train_acc': zd_train_acc, 'val_acc': zd_val_acc, 'test_acc': zd_test_acc}
    
    # 2. Domain classification using zy alone (cross-prediction test)
    print("   Training zy → domain classifier...")
    zy_d_train_acc, zy_d_val_acc, zy_d_test_acc, _ = train_pytorch_classifier(
        X_train=train_zy, y_train=train_a, X_val=val_zy, y_val=val_a, X_test=test_zy, y_test=test_a
    )
    results['domain_zy_alone'] = {'train_acc': zy_d_train_acc, 'val_acc': zy_d_val_acc, 'test_acc': zy_d_test_acc}

    if has_zdy:
        # 3. Domain classification using zd+zdy
        print("   Training zd+zdy → domain classifier...")
        train_zd_zdy = np.concatenate([train_zd, train_zdy], axis=1)
        val_zd_zdy = np.concatenate([val_zd, val_zdy], axis=1)
        test_zd_zdy = np.concatenate([test_zd, test_zdy], axis=1)

        zd_zdy_train_acc, zd_zdy_val_acc, zd_zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zd_zdy, y_train=train_a, X_val=val_zd_zdy, y_val=val_a, X_test=test_zd_zdy, y_test=test_a
        )
        results['domain_zd_zdy'] = {'train_acc': zd_zdy_train_acc, 'val_acc': zd_zdy_val_acc, 'test_acc': zd_zdy_test_acc}

        # 4. Domain classification using zdy alone (for comparison)
        print("   Training zdy → domain classifier...")
        zdy_train_acc, zdy_val_acc, zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zdy, y_train=train_a, X_val=val_zdy, y_val=val_a, X_test=test_zdy, y_test=test_a
        )
        results['domain_zdy_alone'] = {'train_acc': zdy_train_acc, 'val_acc': zdy_val_acc, 'test_acc': zdy_test_acc}
    
    # =============================================================================
    # LABEL CLASSIFICATION (predict 'y' from latent variables)
    # =============================================================================
    print("\n🏷️  Label Classification Experiments:")
    
    # 1. Label classification using zy alone
    print("   Training zy → label classifier...")
    zy_train_acc, zy_val_acc, zy_test_acc, _ = train_pytorch_classifier(
        X_train=train_zy, y_train=train_y, X_val=val_zy, y_val=val_y, X_test=test_zy, y_test=test_y
    )
    results['label_zy_alone'] = {'train_acc': zy_train_acc, 'val_acc': zy_val_acc, 'test_acc': zy_test_acc}

    # 2. Label classification using zd alone (cross-prediction test)
    print("   Training zd → label classifier...")
    zd_y_train_acc, zd_y_val_acc, zd_y_test_acc, _ = train_pytorch_classifier(
        X_train=train_zd, y_train=train_y, X_val=val_zd, y_val=val_y, X_test=test_zd, y_test=test_y
    )
    results['label_zd_alone'] = {'train_acc': zd_y_train_acc, 'val_acc': zd_y_val_acc, 'test_acc': zd_y_test_acc}

    if has_zdy:
        # 3. Label classification using zy+zdy
        print("   Training zy+zdy → label classifier...")
        train_zy_zdy = np.concatenate([train_zy, train_zdy], axis=1)
        val_zy_zdy = np.concatenate([val_zy, val_zdy], axis=1)
        test_zy_zdy = np.concatenate([test_zy, test_zdy], axis=1)

        zy_zdy_train_acc, zy_zdy_val_acc, zy_zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zy_zdy, y_train=train_y, X_val=val_zy_zdy, y_val=val_y, X_test=test_zy_zdy, y_test=test_y
        )
        results['label_zy_zdy'] = {'train_acc': zy_zdy_train_acc, 'val_acc': zy_zdy_val_acc, 'test_acc': zy_zdy_test_acc}

        # 4. Label classification using zdy alone (for comparison)
        print("   Training zdy → label classifier...")
        zdy_y_train_acc, zdy_y_val_acc, zdy_y_test_acc, _ = train_pytorch_classifier(
            X_train=train_zdy, y_train=train_y, X_val=val_zdy, y_val=val_y, X_test=test_zdy, y_test=test_y
        )
        results['label_zdy_alone'] = {'train_acc': zdy_y_train_acc, 'val_acc': zdy_y_val_acc, 'test_acc': zdy_y_test_acc}
    
    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    print("\n📈 EXPRESSIVENESS COMPARISON RESULTS:")
    print("="*60)
    
    print("\n🎯 DOMAIN CLASSIFICATION:")
    print(f"   zd alone:      Train={results['domain_zd_alone']['train_acc']:.4f}, Val={results['domain_zd_alone']['val_acc']:.4f}, Test={results['domain_zd_alone']['test_acc']:.4f}")
    print(f"   zy alone:      Train={results['domain_zy_alone']['train_acc']:.4f}, Val={results['domain_zy_alone']['val_acc']:.4f}, Test={results['domain_zy_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zdy:
        print(f"   zd+zdy:        Train={results['domain_zd_zdy']['train_acc']:.4f}, Val={results['domain_zd_zdy']['val_acc']:.4f}, Test={results['domain_zd_zdy']['test_acc']:.4f}")
        print(f"   zdy alone:     Train={results['domain_zdy_alone']['train_acc']:.4f}, Val={results['domain_zdy_alone']['val_acc']:.4f}, Test={results['domain_zdy_alone']['test_acc']:.4f}")

        domain_improvement_val = results['domain_zd_zdy']['val_acc'] - results['domain_zd_alone']['val_acc']
        domain_improvement_test = results['domain_zd_zdy']['test_acc'] - results['domain_zd_alone']['test_acc']
        baseline_val = results['domain_zd_alone']['val_acc']
        baseline_test = results['domain_zd_alone']['test_acc']
        domain_improvement_val_pct = (domain_improvement_val / baseline_val) * 100 if baseline_val > 0 else 0.0
        domain_improvement_test_pct = (domain_improvement_test / baseline_test) * 100 if baseline_test > 0 else 0.0
        print(f"   📊 IMPROVEMENT (Val): zd+zdy is {domain_improvement_val:.4f} ({domain_improvement_val_pct:.2f}%) better than zd alone")
        print(f"   📊 IMPROVEMENT (Test): zd+zdy is {domain_improvement_test:.4f} ({domain_improvement_test_pct:.2f}%) better than zd alone")
    
    print("\n🏷️  LABEL CLASSIFICATION:")
    print(f"   zy alone:      Train={results['label_zy_alone']['train_acc']:.4f}, Val={results['label_zy_alone']['val_acc']:.4f}, Test={results['label_zy_alone']['test_acc']:.4f}")
    print(f"   zd alone:      Train={results['label_zd_alone']['train_acc']:.4f}, Val={results['label_zd_alone']['val_acc']:.4f}, Test={results['label_zd_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zdy:
        print(f"   zy+zdy:        Train={results['label_zy_zdy']['train_acc']:.4f}, Val={results['label_zy_zdy']['val_acc']:.4f}, Test={results['label_zy_zdy']['test_acc']:.4f}")
        print(f"   zdy alone:     Train={results['label_zdy_alone']['train_acc']:.4f}, Val={results['label_zdy_alone']['val_acc']:.4f}, Test={results['label_zdy_alone']['test_acc']:.4f}")

        label_improvement_val = results['label_zy_zdy']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zdy']['test_acc'] - results['label_zy_alone']['test_acc']
        baseline_val = results['label_zy_alone']['val_acc']
        baseline_test = results['label_zy_alone']['test_acc']
        label_improvement_val_pct = (label_improvement_val / baseline_val) * 100 if baseline_val > 0 else 0.0
        label_improvement_test_pct = (label_improvement_test / baseline_test) * 100 if baseline_test > 0 else 0.0
        print(f"   📊 IMPROVEMENT (Val): zy+zdy is {label_improvement_val:.4f} ({label_improvement_val_pct:.2f}%) better than zy alone")
        print(f"   📊 IMPROVEMENT (Test): zy+zdy is {label_improvement_test:.4f} ({label_improvement_test_pct:.2f}%) better than zy alone")
    
    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    # Always create visualization (handles both NVAE and DIVA models)
    create_expressiveness_visualization(results, save_dir, has_zdy)

    # Save results to JSON
    import json
    results_path = os.path.join(save_dir, 'latent_expressiveness_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")

    return results

def create_expressiveness_visualization(results, save_dir, has_zdy):
    """Create visualization comparing the expressiveness of different latent combinations.

    Args:
        results: Dictionary containing classification results
        save_dir: Directory to save visualization
        has_zdy: Boolean indicating if model has zdy component (NVAE=True, DIVA=False)
    """

    # Prepare data for visualization - include both validation and test
    comparison_data = []

    # Domain classification comparisons
    for split, split_name in [('val_acc', 'Validation'), ('test_acc', 'Test')]:
        comparison_data.append({
            'Task': 'Domain Classification',
            'Method': 'zd alone',
            'Accuracy': results['domain_zd_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Domain Classification',
            'Method': 'zy alone',
            'Accuracy': results['domain_zy_alone'][split],
            'Type': 'Cross-prediction',
            'Split': split_name
        })

        # Only add zdy-related metrics if model has zdy component
        if has_zdy:
            comparison_data.append({
                'Task': 'Domain Classification',
                'Method': 'zd+zdy',
                'Accuracy': results['domain_zd_zdy'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Domain Classification',
                'Method': 'zdy alone',
                'Accuracy': results['domain_zdy_alone'][split],
                'Type': 'Individual',
                'Split': split_name
            })

        # Label classification comparisons
        comparison_data.append({
            'Task': 'Label Classification',
            'Method': 'zy alone',
            'Accuracy': results['label_zy_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Label Classification',
            'Method': 'zd alone',
            'Accuracy': results['label_zd_alone'][split],
            'Type': 'Cross-prediction',
            'Split': split_name
        })

        # Only add zdy-related metrics if model has zdy component
        if has_zdy:
            comparison_data.append({
                'Task': 'Label Classification',
                'Method': 'zy+zdy',
                'Accuracy': results['label_zy_zdy'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Label Classification',
                'Method': 'zdy alone',
                'Accuracy': results['label_zdy_alone'][split],
                'Type': 'Individual',
                'Split': split_name
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Create visualization with validation and test results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate adaptive y-axis limit based on data
    max_acc = df['Accuracy'].max()
    y_max = min(1.0, max_acc + 0.05)  # Add 5% padding, but cap at 1.0

    # Choose colors based on whether model has zdy component
    # NVAE (has_zdy=True): 4 bars - orange, red, green, blue
    # DIVA (has_zdy=False): 2 bars - orange, red
    bar_colors = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4'] if has_zdy else ['#ff7f0e', '#d62728']

    # Domain classification - Validation
    domain_val_data = df[(df['Task'] == 'Domain Classification') & (df['Split'] == 'Validation')]
    bars1 = axes[0,0].bar(domain_val_data['Method'], domain_val_data['Accuracy'], color=bar_colors)
    axes[0,0].set_title('Domain Classification - Validation', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0,0].set_ylim(0, y_max)
    axes[0,0].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (5 training domains: 1/5 = 0.20)
    axes[0,0].axhline(y=0.20, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (20%)')
    axes[0,0].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Domain classification - Test
    domain_test_data = df[(df['Task'] == 'Domain Classification') & (df['Split'] == 'Test')]
    bars2 = axes[0,1].bar(domain_test_data['Method'], domain_test_data['Accuracy'], color=bar_colors)
    axes[0,1].set_title('Domain Classification - Test', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0,1].set_ylim(0, y_max)
    axes[0,1].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (5 training domains: 1/5 = 0.20)
    axes[0,1].axhline(y=0.20, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (20%)')
    axes[0,1].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Label classification - Validation
    label_val_data = df[(df['Task'] == 'Label Classification') & (df['Split'] == 'Validation')]
    bars3 = axes[1,0].bar(label_val_data['Method'], label_val_data['Accuracy'], color=bar_colors)
    axes[1,0].set_title('Label Classification - Validation', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1,0].set_ylim(0, y_max)
    axes[1,0].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (10 digit classes: 1/10 = 0.10)
    axes[1,0].axhline(y=0.10, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (10%)')
    axes[1,0].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Label classification - Test
    label_test_data = df[(df['Task'] == 'Label Classification') & (df['Split'] == 'Test')]
    bars4 = axes[1,1].bar(label_test_data['Method'], label_test_data['Accuracy'], color=bar_colors)
    axes[1,1].set_title('Label Classification - Test', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1,1].set_ylim(0, y_max)
    axes[1,1].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (10 digit classes: 1/10 = 0.10)
    axes[1,1].axhline(y=0.10, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (10%)')
    axes[1,1].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Only calculate and display improvements for models with zdy component
    # (DIVA models don't have zdy, so no improvement to show)
    if has_zdy:
        # Calculate and display improvements
        domain_improvement_val = results['domain_zd_zdy']['val_acc'] - results['domain_zd_alone']['val_acc']
        domain_improvement_test = results['domain_zd_zdy']['test_acc'] - results['domain_zd_alone']['test_acc']
        label_improvement_val = results['label_zy_zdy']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zdy']['test_acc'] - results['label_zy_alone']['test_acc']

        # Calculate percentage improvements (relative to baseline) with zero protection
        domain_base_val = results['domain_zd_alone']['val_acc']
        domain_base_test = results['domain_zd_alone']['test_acc']
        label_base_val = results['label_zy_alone']['val_acc']
        label_base_test = results['label_zy_alone']['test_acc']
        domain_improvement_val_pct = (domain_improvement_val / domain_base_val) * 100 if domain_base_val > 0 else 0.0
        domain_improvement_test_pct = (domain_improvement_test / domain_base_test) * 100 if domain_base_test > 0 else 0.0
        label_improvement_val_pct = (label_improvement_val / label_base_val) * 100 if label_base_val > 0 else 0.0
        label_improvement_test_pct = (label_improvement_test / label_base_test) * 100 if label_base_test > 0 else 0.0

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot
    plot_path = os.path.join(save_dir, 'latent_expressiveness_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Expressiveness comparison plot saved to: {plot_path}") 
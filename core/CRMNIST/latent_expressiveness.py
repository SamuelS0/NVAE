import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


class MLPClassifier(nn.Module):
    """Simple 2-layer MLP for expressiveness evaluation."""

    def __init__(self, input_dim, num_classes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def extract_latent_representations(model, dataloader, device, num_classes=10):
    """
    Extract latent representations from the trained VAE model.

    Args:
        model: Trained VAE model
        dataloader: DataLoader providing batches
        device: Device to run on (cpu/cuda)
        num_classes: Total number of classes (default 10 for MNIST digits)

    Returns:
        dict: Contains zy, zx, zay, za representations and corresponding y, a labels
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
        print("⚠️  Note: AugmentedDANN maps zx=za=zd for interface compatibility.")
        print("   Expressiveness metrics for zx and za will be identical.")

    all_zy = []
    all_zx = []
    all_zay = []
    all_za = []
    all_y = []
    all_a = []

    with torch.no_grad():
        for batch_idx, (x, y, color,  a) in enumerate(dataloader):
            x, a = x.to(device), a.to(device)
            # Use fixed num_classes for all batches to ensure consistent one-hot encoding size
            y_onehot = F.one_hot(y.long(), num_classes=num_classes).float()
            y_onehot = y_onehot.to(device)
            y = y_onehot

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
                # - zd maps to za (domain-specific, maps to auxiliary)
                # - zdy maps to zay (interaction)
                # - zx is empty (no residual space in discriminative models - no decoder)
                zx = torch.zeros(zy.shape[0], 0, device=zy.device)
                za = zd  # Domain features
                zay = zdy  # Interaction features
            else:
                # VAE models (NVAE/DIVA) - use standard forward pass
                x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model(y, x, a)

            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            if zay is not None and zay.shape[1] > 0:
                all_zay.append(zay.cpu())
            else:
                # For DIVA models or empty zay, create zeros
                all_zay.append(torch.zeros(zy.shape[0], 0))
            all_za.append(za.cpu())
            all_y.append(y.cpu())
            all_a.append(a.cpu())
    
    # Concatenate all batches
    latent_data = {
        'zy': torch.cat(all_zy, dim=0),
        'zx': torch.cat(all_zx, dim=0),
        'zay': torch.cat(all_zay, dim=0) if all_zay[0].shape[1] > 0 else None,
        'za': torch.cat(all_za, dim=0),
        'y': torch.cat(all_y, dim=0),
        'a': torch.cat(all_a, dim=0)
    }
    
    return latent_data

def train_pytorch_classifier(X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a 2-layer MLP classifier using PyTorch for classification tasks.

    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (can be one-hot or indices)
        X_val: Validation features (numpy array)
        y_val: Validation labels (can be one-hot or indices)
        X_test: Test features (optional, numpy array)
        y_test: Test labels (optional)

    Returns:
        tuple: (train_acc, val_acc, test_acc, trained_classifier)
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert one-hot encoded labels to class indices if needed
    if len(y_train.shape) > 1:
        y_train_indices = np.argmax(y_train, axis=1)
        y_val_indices = np.argmax(y_val, axis=1)
        if X_test is not None and y_test is not None:
            y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_train_indices = y_train
        y_val_indices = y_val
        if X_test is not None and y_test is not None:
            y_test_indices = y_test

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_indices, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_indices, dtype=torch.long, device=device)

    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test_indices, dtype=torch.long, device=device)

    # Create MLP classifier
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train_indices))
    clf = MLPClassifier(input_dim, num_classes, hidden_dim=32).to(device)

    # Training setup
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        # Training
        clf.train()
        optimizer.zero_grad()
        logits = clf(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
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
        # Train accuracy
        train_preds = clf(X_train_t).argmax(dim=1).cpu().numpy()
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
    1. za alone vs za+zay for domain classification
    2. zy alone vs zy+zay for label classification
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
    train_za = train_data['za'].numpy()
    train_zay = train_data['zay'].numpy() if train_data['zay'] is not None else None
    train_y = train_data['y'].numpy()
    train_a = train_data['a'].numpy()
    #print(f"train data shape: {train_zy.shape}, {train_za.shape}, {train_zay.shape}, {train_y.shape}, {train_a.shape}")
    
    val_zy = val_data['zy'].numpy()
    val_za = val_data['za'].numpy()
    val_zay = val_data['zay'].numpy() if val_data['zay'] is not None else None
    val_y = val_data['y'].numpy()
    val_a = val_data['a'].numpy()
    
    test_zy = test_data['zy'].numpy()
    test_za = test_data['za'].numpy()
    test_zay = test_data['zay'].numpy() if test_data['zay'] is not None else None
    test_y = test_data['y'].numpy()
    test_a = test_data['a'].numpy()

    results = {}
    
    # Skip zay experiments if model doesn't have zay (DIVA case)
    has_zay = train_zay is not None and train_zay.shape[1] > 0
    
    print(f"📊 Training classifiers...")
    print(f"   - zy dim: {train_zy.shape[1]}")
    print(f"   - za dim: {train_za.shape[1]}")
    if has_zay:
        print(f"   - zay dim: {train_zay.shape[1]}")
    else:
        print(f"   - zay: Not available (DIVA model)")
    
    # =============================================================================
    # DOMAIN CLASSIFICATION (predict 'a' from latent variables)
    # =============================================================================
    print("\n🎯 Domain Classification Experiments:")
    
    # 1. Domain classification using za alone
    print("   Training za → domain classifier...")

    za_train_acc, za_val_acc, za_test_acc, _ = train_pytorch_classifier(
        X_train=train_za, y_train=train_a, X_val=val_za, y_val=val_a, X_test=test_za, y_test=test_a
    )
    results['domain_za_alone'] = {'train_acc': za_train_acc, 'val_acc': za_val_acc, 'test_acc': za_test_acc}
    
    # 2. Domain classification using zy alone (cross-prediction test)
    print("   Training zy → domain classifier...")
    zy_d_train_acc, zy_d_val_acc, zy_d_test_acc, _ = train_pytorch_classifier(
        X_train=train_zy, y_train=train_a, X_val=val_zy, y_val=val_a, X_test=test_zy, y_test=test_a
    )
    results['domain_zy_alone'] = {'train_acc': zy_d_train_acc, 'val_acc': zy_d_val_acc, 'test_acc': zy_d_test_acc}

    if has_zay:
        # 3. Domain classification using za+zay
        print("   Training za+zay → domain classifier...")
        train_za_zay = np.concatenate([train_za, train_zay], axis=1)
        val_za_zay = np.concatenate([val_za, val_zay], axis=1)
        test_za_zay = np.concatenate([test_za, test_zay], axis=1)

        za_zay_train_acc, za_zay_val_acc, za_zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_za_zay, y_train=train_a, X_val=val_za_zay, y_val=val_a, X_test=test_za_zay, y_test=test_a
        )
        results['domain_za_zay'] = {'train_acc': za_zay_train_acc, 'val_acc': za_zay_val_acc, 'test_acc': za_zay_test_acc}

        # 4. Domain classification using zay alone (for comparison)
        print("   Training zay → domain classifier...")
        zay_train_acc, zay_val_acc, zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_zay, y_train=train_a, X_val=val_zay, y_val=val_a, X_test=test_zay, y_test=test_a
        )
        results['domain_zay_alone'] = {'train_acc': zay_train_acc, 'val_acc': zay_val_acc, 'test_acc': zay_test_acc}
    
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

    # 2. Label classification using za alone (cross-prediction test)
    print("   Training za → label classifier...")
    za_y_train_acc, za_y_val_acc, za_y_test_acc, _ = train_pytorch_classifier(
        X_train=train_za, y_train=train_y, X_val=val_za, y_val=val_y, X_test=test_za, y_test=test_y
    )
    results['label_za_alone'] = {'train_acc': za_y_train_acc, 'val_acc': za_y_val_acc, 'test_acc': za_y_test_acc}

    if has_zay:
        # 3. Label classification using zy+zay
        print("   Training zy+zay → label classifier...")
        train_zy_zay = np.concatenate([train_zy, train_zay], axis=1)
        val_zy_zay = np.concatenate([val_zy, val_zay], axis=1)
        test_zy_zay = np.concatenate([test_zy, test_zay], axis=1)

        zy_zay_train_acc, zy_zay_val_acc, zy_zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_zy_zay, y_train=train_y, X_val=val_zy_zay, y_val=val_y, X_test=test_zy_zay, y_test=test_y
        )
        results['label_zy_zay'] = {'train_acc': zy_zay_train_acc, 'val_acc': zy_zay_val_acc, 'test_acc': zy_zay_test_acc}

        # 4. Label classification using zay alone (for comparison)
        print("   Training zay → label classifier...")
        zay_y_train_acc, zay_y_val_acc, zay_y_test_acc, _ = train_pytorch_classifier(
            X_train=train_zay, y_train=train_y, X_val=val_zay, y_val=val_y, X_test=test_zay, y_test=test_y
        )
        results['label_zay_alone'] = {'train_acc': zay_y_train_acc, 'val_acc': zay_y_val_acc, 'test_acc': zay_y_test_acc}
    
    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    print("\n📈 EXPRESSIVENESS COMPARISON RESULTS:")
    print("="*60)
    
    print("\n🎯 DOMAIN CLASSIFICATION:")
    print(f"   za alone:      Train={results['domain_za_alone']['train_acc']:.4f}, Val={results['domain_za_alone']['val_acc']:.4f}, Test={results['domain_za_alone']['test_acc']:.4f}")
    print(f"   zy alone:      Train={results['domain_zy_alone']['train_acc']:.4f}, Val={results['domain_zy_alone']['val_acc']:.4f}, Test={results['domain_zy_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zay:
        print(f"   za+zay:        Train={results['domain_za_zay']['train_acc']:.4f}, Val={results['domain_za_zay']['val_acc']:.4f}, Test={results['domain_za_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['domain_zay_alone']['train_acc']:.4f}, Val={results['domain_zay_alone']['val_acc']:.4f}, Test={results['domain_zay_alone']['test_acc']:.4f}")

        domain_improvement_val = results['domain_za_zay']['val_acc'] - results['domain_za_alone']['val_acc']
        domain_improvement_test = results['domain_za_zay']['test_acc'] - results['domain_za_alone']['test_acc']
        baseline_val = results['domain_za_alone']['val_acc']
        baseline_test = results['domain_za_alone']['test_acc']
        domain_improvement_val_pct = (domain_improvement_val / baseline_val) * 100 if baseline_val > 0 else 0.0
        domain_improvement_test_pct = (domain_improvement_test / baseline_test) * 100 if baseline_test > 0 else 0.0
        print(f"   📊 IMPROVEMENT (Val): za+zay is {domain_improvement_val:.4f} ({domain_improvement_val_pct:.2f}%) better than za alone")
        print(f"   📊 IMPROVEMENT (Test): za+zay is {domain_improvement_test:.4f} ({domain_improvement_test_pct:.2f}%) better than za alone")
    
    print("\n🏷️  LABEL CLASSIFICATION:")
    print(f"   zy alone:      Train={results['label_zy_alone']['train_acc']:.4f}, Val={results['label_zy_alone']['val_acc']:.4f}, Test={results['label_zy_alone']['test_acc']:.4f}")
    print(f"   za alone:      Train={results['label_za_alone']['train_acc']:.4f}, Val={results['label_za_alone']['val_acc']:.4f}, Test={results['label_za_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zay:
        print(f"   zy+zay:        Train={results['label_zy_zay']['train_acc']:.4f}, Val={results['label_zy_zay']['val_acc']:.4f}, Test={results['label_zy_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['label_zay_alone']['train_acc']:.4f}, Val={results['label_zay_alone']['val_acc']:.4f}, Test={results['label_zay_alone']['test_acc']:.4f}")

        label_improvement_val = results['label_zy_zay']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zay']['test_acc'] - results['label_zy_alone']['test_acc']
        baseline_val = results['label_zy_alone']['val_acc']
        baseline_test = results['label_zy_alone']['test_acc']
        label_improvement_val_pct = (label_improvement_val / baseline_val) * 100 if baseline_val > 0 else 0.0
        label_improvement_test_pct = (label_improvement_test / baseline_test) * 100 if baseline_test > 0 else 0.0
        print(f"   📊 IMPROVEMENT (Val): zy+zay is {label_improvement_val:.4f} ({label_improvement_val_pct:.2f}%) better than zy alone")
        print(f"   📊 IMPROVEMENT (Test): zy+zay is {label_improvement_test:.4f} ({label_improvement_test_pct:.2f}%) better than zy alone")
    
    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    # Always create visualization (handles both NVAE and DIVA models)
    create_expressiveness_visualization(results, save_dir, has_zay)

    # Save results to JSON
    import json
    results_path = os.path.join(save_dir, 'latent_expressiveness_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")

    return results

def create_expressiveness_visualization(results, save_dir, has_zay):
    """Create visualization comparing the expressiveness of different latent combinations.

    Args:
        results: Dictionary containing classification results
        save_dir: Directory to save visualization
        has_zay: Boolean indicating if model has zay component (NVAE=True, DIVA=False)
    """

    # Prepare data for visualization - include both validation and test
    comparison_data = []

    # Domain classification comparisons
    for split, split_name in [('val_acc', 'Validation'), ('test_acc', 'Test')]:
        comparison_data.append({
            'Task': 'Domain Classification',
            'Method': 'za alone',
            'Accuracy': results['domain_za_alone'][split],
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

        # Only add zay-related metrics if model has zay component
        if has_zay:
            comparison_data.append({
                'Task': 'Domain Classification',
                'Method': 'za+zay',
                'Accuracy': results['domain_za_zay'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Domain Classification',
                'Method': 'zay alone',
                'Accuracy': results['domain_zay_alone'][split],
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
            'Method': 'za alone',
            'Accuracy': results['label_za_alone'][split],
            'Type': 'Cross-prediction',
            'Split': split_name
        })

        # Only add zay-related metrics if model has zay component
        if has_zay:
            comparison_data.append({
                'Task': 'Label Classification',
                'Method': 'zy+zay',
                'Accuracy': results['label_zy_zay'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Label Classification',
                'Method': 'zay alone',
                'Accuracy': results['label_zay_alone'][split],
                'Type': 'Individual',
                'Split': split_name
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Create visualization with validation and test results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate adaptive y-axis limit based on data
    max_acc = df['Accuracy'].max()
    y_max = min(1.0, max_acc + 0.05)  # Add 5% padding, but cap at 1.0

    # Choose colors based on whether model has zay component
    # NVAE (has_zay=True): 4 bars - orange, red, green, blue
    # DIVA (has_zay=False): 2 bars - orange, red
    bar_colors = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4'] if has_zay else ['#ff7f0e', '#d62728']

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
    
    # Only calculate and display improvements for models with zay component
    # (DIVA models don't have zay, so no improvement to show)
    if has_zay:
        # Calculate and display improvements
        domain_improvement_val = results['domain_za_zay']['val_acc'] - results['domain_za_alone']['val_acc']
        domain_improvement_test = results['domain_za_zay']['test_acc'] - results['domain_za_alone']['test_acc']
        label_improvement_val = results['label_zy_zay']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zay']['test_acc'] - results['label_zy_alone']['test_acc']

        # Calculate percentage improvements (relative to baseline) with zero protection
        domain_base_val = results['domain_za_alone']['val_acc']
        domain_base_test = results['domain_za_alone']['test_acc']
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def extract_latent_representations(model, dataloader, device, num_classes=2):
    """
    Extract latent representations from the trained VAE model.

    Args:
        model: Trained VAE model
        dataloader: DataLoader providing batches
        device: Device to run on (cpu/cuda)
        num_classes: Total number of classes (default 2 for WILD: Normal, Tumor)

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
                         model.name == 'dann')

    if is_augmented_dann:
        print("Detected AugmentedDANN model - using extract_features() for 3-space latent extraction")

    all_zy = []
    all_zx = []
    all_zay = []
    all_za = []
    all_y = []
    all_a = []

    with torch.no_grad():
        for batch_idx, (x, y, metadata) in enumerate(dataloader):
            x = x.to(device)
            # Extract hospital ID from metadata
            a = metadata[:, 0].long().to(device)

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
    """Train a logistic regression classifier using sklearn for classification tasks.

    Args:
        X_train: Training features
        y_train: Training labels (can be one-hot or indices)
        X_val: Validation features
        y_val: Validation labels (can be one-hot or indices)
        X_test: Test features (optional)
        y_test: Test labels (optional)

    Returns:
        tuple: (train_acc, val_acc, test_acc, trained_classifier)
    """
    from sklearn.linear_model import LogisticRegression

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

    # Train logistic regression classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train_indices)

    # Evaluate on train set
    train_preds = clf.predict(X_train)
    train_acc = accuracy_score(y_train_indices, train_preds)

    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val_indices, val_preds)

    # Evaluate on test set (if provided)
    test_acc = None
    if X_test is not None and y_test is not None:
        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(y_test_indices, test_preds)

    return train_acc, val_acc, test_acc, clf

def evaluate_latent_expressiveness(model, train_loader, val_loader, test_loader, device, save_dir):
    """
    Evaluate the expressiveness of different latent variable combinations.

    Compare:
    1. za alone vs za+zay for hospital classification
    2. zy alone vs zy+zay for tumor classification
    """

    print("🔍 Extracting latent representations...")

    # Extract latent representations
    # Use model's y_dim if available, otherwise default to 2 (WILD: Normal, Tumor)
    num_y_classes = model.y_dim if hasattr(model, 'y_dim') else 2
    train_data = extract_latent_representations(model, train_loader, device, num_y_classes)
    val_data = extract_latent_representations(model, val_loader, device, num_y_classes)
    test_data = extract_latent_representations(model, test_loader, device, num_y_classes)
    # Convert to numpy for sklearn compatibility
    train_zy = train_data['zy'].numpy()
    train_za = train_data['za'].numpy()
    train_zay = train_data['zay'].numpy() if train_data['zay'] is not None else None
    train_y = train_data['y'].numpy()
    train_a = train_data['a'].numpy()

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
    # HOSPITAL CLASSIFICATION (predict 'a' from latent variables)
    # =============================================================================
    print("\n🏥 Hospital Classification Experiments:")

    # 1. Hospital classification using za alone
    print("   Training za → hospital classifier...")

    za_train_acc, za_val_acc, za_test_acc, _ = train_pytorch_classifier(
        X_train=train_za, y_train=train_a, X_val=val_za, y_val=val_a, X_test=test_za, y_test=test_a
    )
    results['hospital_za_alone'] = {'train_acc': za_train_acc, 'val_acc': za_val_acc, 'test_acc': za_test_acc}

    if has_zay:
        # 2. Hospital classification using za+zay
        print("   Training za+zay → hospital classifier...")
        train_za_zay = np.concatenate([train_za, train_zay], axis=1)
        val_za_zay = np.concatenate([val_za, val_zay], axis=1)
        test_za_zay = np.concatenate([test_za, test_zay], axis=1)

        za_zay_train_acc, za_zay_val_acc, za_zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_za_zay, y_train=train_a, X_val=val_za_zay, y_val=val_a, X_test=test_za_zay, y_test=test_a
        )
        results['hospital_za_zay'] = {'train_acc': za_zay_train_acc, 'val_acc': za_zay_val_acc, 'test_acc': za_zay_test_acc}

        # 3. Hospital classification using zay alone (for comparison)
        print("   Training zay → hospital classifier...")
        zay_train_acc, zay_val_acc, zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_zay, y_train=train_a, X_val=val_zay, y_val=val_a, X_test=test_zay, y_test=test_a
        )
        results['hospital_zay_alone'] = {'train_acc': zay_train_acc, 'val_acc': zay_val_acc, 'test_acc': zay_test_acc}

    # =============================================================================
    # TUMOR CLASSIFICATION (predict 'y' from latent variables)
    # =============================================================================
    print("\n🔬 Tumor Classification Experiments:")

    # 1. Tumor classification using zy alone
    print("   Training zy → tumor classifier...")
    zy_train_acc, zy_val_acc, zy_test_acc, _ = train_pytorch_classifier(
        X_train=train_zy, y_train=train_y, X_val=val_zy, y_val=val_y, X_test=test_zy, y_test=test_y
    )
    results['tumor_zy_alone'] = {'train_acc': zy_train_acc, 'val_acc': zy_val_acc, 'test_acc': zy_test_acc}

    if has_zay:
        # 2. Tumor classification using zy+zay
        print("   Training zy+zay → tumor classifier...")
        train_zy_zay = np.concatenate([train_zy, train_zay], axis=1)
        val_zy_zay = np.concatenate([val_zy, val_zay], axis=1)
        test_zy_zay = np.concatenate([test_zy, test_zay], axis=1)

        zy_zay_train_acc, zy_zay_val_acc, zy_zay_test_acc, _ = train_pytorch_classifier(
            X_train=train_zy_zay, y_train=train_y, X_val=val_zy_zay, y_val=val_y, X_test=test_zy_zay, y_test=test_y
        )
        results['tumor_zy_zay'] = {'train_acc': zy_zay_train_acc, 'val_acc': zy_zay_val_acc, 'test_acc': zy_zay_test_acc}

        # 3. Tumor classification using zay alone (for comparison)
        print("   Training zay → tumor classifier...")
        zay_y_train_acc, zay_y_val_acc, zay_y_test_acc, _ = train_pytorch_classifier(
            X_train=train_zay, y_train=train_y, X_val=val_zay, y_val=val_y, X_test=test_zay, y_test=test_y
        )
        results['tumor_zay_alone'] = {'train_acc': zay_y_train_acc, 'val_acc': zay_y_val_acc, 'test_acc': zay_y_test_acc}

    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    print("\n📈 EXPRESSIVENESS COMPARISON RESULTS:")
    print("="*60)

    print("\n🏥 HOSPITAL CLASSIFICATION:")
    print(f"   za alone:      Train={results['hospital_za_alone']['train_acc']:.4f}, Val={results['hospital_za_alone']['val_acc']:.4f}, Test={results['hospital_za_alone']['test_acc']:.4f}")
    if has_zay:
        print(f"   za+zay:        Train={results['hospital_za_zay']['train_acc']:.4f}, Val={results['hospital_za_zay']['val_acc']:.4f}, Test={results['hospital_za_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['hospital_zay_alone']['train_acc']:.4f}, Val={results['hospital_zay_alone']['val_acc']:.4f}, Test={results['hospital_zay_alone']['test_acc']:.4f}")

        hospital_improvement_val = results['hospital_za_zay']['val_acc'] - results['hospital_za_alone']['val_acc']
        hospital_improvement_test = results['hospital_za_zay']['test_acc'] - results['hospital_za_alone']['test_acc']
        hospital_improvement_val_pct = (hospital_improvement_val / results['hospital_za_alone']['val_acc']) * 100
        hospital_improvement_test_pct = (hospital_improvement_test / results['hospital_za_alone']['test_acc']) * 100
        print(f"   📊 IMPROVEMENT (Val): za+zay is {hospital_improvement_val:.4f} ({hospital_improvement_val_pct:.2f}%) better than za alone")
        print(f"   📊 IMPROVEMENT (Test): za+zay is {hospital_improvement_test:.4f} ({hospital_improvement_test_pct:.2f}%) better than za alone")

    print("\n🔬 TUMOR CLASSIFICATION:")
    print(f"   zy alone:      Train={results['tumor_zy_alone']['train_acc']:.4f}, Val={results['tumor_zy_alone']['val_acc']:.4f}, Test={results['tumor_zy_alone']['test_acc']:.4f}")
    if has_zay:
        print(f"   zy+zay:        Train={results['tumor_zy_zay']['train_acc']:.4f}, Val={results['tumor_zy_zay']['val_acc']:.4f}, Test={results['tumor_zy_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['tumor_zay_alone']['train_acc']:.4f}, Val={results['tumor_zay_alone']['val_acc']:.4f}, Test={results['tumor_zay_alone']['test_acc']:.4f}")

        tumor_improvement_val = results['tumor_zy_zay']['val_acc'] - results['tumor_zy_alone']['val_acc']
        tumor_improvement_test = results['tumor_zy_zay']['test_acc'] - results['tumor_zy_alone']['test_acc']
        tumor_improvement_val_pct = (tumor_improvement_val / results['tumor_zy_alone']['val_acc']) * 100
        tumor_improvement_test_pct = (tumor_improvement_test / results['tumor_zy_alone']['test_acc']) * 100
        print(f"   📊 IMPROVEMENT (Val): zy+zay is {tumor_improvement_val:.4f} ({tumor_improvement_val_pct:.2f}%) better than zy alone")
        print(f"   📊 IMPROVEMENT (Test): zy+zay is {tumor_improvement_test:.4f} ({tumor_improvement_test_pct:.2f}%) better than zy alone")

    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    if has_zay:
        create_expressiveness_visualization(results, save_dir)

    # Save results to JSON
    import json
    results_path = os.path.join(save_dir, 'latent_expressiveness_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")

    return results

def create_expressiveness_visualization(results, save_dir):
    """Create visualization comparing the expressiveness of different latent combinations."""

    # Prepare data for visualization - include both validation and test
    comparison_data = []

    # Hospital classification comparisons
    for split, split_name in [('val_acc', 'Validation'), ('test_acc', 'Test')]:
        comparison_data.append({
            'Task': 'Hospital Classification',
            'Method': 'za alone',
            'Accuracy': results['hospital_za_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Hospital Classification',
            'Method': 'za+zay',
            'Accuracy': results['hospital_za_zay'][split],
            'Type': 'Combined',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Hospital Classification',
            'Method': 'zay alone',
            'Accuracy': results['hospital_zay_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })

        # Tumor classification comparisons
        comparison_data.append({
            'Task': 'Tumor Classification',
            'Method': 'zy alone',
            'Accuracy': results['tumor_zy_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Tumor Classification',
            'Method': 'zy+zay',
            'Accuracy': results['tumor_zy_zay'][split],
            'Type': 'Combined',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Tumor Classification',
            'Method': 'zay alone',
            'Accuracy': results['tumor_zay_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })

    df = pd.DataFrame(comparison_data)

    # Create visualization with validation and test results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate adaptive y-axis limit based on data
    max_acc = df['Accuracy'].max()
    y_max = min(1.0, max_acc + 0.05)  # Add 5% padding, but cap at 1.0

    # Hospital classification - Validation
    hospital_val_data = df[(df['Task'] == 'Hospital Classification') & (df['Split'] == 'Validation')]
    bars1 = axes[0,0].bar(hospital_val_data['Method'], hospital_val_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0,0].set_title('Hospital Classification - Validation', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0,0].set_ylim(0, y_max)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Hospital classification - Test
    hospital_test_data = df[(df['Task'] == 'Hospital Classification') & (df['Split'] == 'Test')]
    bars2 = axes[0,1].bar(hospital_test_data['Method'], hospital_test_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0,1].set_title('Hospital Classification - Test', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0,1].set_ylim(0, y_max)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Tumor classification - Validation
    tumor_val_data = df[(df['Task'] == 'Tumor Classification') & (df['Split'] == 'Validation')]
    bars3 = axes[1,0].bar(tumor_val_data['Method'], tumor_val_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[1,0].set_title('Tumor Classification - Validation', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1,0].set_ylim(0, y_max)

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Tumor classification - Test
    tumor_test_data = df[(df['Task'] == 'Tumor Classification') & (df['Split'] == 'Test')]
    bars4 = axes[1,1].bar(tumor_test_data['Method'], tumor_test_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[1,1].set_title('Tumor Classification - Test', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1,1].set_ylim(0, y_max)

    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Calculate and display improvements
    hospital_improvement_val = results['hospital_za_zay']['val_acc'] - results['hospital_za_alone']['val_acc']
    hospital_improvement_test = results['hospital_za_zay']['test_acc'] - results['hospital_za_alone']['test_acc']
    tumor_improvement_val = results['tumor_zy_zay']['val_acc'] - results['tumor_zy_alone']['val_acc']
    tumor_improvement_test = results['tumor_zy_zay']['test_acc'] - results['tumor_zy_alone']['test_acc']

    # Calculate percentage improvements (relative to baseline)
    hospital_improvement_val_pct = (hospital_improvement_val / results['hospital_za_alone']['val_acc']) * 100
    hospital_improvement_test_pct = (hospital_improvement_test / results['hospital_za_alone']['test_acc']) * 100
    tumor_improvement_val_pct = (tumor_improvement_val / results['tumor_zy_alone']['val_acc']) * 100
    tumor_improvement_test_pct = (tumor_improvement_test / results['tumor_zy_alone']['test_acc']) * 100

    # Add improvement annotations
    axes[0,0].annotate(f'Val Improvement:\n+{hospital_improvement_val:.3f} ({hospital_improvement_val_pct:.1f}%)',
                      xy=(1, results['hospital_za_zay']['val_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')

    axes[0,1].annotate(f'Test Improvement:\n+{hospital_improvement_test:.3f} ({hospital_improvement_test_pct:.1f}%)',
                      xy=(1, results['hospital_za_zay']['test_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')

    axes[1,0].annotate(f'Val Improvement:\n+{tumor_improvement_val:.3f} ({tumor_improvement_val_pct:.1f}%)',
                      xy=(1, results['tumor_zy_zay']['val_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')

    axes[1,1].annotate(f'Test Improvement:\n+{tumor_improvement_test:.3f} ({tumor_improvement_test_pct:.1f}%)',
                      xy=(1, results['tumor_zy_zay']['test_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, 'latent_expressiveness_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Expressiveness comparison plot saved to: {plot_path}")

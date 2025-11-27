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
        dict: Contains zy, zx, zdy, zd representations and corresponding y, a labels
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
    all_zdy = []
    all_zd = []
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
                # - zd stays as zd (domain-specific)
                # - zdy stays as zdy (interaction)
                # - zx is empty (no residual space in discriminative models - no decoder)
                zx = torch.zeros(zy.shape[0], 0, device=zy.device)
                # zd and zdy already have correct names from extract_features()
            else:
                # VAE models (NVAE/DIVA) - use standard forward pass
                x_recon, z, qz, pzy, pzx, pzd, pzdy, y_hat, a_hat, zy, zx, zdy, zd = model(y, x, a)

            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            if zdy is not None and zdy.shape[1] > 0:
                all_zdy.append(zdy.cpu())
            else:
                # For DIVA models or empty zdy, create zeros on CPU (matching other tensors)
                all_zdy.append(torch.zeros(zy.shape[0], 0, device='cpu'))
            all_zd.append(zd.cpu())
            all_y.append(y.cpu())
            all_a.append(a.cpu())

    # Validate we processed at least one batch
    if len(all_zy) == 0:
        raise ValueError(
            "❌ No batches were processed from dataloader. This likely means:\n"
            "   - Dataloader is empty (0 samples)\n"
            "   - All data was filtered out during preprocessing\n"
            "   - Check your data loading and filtering settings"
        )

    # Concatenate all batches
    latent_data = {
        'zy': torch.cat(all_zy, dim=0),
        'zx': torch.cat(all_zx, dim=0),
        'zdy': torch.cat(all_zdy, dim=0) if len(all_zdy) > 0 and all_zdy[0].shape[1] > 0 else None,
        'zd': torch.cat(all_zd, dim=0),
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
    1. zd alone vs zd+zdy for hospital classification
    2. zy alone vs zy+zdy for tumor classification
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
    train_zd = train_data['zd'].numpy()
    train_zdy = train_data['zdy'].numpy() if train_data['zdy'] is not None else None
    train_y = train_data['y'].numpy()
    train_a = train_data['a'].numpy()

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
    print(f"   - zy dim: {train_zy.shape[1]}")
    print(f"   - zd dim: {train_zd.shape[1]}")
    if has_zdy:
        print(f"   - zdy dim: {train_zdy.shape[1]}")
    else:
        print(f"   - zdy: Not available (DIVA model)")

    # =============================================================================
    # HOSPITAL CLASSIFICATION (predict 'a' from latent variables)
    # =============================================================================
    print("\n🏥 Hospital Classification Experiments:")

    # 1. Hospital classification using zd alone
    print("   Training zd → hospital classifier...")

    za_train_acc, za_val_acc, za_test_acc, _ = train_pytorch_classifier(
        X_train=train_zd, y_train=train_a, X_val=val_zd, y_val=val_a, X_test=test_zd, y_test=test_a
    )
    results['hospital_zd_alone'] = {'train_acc': za_train_acc, 'val_acc': za_val_acc, 'test_acc': za_test_acc}

    # 2. Hospital classification using zy alone (cross-prediction test)
    print("   Training zy → hospital classifier...")
    zy_h_train_acc, zy_h_val_acc, zy_h_test_acc, _ = train_pytorch_classifier(
        X_train=train_zy, y_train=train_a, X_val=val_zy, y_val=val_a, X_test=test_zy, y_test=test_a
    )
    results['hospital_zy_alone'] = {'train_acc': zy_h_train_acc, 'val_acc': zy_h_val_acc, 'test_acc': zy_h_test_acc}

    if has_zdy:
        # 3. Hospital classification using zd+zdy
        print("   Training zd+zdy → hospital classifier...")
        train_zd_zdy = np.concatenate([train_zd, train_zdy], axis=1)
        val_zd_zdy = np.concatenate([val_zd, val_zdy], axis=1)
        test_zd_zdy = np.concatenate([test_zd, test_zdy], axis=1)

        zd_zdy_train_acc, zd_zdy_val_acc, zd_zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zd_zdy, y_train=train_a, X_val=val_zd_zdy, y_val=val_a, X_test=test_zd_zdy, y_test=test_a
        )
        results['hospital_zd_zdy'] = {'train_acc': zd_zdy_train_acc, 'val_acc': zd_zdy_val_acc, 'test_acc': zd_zdy_test_acc}

        # 4. Hospital classification using zdy alone (for comparison)
        print("   Training zdy → hospital classifier...")
        zdy_train_acc, zdy_val_acc, zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zdy, y_train=train_a, X_val=val_zdy, y_val=val_a, X_test=test_zdy, y_test=test_a
        )
        results['hospital_zdy_alone'] = {'train_acc': zdy_train_acc, 'val_acc': zdy_val_acc, 'test_acc': zdy_test_acc}

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

    # 2. Tumor classification using zd alone (cross-prediction test)
    print("   Training zd → tumor classifier...")
    za_t_train_acc, za_t_val_acc, za_t_test_acc, _ = train_pytorch_classifier(
        X_train=train_zd, y_train=train_y, X_val=val_zd, y_val=val_y, X_test=test_zd, y_test=test_y
    )
    results['tumor_zd_alone'] = {'train_acc': za_t_train_acc, 'val_acc': za_t_val_acc, 'test_acc': za_t_test_acc}

    if has_zdy:
        # 3. Tumor classification using zy+zdy
        print("   Training zy+zdy → tumor classifier...")
        train_zy_zdy = np.concatenate([train_zy, train_zdy], axis=1)
        val_zy_zdy = np.concatenate([val_zy, val_zdy], axis=1)
        test_zy_zdy = np.concatenate([test_zy, test_zdy], axis=1)

        zy_zdy_train_acc, zy_zdy_val_acc, zy_zdy_test_acc, _ = train_pytorch_classifier(
            X_train=train_zy_zdy, y_train=train_y, X_val=val_zy_zdy, y_val=val_y, X_test=test_zy_zdy, y_test=test_y
        )
        results['tumor_zy_zdy'] = {'train_acc': zy_zdy_train_acc, 'val_acc': zy_zdy_val_acc, 'test_acc': zy_zdy_test_acc}

        # 4. Tumor classification using zdy alone (for comparison)
        print("   Training zdy → tumor classifier...")
        zdy_y_train_acc, zdy_y_val_acc, zdy_y_test_acc, _ = train_pytorch_classifier(
            X_train=train_zdy, y_train=train_y, X_val=val_zdy, y_val=val_y, X_test=test_zdy, y_test=test_y
        )
        results['tumor_zdy_alone'] = {'train_acc': zdy_y_train_acc, 'val_acc': zdy_y_val_acc, 'test_acc': zdy_y_test_acc}

    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    print("\n📈 EXPRESSIVENESS COMPARISON RESULTS:")
    print("="*60)

    print("\n🏥 HOSPITAL CLASSIFICATION:")
    print(f"   zd alone:      Train={results['hospital_zd_alone']['train_acc']:.4f}, Val={results['hospital_zd_alone']['val_acc']:.4f}, Test={results['hospital_zd_alone']['test_acc']:.4f}")
    print(f"   zy alone:      Train={results['hospital_zy_alone']['train_acc']:.4f}, Val={results['hospital_zy_alone']['val_acc']:.4f}, Test={results['hospital_zy_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zdy:
        print(f"   zd+zdy:        Train={results['hospital_zd_zdy']['train_acc']:.4f}, Val={results['hospital_zd_zdy']['val_acc']:.4f}, Test={results['hospital_zd_zdy']['test_acc']:.4f}")
        print(f"   zdy alone:     Train={results['hospital_zdy_alone']['train_acc']:.4f}, Val={results['hospital_zdy_alone']['val_acc']:.4f}, Test={results['hospital_zdy_alone']['test_acc']:.4f}")

        hospital_improvement_val = results['hospital_zd_zdy']['val_acc'] - results['hospital_zd_alone']['val_acc']
        hospital_improvement_test = results['hospital_zd_zdy']['test_acc'] - results['hospital_zd_alone']['test_acc']

        # Handle division by zero for percentage calculations
        if results['hospital_zd_alone']['val_acc'] > 0:
            hospital_improvement_val_pct = (hospital_improvement_val / results['hospital_zd_alone']['val_acc']) * 100
            print(f"   📊 IMPROVEMENT (Val): zd+zdy is {hospital_improvement_val:.4f} ({hospital_improvement_val_pct:.2f}%) better than zd alone")
        else:
            print(f"   📊 IMPROVEMENT (Val): zd+zdy is {hospital_improvement_val:.4f} better than zd alone")

        if results['hospital_zd_alone']['test_acc'] > 0:
            hospital_improvement_test_pct = (hospital_improvement_test / results['hospital_zd_alone']['test_acc']) * 100
            print(f"   📊 IMPROVEMENT (Test): zd+zdy is {hospital_improvement_test:.4f} ({hospital_improvement_test_pct:.2f}%) better than zd alone")
        else:
            print(f"   📊 IMPROVEMENT (Test): zd+zdy is {hospital_improvement_test:.4f} better than zd alone (baseline: 0)")

    print("\n🔬 TUMOR CLASSIFICATION:")
    print(f"   zy alone:      Train={results['tumor_zy_alone']['train_acc']:.4f}, Val={results['tumor_zy_alone']['val_acc']:.4f}, Test={results['tumor_zy_alone']['test_acc']:.4f}")
    print(f"   zd alone:      Train={results['tumor_zd_alone']['train_acc']:.4f}, Val={results['tumor_zd_alone']['val_acc']:.4f}, Test={results['tumor_zd_alone']['test_acc']:.4f} [cross-prediction]")
    if has_zdy:
        print(f"   zy+zdy:        Train={results['tumor_zy_zdy']['train_acc']:.4f}, Val={results['tumor_zy_zdy']['val_acc']:.4f}, Test={results['tumor_zy_zdy']['test_acc']:.4f}")
        print(f"   zdy alone:     Train={results['tumor_zdy_alone']['train_acc']:.4f}, Val={results['tumor_zdy_alone']['val_acc']:.4f}, Test={results['tumor_zdy_alone']['test_acc']:.4f}")

        tumor_improvement_val = results['tumor_zy_zdy']['val_acc'] - results['tumor_zy_alone']['val_acc']
        tumor_improvement_test = results['tumor_zy_zdy']['test_acc'] - results['tumor_zy_alone']['test_acc']

        # Handle division by zero for percentage calculations
        if results['tumor_zy_alone']['val_acc'] > 0:
            tumor_improvement_val_pct = (tumor_improvement_val / results['tumor_zy_alone']['val_acc']) * 100
            print(f"   📊 IMPROVEMENT (Val): zy+zdy is {tumor_improvement_val:.4f} ({tumor_improvement_val_pct:.2f}%) better than zy alone")
        else:
            print(f"   📊 IMPROVEMENT (Val): zy+zdy is {tumor_improvement_val:.4f} better than zy alone")

        if results['tumor_zy_alone']['test_acc'] > 0:
            tumor_improvement_test_pct = (tumor_improvement_test / results['tumor_zy_alone']['test_acc']) * 100
            print(f"   📊 IMPROVEMENT (Test): zy+zdy is {tumor_improvement_test:.4f} ({tumor_improvement_test_pct:.2f}%) better than zy alone")
        else:
            print(f"   📊 IMPROVEMENT (Test): zy+zdy is {tumor_improvement_test:.4f} better than zy alone (baseline: 0)")

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

    # Hospital classification comparisons
    for split, split_name in [('val_acc', 'Validation'), ('test_acc', 'Test')]:
        comparison_data.append({
            'Task': 'Hospital Classification',
            'Method': 'zd alone',
            'Accuracy': results['hospital_zd_alone'][split],
            'Type': 'Individual',
            'Split': split_name
        })
        comparison_data.append({
            'Task': 'Hospital Classification',
            'Method': 'zy alone',
            'Accuracy': results['hospital_zy_alone'][split],
            'Type': 'Cross-prediction',
            'Split': split_name
        })

        # Only add zdy-related metrics if model has zdy component
        if has_zdy:
            comparison_data.append({
                'Task': 'Hospital Classification',
                'Method': 'zd+zdy',
                'Accuracy': results['hospital_zd_zdy'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Hospital Classification',
                'Method': 'zdy alone',
                'Accuracy': results['hospital_zdy_alone'][split],
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
            'Method': 'zd alone',
            'Accuracy': results['tumor_zd_alone'][split],
            'Type': 'Cross-prediction',
            'Split': split_name
        })

        # Only add zdy-related metrics if model has zdy component
        if has_zdy:
            comparison_data.append({
                'Task': 'Tumor Classification',
                'Method': 'zy+zdy',
                'Accuracy': results['tumor_zy_zdy'][split],
                'Type': 'Combined',
                'Split': split_name
            })
            comparison_data.append({
                'Task': 'Tumor Classification',
                'Method': 'zdy alone',
                'Accuracy': results['tumor_zdy_alone'][split],
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

    # Hospital classification - Validation
    hospital_val_data = df[(df['Task'] == 'Hospital Classification') & (df['Split'] == 'Validation')]
    bars1 = axes[0,0].bar(hospital_val_data['Method'], hospital_val_data['Accuracy'], color=bar_colors)
    axes[0,0].set_title('Hospital Classification - Validation', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0,0].set_ylim(0, y_max)
    axes[0,0].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (3 training hospitals: 1/3 = 0.333)
    axes[0,0].axhline(y=0.333, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (33%)')
    axes[0,0].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Hospital classification - Test
    hospital_test_data = df[(df['Task'] == 'Hospital Classification') & (df['Split'] == 'Test')]
    bars2 = axes[0,1].bar(hospital_test_data['Method'], hospital_test_data['Accuracy'], color=bar_colors)
    axes[0,1].set_title('Hospital Classification - Test', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0,1].set_ylim(0, y_max)
    axes[0,1].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (3 training hospitals: 1/3 = 0.333)
    axes[0,1].axhline(y=0.333, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (33%)')
    axes[0,1].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Tumor classification - Validation
    tumor_val_data = df[(df['Task'] == 'Tumor Classification') & (df['Split'] == 'Validation')]
    bars3 = axes[1,0].bar(tumor_val_data['Method'], tumor_val_data['Accuracy'], color=bar_colors)
    axes[1,0].set_title('Tumor Classification - Validation', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1,0].set_ylim(0, y_max)
    axes[1,0].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (2 tumor classes: 1/2 = 0.50)
    axes[1,0].axhline(y=0.50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (50%)')
    axes[1,0].legend(loc='upper right', fontsize=9)

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Tumor classification - Test
    tumor_test_data = df[(df['Task'] == 'Tumor Classification') & (df['Split'] == 'Test')]
    bars4 = axes[1,1].bar(tumor_test_data['Method'], tumor_test_data['Accuracy'], color=bar_colors)
    axes[1,1].set_title('Tumor Classification - Test', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1,1].set_ylim(0, y_max)
    axes[1,1].tick_params(axis='x', rotation=15)

    # Add random guessing baseline (2 tumor classes: 1/2 = 0.50)
    axes[1,1].axhline(y=0.50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Guess (50%)')
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
        hospital_improvement_val = results['hospital_zd_zdy']['val_acc'] - results['hospital_zd_alone']['val_acc']
        hospital_improvement_test = results['hospital_zd_zdy']['test_acc'] - results['hospital_zd_alone']['test_acc']
        tumor_improvement_val = results['tumor_zy_zdy']['val_acc'] - results['tumor_zy_alone']['val_acc']
        tumor_improvement_test = results['tumor_zy_zdy']['test_acc'] - results['tumor_zy_alone']['test_acc']

        # Calculate percentage improvements (relative to baseline)
        # Handle division by zero for percentage calculations
        if results['hospital_zd_alone']['val_acc'] > 0:
            hospital_improvement_val_pct = (hospital_improvement_val / results['hospital_zd_alone']['val_acc']) * 100
        else:
            hospital_improvement_val_pct = 0

        if results['hospital_zd_alone']['test_acc'] > 0:
            hospital_improvement_test_pct = (hospital_improvement_test / results['hospital_zd_alone']['test_acc']) * 100
        else:
            hospital_improvement_test_pct = 0

        if results['tumor_zy_alone']['val_acc'] > 0:
            tumor_improvement_val_pct = (tumor_improvement_val / results['tumor_zy_alone']['val_acc']) * 100
        else:
            tumor_improvement_val_pct = 0

        if results['tumor_zy_alone']['test_acc'] > 0:
            tumor_improvement_test_pct = (tumor_improvement_test / results['tumor_zy_alone']['test_acc']) * 100
        else:
            tumor_improvement_test_pct = 0

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot
    plot_path = os.path.join(save_dir, 'latent_expressiveness_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Expressiveness comparison plot saved to: {plot_path}")

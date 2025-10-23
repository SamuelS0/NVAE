import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class LatentClassifier(nn.Module):
    """Simple MLP classifier for latent representations."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        '''self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )'''
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def extract_latent_representations(model, dataloader, device):
    """
    Extract latent representations from the trained VAE model.
    
    Returns:
        dict: Contains zy, zx, zay, za representations and corresponding y, a labels
    """
    model.eval()
    
    all_zy = []
    all_zx = []
    all_zay = []
    all_za = []
    all_y = []
    all_a = []
    
    with torch.no_grad():
        for batch_idx, (x, y, color,  a) in enumerate(dataloader):
            x, a = x.to(device), a.to(device)
            y_onehot = F.one_hot(y.long(), num_classes=10).float()  # Convert 0-9 labels to one-hot
            y_onehot = y_onehot.to(device)
            y = y_onehot
            # Forward pass through the model to get latent representations
            x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model(y, x, a)
            
            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            if zay is not None:
                all_zay.append(zay.cpu())
            else:
                # For DIVA models, zay is None, so create zeros
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

def train_pytorch_classifier(X_train, y_train, X_val, y_val, X_test=None, y_test=None, num_classes=None, device=None, epochs=500):
    """Train a linear regression classifier using PyTorch."""
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
    1. za alone vs za+zay for domain classification
    2. zy alone vs zy+zay for label classification
    """
    
    print("🔍 Extracting latent representations...")
    
    # Extract latent representations
    train_data = extract_latent_representations(model, train_loader, device)
    val_data = extract_latent_representations(model, val_loader, device)
    test_data = extract_latent_representations(model, test_loader, device)
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
        X_train = train_za, y_train = train_a, X_val = val_za, y_val = val_a, X_test = test_za, y_test = test_a,
        num_classes=train_a.shape[1]
    )
    results['domain_za_alone'] = {'train_acc': za_train_acc, 'val_acc': za_val_acc, 'test_acc': za_test_acc}
    
    if has_zay:
        # 2. Domain classification using za+zay
        print("   Training za+zay → domain classifier...")
        train_za_zay = np.concatenate([train_za, train_zay], axis=1)
        val_za_zay = np.concatenate([val_za, val_zay], axis=1)
        test_za_zay = np.concatenate([test_za, test_zay], axis=1)
        
        za_zay_train_acc, za_zay_val_acc, za_zay_test_acc, _ = train_pytorch_classifier(
            X_train = train_za_zay, y_train = train_a, X_val = val_za_zay, y_val = val_a, X_test = test_za_zay, y_test = test_a,
            num_classes=train_a.shape[1]
        )
        results['domain_za_zay'] = {'train_acc': za_zay_train_acc, 'val_acc': za_zay_val_acc, 'test_acc': za_zay_test_acc}
        
        # 3. Domain classification using zay alone (for comparison)
        print("   Training zay → domain classifier...")
        zay_train_acc, zay_val_acc, zay_test_acc, _ = train_pytorch_classifier(
            X_train = train_zay, y_train = train_a, X_val = val_zay, y_val = val_a, X_test = test_zay, y_test = test_a,
            num_classes=train_a.shape[1]
        )
        results['domain_zay_alone'] = {'train_acc': zay_train_acc, 'val_acc': zay_val_acc, 'test_acc': zay_test_acc}
    
    # =============================================================================
    # LABEL CLASSIFICATION (predict 'y' from latent variables)
    # =============================================================================
    print("\n🏷️  Label Classification Experiments:")
    
    # 1. Label classification using zy alone
    print("   Training zy → label classifier...")
    zy_train_acc, zy_val_acc, zy_test_acc, _ = train_pytorch_classifier(
        X_train = train_zy, y_train = train_y, X_val = val_zy, y_val = val_y, X_test = test_zy, y_test = test_y,
        num_classes=train_y.shape[1]
    )
    results['label_zy_alone'] = {'train_acc': zy_train_acc, 'val_acc': zy_val_acc, 'test_acc': zy_test_acc}
    
    if has_zay:
        # 2. Label classification using zy+zay
        print("   Training zy+zay → label classifier...")
        train_zy_zay = np.concatenate([train_zy, train_zay], axis=1)
        val_zy_zay = np.concatenate([val_zy, val_zay], axis=1)
        test_zy_zay = np.concatenate([test_zy, test_zay], axis=1)
        
        zy_zay_train_acc, zy_zay_val_acc, zy_zay_test_acc, _ = train_pytorch_classifier(
            X_train = train_zy_zay, y_train = train_y, X_val = val_zy_zay, y_val = val_y, X_test = test_zy_zay, y_test = test_y,
            num_classes=train_y.shape[1]
        )
        results['label_zy_zay'] = {'train_acc': zy_zay_train_acc, 'val_acc': zy_zay_val_acc, 'test_acc': zy_zay_test_acc}
        
        # 3. Label classification using zay alone (for comparison)
        print("   Training zay → label classifier...")
        zay_y_train_acc, zay_y_val_acc, zay_y_test_acc, _ = train_pytorch_classifier(
            X_train = train_zay, y_train = train_y, X_val = val_zay, y_val = val_y, X_test = test_zay, y_test = test_y,
            num_classes=train_y.shape[1]
        )
        results['label_zay_alone'] = {'train_acc': zay_y_train_acc, 'val_acc': zay_y_val_acc, 'test_acc': zay_y_test_acc}
    
    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    print("\n📈 EXPRESSIVENESS COMPARISON RESULTS:")
    print("="*60)
    
    print("\n🎯 DOMAIN CLASSIFICATION:")
    print(f"   za alone:      Train={results['domain_za_alone']['train_acc']:.4f}, Val={results['domain_za_alone']['val_acc']:.4f}, Test={results['domain_za_alone']['test_acc']:.4f}")
    if has_zay:
        print(f"   za+zay:        Train={results['domain_za_zay']['train_acc']:.4f}, Val={results['domain_za_zay']['val_acc']:.4f}, Test={results['domain_za_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['domain_zay_alone']['train_acc']:.4f}, Val={results['domain_zay_alone']['val_acc']:.4f}, Test={results['domain_zay_alone']['test_acc']:.4f}")
        
        domain_improvement_val = results['domain_za_zay']['val_acc'] - results['domain_za_alone']['val_acc']
        domain_improvement_test = results['domain_za_zay']['test_acc'] - results['domain_za_alone']['test_acc']
        print(f"   📊 IMPROVEMENT (Val): za+zay is {domain_improvement_val:.4f} ({domain_improvement_val*100:.2f}%) better than za alone")
        print(f"   📊 IMPROVEMENT (Test): za+zay is {domain_improvement_test:.4f} ({domain_improvement_test*100:.2f}%) better than za alone")
    
    print("\n🏷️  LABEL CLASSIFICATION:")
    print(f"   zy alone:      Train={results['label_zy_alone']['train_acc']:.4f}, Val={results['label_zy_alone']['val_acc']:.4f}, Test={results['label_zy_alone']['test_acc']:.4f}")
    if has_zay:
        print(f"   zy+zay:        Train={results['label_zy_zay']['train_acc']:.4f}, Val={results['label_zy_zay']['val_acc']:.4f}, Test={results['label_zy_zay']['test_acc']:.4f}")
        print(f"   zay alone:     Train={results['label_zay_alone']['train_acc']:.4f}, Val={results['label_zay_alone']['val_acc']:.4f}, Test={results['label_zay_alone']['test_acc']:.4f}")
        
        label_improvement_val = results['label_zy_zay']['val_acc'] - results['label_zy_alone']['val_acc']
        label_improvement_test = results['label_zy_zay']['test_acc'] - results['label_zy_alone']['test_acc']
        print(f"   📊 IMPROVEMENT (Val): zy+zay is {label_improvement_val:.4f} ({label_improvement_val*100:.2f}%) better than zy alone")
        print(f"   📊 IMPROVEMENT (Test): zy+zay is {label_improvement_test:.4f} ({label_improvement_test*100:.2f}%) better than zy alone")
    
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
    
    # Domain classification - Validation
    domain_val_data = df[(df['Task'] == 'Domain Classification') & (df['Split'] == 'Validation')]
    bars1 = axes[0,0].bar(domain_val_data['Method'], domain_val_data['Accuracy'], 
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0,0].set_title('Domain Classification - Validation', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0,0].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Domain classification - Test
    domain_test_data = df[(df['Task'] == 'Domain Classification') & (df['Split'] == 'Test')]
    bars2 = axes[0,1].bar(domain_test_data['Method'], domain_test_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0,1].set_title('Domain Classification - Test', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0,1].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Label classification - Validation
    label_val_data = df[(df['Task'] == 'Label Classification') & (df['Split'] == 'Validation')]
    bars3 = axes[1,0].bar(label_val_data['Method'], label_val_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[1,0].set_title('Label Classification - Validation', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1,0].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Label classification - Test
    label_test_data = df[(df['Task'] == 'Label Classification') & (df['Split'] == 'Test')]
    bars4 = axes[1,1].bar(label_test_data['Method'], label_test_data['Accuracy'],
                         color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[1,1].set_title('Label Classification - Test', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1,1].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Calculate and display improvements
    domain_improvement_val = results['domain_za_zay']['val_acc'] - results['domain_za_alone']['val_acc']
    domain_improvement_test = results['domain_za_zay']['test_acc'] - results['domain_za_alone']['test_acc']
    label_improvement_val = results['label_zy_zay']['val_acc'] - results['label_zy_alone']['val_acc']
    label_improvement_test = results['label_zy_zay']['test_acc'] - results['label_zy_alone']['test_acc']
    
    # Add improvement annotations
    axes[0,0].annotate(f'Val Improvement:\n+{domain_improvement_val:.3f} ({domain_improvement_val*100:.1f}%)',
                      xy=(1, results['domain_za_zay']['val_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')
    
    axes[0,1].annotate(f'Test Improvement:\n+{domain_improvement_test:.3f} ({domain_improvement_test*100:.1f}%)',
                      xy=(1, results['domain_za_zay']['test_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')
    
    axes[1,0].annotate(f'Val Improvement:\n+{label_improvement_val:.3f} ({label_improvement_val*100:.1f}%)',
                      xy=(1, results['label_zy_zay']['val_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')
    
    axes[1,1].annotate(f'Test Improvement:\n+{label_improvement_test:.3f} ({label_improvement_test*100:.1f}%)',
                      xy=(1, results['label_zy_zay']['test_acc']), xytext=(1, 0.9),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
                      fontsize=9, fontweight='bold', color='red', ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'latent_expressiveness_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Expressiveness comparison plot saved to: {plot_path}")

def run_sklearn_comparison(train_data, val_data, save_dir):
    """Run additional comparison using sklearn classifiers for robustness."""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Extract data
    train_zy = train_data['zy'].numpy()
    train_za = train_data['za'].numpy()
    train_zay = train_data['zay'].numpy() if train_data['zay'] is not None else None
    train_y = train_data['y'].numpy().flatten()
    train_a = train_data['a'].numpy().flatten()
    
    val_zy = val_data['zy'].numpy()
    val_za = val_data['za'].numpy()
    val_zay = val_data['zay'].numpy() if val_data['zay'] is not None else None
    val_y = val_data['y'].numpy().flatten()
    val_a = val_data['a'].numpy().flatten()
    
    if train_zay is None:
        print("⚠️  Skipping sklearn comparison - no zay available (DIVA model)")
        return
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42)
    }
    
    sklearn_results = {}
    
    print("\n🔬 Running sklearn classifier comparison...")
    
    for clf_name, clf in classifiers.items():
        print(f"\n   Testing {clf_name}:")
        
        # Domain classification
        clf_za = clf.__class__(**clf.get_params())
        clf_za.fit(train_za, train_a)
        za_acc = clf_za.score(val_za, val_a)
        
        clf_za_zay = clf.__class__(**clf.get_params())
        train_za_zay = np.concatenate([train_za, train_zay], axis=1)
        val_za_zay = np.concatenate([val_za, val_zay], axis=1)
        clf_za_zay.fit(train_za_zay, train_a)
        za_zay_acc = clf_za_zay.score(val_za_zay, val_a)
        
        print(f"     Domain: za={za_acc:.4f}, za+zay={za_zay_acc:.4f} (Δ={za_zay_acc-za_acc:.4f})")
        
        # Label classification
        clf_zy = clf.__class__(**clf.get_params())
        clf_zy.fit(train_zy, train_y)
        zy_acc = clf_zy.score(val_zy, val_y)
        
        clf_zy_zay = clf.__class__(**clf.get_params())
        train_zy_zay = np.concatenate([train_zy, train_zay], axis=1)
        val_zy_zay = np.concatenate([val_zy, val_zay], axis=1)
        clf_zy_zay.fit(train_zy_zay, train_y)
        zy_zay_acc = clf_zy_zay.score(val_zy_zay, val_y)
        
        print(f"     Label:  zy={zy_acc:.4f}, zy+zay={zy_zay_acc:.4f} (Δ={zy_zay_acc-zy_acc:.4f})")
        
        sklearn_results[clf_name] = {
            'domain_za': za_acc,
            'domain_za_zay': za_zay_acc,
            'label_zy': zy_acc,
            'label_zy_zay': zy_zay_acc
        }
    
    # Save sklearn results
    import json
    sklearn_results_path = os.path.join(save_dir, 'sklearn_expressiveness_results.json')
    with open(sklearn_results_path, 'w') as f:
        json.dump(sklearn_results, f, indent=2)
    
    return sklearn_results 
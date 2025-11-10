import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
from tqdm import tqdm

def compute_mutual_information(z1, z2, bins=50):
    """Compute mutual information between two latent variables."""
    # Convert to numpy if tensors
    if torch.is_tensor(z1):
        z1 = z1.detach().cpu().numpy()
    if torch.is_tensor(z2):
        z2 = z2.detach().cpu().numpy()
    
    # Discretize continuous variables
    z1_discrete = np.digitize(z1.flatten(), np.histogram(z1.flatten(), bins=bins)[1])
    z2_discrete = np.digitize(z2.flatten(), np.histogram(z2.flatten(), bins=bins)[1])
    
    return mutual_info_score(z1_discrete, z2_discrete)

def compute_correlation_matrix(latent_dict):
    """Compute correlation matrix between different latent variables."""
    correlations = {}
    
    for name1, z1 in latent_dict.items():
        correlations[name1] = {}
        for name2, z2 in latent_dict.items():
            if torch.is_tensor(z1):
                z1_flat = z1.detach().cpu().numpy().flatten()
            else:
                z1_flat = z1.flatten()
                
            if torch.is_tensor(z2):
                z2_flat = z2.detach().cpu().numpy().flatten()
            else:
                z2_flat = z2.flatten()
            
            corr, _ = pearsonr(z1_flat, z2_flat)
            correlations[name1][name2] = abs(corr)  # Use absolute correlation
    
    return correlations

def analyze_disentanglement(model, dataloader, device, save_path=None):
    """
    Analyze the quality of disentanglement in the trained model.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for analysis
        device: Device to run analysis on
        save_path: Path to save analysis results
    
    Returns:
        dict: Analysis results including correlations, mutual information, etc.
    """
    model.eval()
    
    # Collect latent representations
    all_zy, all_zx, all_zay, all_za = [], [], [], []
    all_y, all_a = [], []
    
    print("🔍 Collecting latent representations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx >= 100:  # Limit analysis to first 100 batches for efficiency
                break
                
            x, y, a = batch[0], batch[1], batch[2]
            if hasattr(batch[2], 'squeeze'):
                a = batch[2].squeeze()
            else:
                a = batch[2]
            
            x, y, a = x.to(device), y.to(device), a.to(device)
            
            # Get latent representations
            qz_loc, qz_scale = model.qz(x)
            
            zy = qz_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
            zx = qz_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
            if not model.diva:
                zay = qz_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
            else:
                zay = torch.zeros_like(zy)
            za = qz_loc[:, model.za_index_range[0]:model.za_index_range[1]]
            
            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            all_zay.append(zay.cpu())
            all_za.append(za.cpu())
            all_y.append(y.cpu())
            all_a.append(a.cpu())
    
    # Concatenate all batches
    all_zy = torch.cat(all_zy, dim=0)
    all_zx = torch.cat(all_zx, dim=0)
    all_zay = torch.cat(all_zay, dim=0)
    all_za = torch.cat(all_za, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_a = torch.cat(all_a, dim=0)
    
    print(f"📊 Analyzing {all_zy.shape[0]} samples...")
    
    # Compute correlations between latent variables
    latent_dict = {
        'zy': all_zy,
        'zx': all_zx, 
        'zay': all_zay,
        'za': all_za
    }
    
    #correlations = compute_correlation_matrix(latent_dict)
    
    # # Compute mutual information
    # print("🧮 Computing mutual information...")
    # mi_results = {}
    # for name1, z1 in latent_dict.items():
    #     mi_results[name1] = {}
    #     for name2, z2 in latent_dict.items():
    #         if name1 != name2:
    #             mi_results[name1][name2] = compute_mutual_information(z1, z2)
    
    # Analyze label-specific information
    print("🏷️  Analyzing label-specific information...")
    
    # Compute how well each latent variable predicts labels
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    label_prediction_scores = {}
    
    # Prepare data
    scaler = StandardScaler()
    
    for name, z in latent_dict.items():
        z_np = z.numpy().reshape(z.shape[0], -1)
        z_scaled = scaler.fit_transform(z_np)
        
        # Predict y labels
        try:
            clf_y = LogisticRegression(max_iter=1000)
            clf_y.fit(z_scaled, all_y.numpy())
            y_pred = clf_y.predict(z_scaled)
            y_acc = accuracy_score(all_y.numpy(), y_pred)
        except:
            y_acc = 0.0
        
        # Predict a labels  
        try:
            clf_a = LogisticRegression(max_iter=1000)
            clf_a.fit(z_scaled, all_a.numpy())
            a_pred = clf_a.predict(z_scaled)
            a_acc = accuracy_score(all_a.numpy(), a_pred)
        except:
            a_acc = 0.0
            
        label_prediction_scores[name] = {
            'y_accuracy': y_acc,
            'a_accuracy': a_acc
        }
    
    # Create analysis results
    results = {
        #'correlations': correlations,
        #'mutual_information': mi_results,
        'label_prediction': label_prediction_scores,
        'summary': {
            #'zy_za_correlation': correlations['zy']['za'],
            #'zy_zay_correlation': correlations['zy']['zay'],
            #'za_zay_correlation': correlations['za']['zay'],
            #'zay_independence_score': 1.0 - (correlations['zy']['zay'] + correlations['za']['zay']) / 2,
            'zy_y_specificity': label_prediction_scores['zy']['y_accuracy'] - label_prediction_scores['zy']['a_accuracy'],
            'za_a_specificity': label_prediction_scores['za']['a_accuracy'] - label_prediction_scores['za']['y_accuracy'],
        }
    }
    
    print("📈 Analysis Results Summary:")
    #print(f"  zy-za correlation: {results['summary']['zy_za_correlation']:.4f}")
    #print(f"  zy-zay correlation: {results['summary']['zy_zay_correlation']:.4f}")
    #print(f"  za-zay correlation: {results['summary']['za_zay_correlation']:.4f}")
    #print(f"  zay independence score: {results['summary']['zay_independence_score']:.4f}")
    print(f"  zy y-specificity: {results['summary']['zy_y_specificity']:.4f}")
    print(f"  za a-specificity: {results['summary']['za_a_specificity']:.4f}")
    
    # Save results and create visualizations
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save numerical results
        import json
        with open(os.path.join(save_path, 'disentanglement_analysis.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                else:
                    return obj
            
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        # corr_matrix = np.array([[correlations[n1][n2] for n2 in ['zy', 'zx', 'zay', 'za']] 
        #                       for n1 in ['zy', 'zx', 'zay', 'za']])
        
        sns.heatmap(corr_matrix, 
                   xticklabels=['zy', 'zx', 'zay', 'za'],
                   yticklabels=['zy', 'zx', 'zay', 'za'],
                   annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.3f')
        plt.title('Latent Variable Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create label prediction accuracy plot
        plt.figure(figsize=(10, 6))
        latent_names = list(label_prediction_scores.keys())
        y_accs = [label_prediction_scores[name]['y_accuracy'] for name in latent_names]
        a_accs = [label_prediction_scores[name]['a_accuracy'] for name in latent_names]
        
        x = np.arange(len(latent_names))
        width = 0.35
        
        plt.bar(x - width/2, y_accs, width, label='Y accuracy', alpha=0.8)
        plt.bar(x + width/2, a_accs, width, label='A accuracy', alpha=0.8)
        
        plt.xlabel('Latent Variables')
        plt.ylabel('Prediction Accuracy')
        plt.title('Label Prediction Accuracy by Latent Variable')
        plt.xticks(x, latent_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'label_prediction_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Analysis results saved to {save_path}")
    
    return results

def analyze_disentanglement_builtin_predictors(model, dataloader, device, save_path=None):
    """
    Analyze disentanglement using the model's built-in predictors instead of training new ones.
    
    This approach tests how well the model's actual predictors work on isolated latent spaces,
    which is more representative of the model's learned representations during training.
    """
    model.eval()
    
    # Collect latent representations and predictions
    all_zy, all_zx, all_zay, all_za = [], [], [], []
    all_y, all_a = [], []
    builtin_y_preds, builtin_a_preds = [], []
    
    print("🔍 Collecting latent representations and built-in predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx >= 100:  # Limit analysis to first 100 batches for efficiency
                break
                
            x, y, a = batch[0], batch[1], batch[2]
            if hasattr(batch[2], 'squeeze'):
                a = batch[2].squeeze()
            else:
                a = batch[2]
            
            x, y, a = x.to(device), y.to(device), a.to(device)
            
            # Get full forward pass with built-in predictions
            if hasattr(model, 'forward'):
                x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = model.forward(a, x, y)
                builtin_y_preds.append(y_hat.cpu())
                builtin_a_preds.append(a_hat.cpu())
            else:
                # Fallback to manual extraction
                qz_loc, qz_scale = model.qz(x)
                zy = qz_loc[:, model.zy_index_range[0]:model.zy_index_range[1]]
                zx = qz_loc[:, model.zx_index_range[0]:model.zx_index_range[1]]
                if not model.diva:
                    zay = qz_loc[:, model.zay_index_range[0]:model.zay_index_range[1]]
                else:
                    zay = torch.zeros_like(zy)
                za = qz_loc[:, model.za_index_range[0]:model.za_index_range[1]]
                
                # Get predictions from model's predictors
                y_hat = model.qy(zy, zay) if hasattr(model, 'qy') else torch.zeros_like(y)
                a_hat = model.qa(za, zay) if hasattr(model, 'qa') else torch.zeros_like(a)
                builtin_y_preds.append(y_hat.cpu())
                builtin_a_preds.append(a_hat.cpu())
            
            all_zy.append(zy.cpu())
            all_zx.append(zx.cpu())
            all_zay.append(zay.cpu())
            all_za.append(za.cpu())
            all_y.append(y.cpu())
            all_a.append(a.cpu())
    
    # Concatenate all batches
    all_zy = torch.cat(all_zy, dim=0)
    all_zx = torch.cat(all_zx, dim=0)
    all_zay = torch.cat(all_zay, dim=0)
    all_za = torch.cat(all_za, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_a = torch.cat(all_a, dim=0)
    builtin_y_preds = torch.cat(builtin_y_preds, dim=0)
    builtin_a_preds = torch.cat(builtin_a_preds, dim=0)
    
    print(f"📊 Analyzing {all_zy.shape[0]} samples with built-in predictors...")
    
    # Test built-in predictors on isolated latent spaces
    from sklearn.metrics import accuracy_score
    import torch.nn.functional as F
    
    builtin_prediction_scores = {}
    
    # Test predictions on individual latent spaces by zeroing others
    latent_combinations = {
        'zy_only': (all_zy, torch.zeros_like(all_zx), torch.zeros_like(all_zay), torch.zeros_like(all_za)),
        'zx_only': (torch.zeros_like(all_zy), all_zx, torch.zeros_like(all_zay), torch.zeros_like(all_za)),
        'zay_only': (torch.zeros_like(all_zy), torch.zeros_like(all_zx), all_zay, torch.zeros_like(all_za)),
        'za_only': (torch.zeros_like(all_zy), torch.zeros_like(all_zx), torch.zeros_like(all_zay), all_za),
        'full': (all_zy, all_zx, all_zay, all_za)
    }
    
    for combo_name, (zy, zx, zay, za) in latent_combinations.items():
        with torch.no_grad():
            zy, zx, zay, za = zy.to(device), zx.to(device), zay.to(device), za.to(device)
            
            # Get predictions using model's built-in predictors
            if hasattr(model, 'qy') and hasattr(model, 'qa'):
                y_pred_logits = model.qy(zy, zay)
                a_pred_logits = model.qa(za, zay)
                
                # Convert to predictions (predictors now return logits)
                y_pred = torch.argmax(y_pred_logits, dim=1).cpu()
                a_pred = torch.argmax(a_pred_logits, dim=1).cpu()
                
                # Calculate accuracies
                y_acc = accuracy_score(all_y.numpy(), y_pred.numpy())
                a_acc = accuracy_score(all_a.numpy(), a_pred.numpy())
            else:
                y_acc, a_acc = 0.0, 0.0
            
            builtin_prediction_scores[combo_name] = {
                'y_accuracy': y_acc,
                'a_accuracy': a_acc
            }
    
    # Also get standard correlation analysis
    latent_dict = {
        'zy': all_zy,
        'zx': all_zx, 
        'zay': all_zay,
        'za': all_za
    }
    # correlations = compute_correlation_matrix(latent_dict)
    
    # Create comprehensive results
    results = {
        # 'correlations': correlations,
        'builtin_predictions': builtin_prediction_scores,
        'summary': {
            # 'zy_za_correlation': correlations['zy']['za'],
            # 'zy_zay_correlation': correlations['zy']['zay'],
            # 'za_zay_correlation': correlations['za']['zay'],
            'builtin_zy_y_specificity': builtin_prediction_scores['zy_only']['y_accuracy'],
            'builtin_za_a_specificity': builtin_prediction_scores['za_only']['a_accuracy'],
            'builtin_full_y_accuracy': builtin_prediction_scores['full']['y_accuracy'],
            'builtin_full_a_accuracy': builtin_prediction_scores['full']['a_accuracy'],
        }
    }
    
    print("📈 Built-in Predictor Analysis Results:")
    print(f"  Full model Y accuracy: {results['summary']['builtin_full_y_accuracy']:.4f}")
    print(f"  Full model A accuracy: {results['summary']['builtin_full_a_accuracy']:.4f}")
    print(f"  zy-only Y accuracy: {results['summary']['builtin_zy_y_specificity']:.4f}")
    print(f"  za-only A accuracy: {results['summary']['builtin_za_a_specificity']:.4f}")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save results
        import json
        with open(os.path.join(save_path, 'builtin_predictor_analysis.json'), 'w') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                else:
                    return obj
            
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        combo_names = list(builtin_prediction_scores.keys())
        y_accs = [builtin_prediction_scores[name]['y_accuracy'] for name in combo_names]
        a_accs = [builtin_prediction_scores[name]['a_accuracy'] for name in combo_names]
        
        x = np.arange(len(combo_names))
        width = 0.35
        
        plt.bar(x - width/2, y_accs, width, label='Y accuracy (built-in predictor)', alpha=0.8)
        plt.bar(x + width/2, a_accs, width, label='A accuracy (built-in predictor)', alpha=0.8)
        
        plt.xlabel('Latent Combination')
        plt.ylabel('Prediction Accuracy')
        plt.title('Built-in Predictor Accuracy on Different Latent Combinations')
        plt.xticks(x, combo_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'builtin_predictor_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Built-in predictor analysis saved to {save_path}")
    
    return results

def compare_disentanglement(standard_model, staged_model, dataloader, device, save_path=None):
    """Compare disentanglement quality between standard and staged training."""
    print("🔬 Comparing disentanglement quality...")
    
    # Analyze both models
    print("\n📊 Analyzing standard model...")
    standard_results = analyze_disentanglement(standard_model, dataloader, device)
    
    print("\n📊 Analyzing staged model...")
    staged_results = analyze_disentanglement(staged_model, dataloader, device)
    
    # Compare results
    comparison = {
        'standard': standard_results['summary'],
        'staged': staged_results['summary'],
        'improvements': {}
    }
    
    for metric in standard_results['summary']:
        improvement = staged_results['summary'][metric] - standard_results['summary'][metric]
        comparison['improvements'][metric] = improvement
    
    print("\n📈 Comparison Results:")
    print("Metric | Standard | Staged | Improvement")
    print("-" * 50)
    for metric in comparison['standard']:
        std_val = comparison['standard'][metric]
        staged_val = comparison['staged'][metric]
        imp_val = comparison['improvements'][metric]
        print(f"{metric:20s} | {std_val:8.4f} | {staged_val:6.4f} | {imp_val:+7.4f}")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save comparison results
        import json
        with open(os.path.join(save_path, 'disentanglement_comparison.json'), 'w') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                else:
                    return obj
            
            serializable_comparison = convert_to_serializable(comparison)
            json.dump(serializable_comparison, f, indent=2)
        
        print(f"💾 Comparison results saved to {save_path}")
    
    return comparison 
from core.train import train
from core.test import test_nvae, test_dann
from core.CRMNIST.model import VAE
from core.CRMNIST.comparison.dann import DANN
from core.CRMNIST.comparison.irm import IRM
import torch
from core.CRMNIST.data_generation import generate_crmnist_dataset
from core.CRMNIST.comparison.train import train_nvae, train_diva, train_dann, train_irm
import core.CRMNIST.utils
import core.utils
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
from scipy import stats
import datetime

"""
    Training and evaluation of NVAE, DIVA, DANN, and IRM models

    To run the script, use the following command:

    python -m core.CRMNIST.comparison.evaluation --config conf/crmnist.json --out results/ --cuda
"""

device = None 

def test_irm(model, test_loader, device):
    """
    Test function for IRM model
    
    Args:
        model: IRM model to test
        test_loader: DataLoader for test data
        device: Device to run inference on
        
    Returns:
        tuple: (test_loss, metrics_dict)
    """
    if model is None:
        raise ValueError("Model cannot be None")
    if test_loader is None:
        raise ValueError("Test loader cannot be None")
        
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (x, y, c, r) in enumerate(test_loader):
            try:
                x, y, r = x.to(device), y.to(device), r.to(device)
                
                logits, _ = model.forward(x, y, r)
                loss = torch.nn.functional.cross_entropy(logits, y)
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                test_loss += loss.item()
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if len(test_loader) == 0:
        print("Warning: Empty test loader")
        return 0.0, {'y_accuracy': 0.0, 'classification_accuracy': 0.0}
    
    test_loss /= len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'y_accuracy': accuracy,
        'classification_accuracy': accuracy  # For consistency with other models
    }
    
    return test_loss, metrics

def calculate_confidence_interval(accuracies, confidence=0.95):
    """
    Calculate confidence interval for a list of accuracies using t-distribution
    """
    n = len(accuracies)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)  # Sample standard deviation
    
    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * (std / np.sqrt(n))
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, std, ci_lower, ci_upper

def evaluate_models_with_seeds(nvae, diva, dann, irm, test_loader, test_domain=None, path=None):
    """
    Evaluate models on the test set, optionally for a specific domain.
    
    Args:
        nvae: NVAE model
        diva: DIVA model
        dann: DANN model
        irm: IRM model
        test_loader: DataLoader for test set
        test_domain: Optional domain to evaluate on (0-5 for rotations)
    """
    print("Evaluating models" + (f" on domain {test_domain}" if test_domain is not None else ""))
    
    # Filter test loader for specific domain if requested
    if test_domain is not None:
        domain_test_loader = filter_domain_loader(test_loader, test_domain)
    else:
        domain_test_loader = test_loader
    
    test_loss_nvae, test_metrics_nvae = test_nvae(nvae, domain_test_loader, device)
    test_loss_diva, test_metrics_diva = test_nvae(diva, domain_test_loader, device)  # DIVA uses same test function as NVAE (both are VAE architectures)
    test_loss_dann, test_metrics_dann = test_dann(dann, domain_test_loader, device)
    test_loss_irm, test_metrics_irm = test_irm(irm, domain_test_loader, device)

    print("--------------------------------")
    print(f"NVAE test results:")
    print(f"Test loss: {test_loss_nvae}")
    print(f"Test metrics: {test_metrics_nvae}")
    
    print("--------------------------------")
    print(f"DIVA test results:")
    print(f"Test loss: {test_loss_diva}")
    print(f"Test metrics: {test_metrics_diva}")
    
    print("--------------------------------")
    print(f"DANN test results:")
    print(f"Test loss: {test_loss_dann}")
    print(f"Test metrics: {test_metrics_dann}")
    
    print("--------------------------------")
    print(f"IRM test results:")
    print(f"Test loss: {test_loss_irm}")
    print(f"Test metrics: {test_metrics_irm}")
   
    return {
        'nvae': {'loss': test_loss_nvae, 'metrics': test_metrics_nvae},
        'diva': {'loss': test_loss_diva, 'metrics': test_metrics_diva},
        'dann': {'loss': test_loss_dann, 'metrics': test_metrics_dann},
        'irm': {'loss': test_loss_irm, 'metrics': test_metrics_irm}
    }

def filter_domain_loader(loader, domain, exclude=False):
    """
    Filter a DataLoader to include or exclude samples from a specific domain.
    
    Args:
        loader: Original DataLoader
        domain: Domain to filter for (0-5 for rotations)
        exclude: If True, exclude samples from the specified domain instead of including them
        
    Returns:
        Filtered DataLoader with only samples from the specified domain (or all except that domain)
        
    Raises:
        ValueError: If no samples found for the specified domain or if domain is invalid
    """
    if domain < 0:
        raise ValueError(f"Domain must be non-negative, got {domain}")
    
    domain_samples = []
    total_samples = 0
    
    for batch in loader:
        x, y, c, r = batch
        total_samples += len(x)
        
        # Get rotation domain for each sample
        if len(r.shape) > 1:
            rotation_indices = torch.argmax(r, dim=1)
        else:
            rotation_indices = r
            
        # Validate domain values
        max_domain = torch.max(rotation_indices).item()
        if domain > max_domain:
            continue  # Skip this batch if domain is out of range
            
        # Keep samples based on exclude flag
        mask = rotation_indices != domain if exclude else rotation_indices == domain
        if mask.any():
            domain_samples.append((x[mask], y[mask], c[mask], r[mask]))
    
    # Combine all domain samples
    if domain_samples:
        x_domain = torch.cat([s[0] for s in domain_samples])
        y_domain = torch.cat([s[1] for s in domain_samples])
        c_domain = torch.cat([s[2] for s in domain_samples])
        r_domain = torch.cat([s[3] for s in domain_samples])
        
        print(f"Filtered {'excluding' if exclude else 'including'} domain {domain}: "
              f"{len(x_domain)} samples out of {total_samples} total")
        
        # Create a new dataset with only domain samples
        domain_dataset = torch.utils.data.TensorDataset(x_domain, y_domain, c_domain, r_domain)
        return torch.utils.data.DataLoader(
            domain_dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory
        )
    else:
        action = "excluding" if exclude else "including"
        raise ValueError(f"No samples found when {action} domain {domain}. "
                        f"Total samples processed: {total_samples}")
    
def print_results(results, holdout=False):
    """
    Print results for each domain and model

    results is a dictionary with the following structure:
    {
        'model_name1': {
            'domain_0': {'loss': 0.1, 'metrics': {'metric_1': 0.2, 'metric_2': 0.3}},
            'domain_1': {'loss': 0.1, 'metrics': {'metric_1': 0.2, 'metric_2': 0.3}},
            ...
        },
        'model_name2': {
            'domain_0': {'loss': 0.1, 'metrics': {'metric_1': 0.2, 'metric_2': 0.3}},
            'domain_1': {'loss': 0.1, 'metrics': {'metric_1': 0.2, 'metric_2': 0.3}},
            ...
        },
        ...
    }

    holdout: True if holdout results, False if cross-domain results
    """

    print(f"--------------------------------")
    print(f"{'Holdout' if holdout else 'Cross-domain'} results:")

    for model in results:
        print(f"--------------------------------")
        print(f"Domain results for {model.upper()}:")
        for domain in results[model]:
            print(f"\n{'Held out test' if holdout else 'Cross-domain test'} domain {domain} ({domain * 15}°):")
            print(f"Loss: {results[model][domain]['loss']:.4f}")
            for metric in results[model][domain]['metrics']:
                print(f"{metric}: {results[model][domain]['metrics'][metric]:.4f}")
        print(f"--------------------------------")
    print(f"--------------------------------")



def run_cross_domain_evaluation(args, nvae, diva, dann, irm, test_loader, spec_data):
    """
    UNUSED FUNCTION - kept for potential future use.
    
    Run evaluation across all domains and compute statistics.
    
    Args:
        args: Command line arguments
        nvae: NVAE model
        diva: DIVA model
        dann: DANN model
        irm: IRM model
        test_loader: DataLoader for test set
        
    Returns:
        Dictionary containing mean and std of metrics across domains
    """
    num_domains = spec_data['num_r_classes'] 
    cross_domain_results = {'nvae': {}, 'diva': {}, 'dann': {}, 'irm': {}}
    
    # Evaluate on each domain
    for domain in range(num_domains):
        print(f"\nEvaluating on domain {domain} ({domain * 15}°)")
        results = evaluate_models_with_seeds(nvae, diva, dann, irm, test_loader, test_domain=domain, path=os.path.join(args.out, f'cross_domain_domain_{domain}'))
        for model in cross_domain_results:
            cross_domain_results[model][domain] = {
                'loss': results[model]['loss'],
                'metrics': results[model]['metrics']
            }
    
    return cross_domain_results

def run_holdout_evaluation_with_seeds(args, train_loader, test_loader, class_map, spec_data, num_seeds=10):
    """
    Run evaluation where we hold out the final domain (domain 5, 75°) for testing.
    Train each model 10 times with different seeds and compute confidence intervals.
    
    Args:
        args: Command line arguments
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        class_map: Class mapping
        spec_data: Specification data
        num_seeds: Number of different random seeds to use
        
    Returns:
        Dictionary containing results with confidence intervals for the final holdout domain
    """
    num_domains = spec_data['num_r_classes']  # Use the correct number from config (6 domains)
    holdout_results = {'nvae': {}, 'diva': {}, 'dann': {}, 'irm': {}}
    
    # Hold out only the final domain (domain 5, which is 75°)
    holdout_domain = num_domains - 1  # Domain 5 (75°)
    print(f"\nHolding out final domain {holdout_domain} ({holdout_domain * 15}°)")
    
    # Filter training data to exclude holdout domain (train on domains 0-4)
    filtered_train_loader = filter_domain_loader(train_loader, holdout_domain, exclude=True)
    holdout_test_loader = filter_domain_loader(test_loader, holdout_domain)
    
    # Store accuracies for each model across seeds
    model_accuracies = {'nvae': [], 'diva': [], 'dann': [], 'irm': []}
    
    # Train each model with different seeds (10 models per type = 40 total models)
    for seed in range(num_seeds):
        print(f"\n  Training with seed {seed+1}/{num_seeds}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Train NVAE
        nvae, _ = train_nvae(args, spec_data, filtered_train_loader, holdout_test_loader, None)
        _, nvae_metrics = test_nvae(nvae, holdout_test_loader, args.device)
        model_accuracies['nvae'].append(nvae_metrics.get('y_accuracy', 0.0))
        del nvae  # Free memory
        
        # Train DIVA
        diva, _ = train_diva(args, spec_data, filtered_train_loader, holdout_test_loader, None)
        _, diva_metrics = test_nvae(diva, holdout_test_loader, args.device)  # DIVA uses same test function as NVAE
        model_accuracies['diva'].append(diva_metrics.get('y_accuracy', 0.0))
        del diva  # Free memory
        
        # Train DANN
        dann, _ = train_dann(args, spec_data, filtered_train_loader, holdout_test_loader, None)
        _, dann_metrics = test_dann(dann, holdout_test_loader, args.device)
        model_accuracies['dann'].append(dann_metrics.get('y_accuracy', 0.0))
        del dann  # Free memory
        
        # Train IRM
        irm, _ = train_irm(args, spec_data, filtered_train_loader, holdout_test_loader, None, seed=seed)
        _, irm_metrics = test_irm(irm, holdout_test_loader, args.device)
        model_accuracies['irm'].append(irm_metrics.get('y_accuracy', 0.0))
        del irm  # Free memory
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate confidence intervals for each model
    for model in model_accuracies:
        accuracies = model_accuracies[model]
        mean, std, ci_lower, ci_upper = calculate_confidence_interval(accuracies)
        
        margin_error = (ci_upper - ci_lower) / 2
        
        holdout_results[model][holdout_domain] = {
            'accuracies': accuracies,
            'mean_accuracy': mean,
            'std_accuracy': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_interval': f"{mean:.4f} ± {margin_error:.4f}",
            'metrics': {'y_accuracy': mean}  # For compatibility
        }
    
    return holdout_results

def run_cross_domain_evaluation_with_seeds(args, train_loader, test_loader, spec_data, num_seeds=10):
    """
    Run cross-domain evaluation with multiple seeds and confidence intervals.
    Train models once with multiple seeds, then test on each domain.
    
    Args:
        args: Command line arguments
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        spec_data: Specification data
        num_seeds: Number of different random seeds to use
        
    Returns:
        Dictionary containing results with confidence intervals for each domain
    """
    num_domains = spec_data['num_r_classes']
    cross_domain_results = {'nvae': {}, 'diva': {}, 'dann': {}, 'irm': {}}
    
    # Train models with different seeds
    trained_models = {'nvae': [], 'diva': [], 'dann': [], 'irm': []}
    
    print("Training models with different seeds...")
    for seed in range(num_seeds):
        print(f"\nTraining with seed {seed+1}/{num_seeds}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Train NVAE
        nvae, _ = train_nvae(args, spec_data, train_loader, test_loader, None)
        trained_models['nvae'].append(nvae)
        
        # Train DIVA
        diva, _ = train_diva(args, spec_data, train_loader, test_loader, None)
        trained_models['diva'].append(diva)
        
        # Train DANN
        dann, _ = train_dann(args, spec_data, train_loader, test_loader, None)
        trained_models['dann'].append(dann)
        
        # Train IRM
        irm, _ = train_irm(args, spec_data, train_loader, test_loader, None, seed=seed)
        trained_models['irm'].append(irm)
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Evaluate on each domain
    for domain in range(num_domains):
        print(f"\nEvaluating on domain {domain} ({domain * 15}°)")
        domain_test_loader = filter_domain_loader(test_loader, domain)
        
        # Store accuracies for each model across seeds
        model_accuracies = {'nvae': [], 'diva': [], 'dann': [], 'irm': []}
        
        # Test each trained model on this domain
        for seed in range(num_seeds):
            # Test NVAE
            _, nvae_metrics = test_nvae(trained_models['nvae'][seed], domain_test_loader, args.device)
            model_accuracies['nvae'].append(nvae_metrics.get('y_accuracy', 0.0))
            
            # Test DIVA
            _, diva_metrics = test_nvae(trained_models['diva'][seed], domain_test_loader, args.device)  # DIVA uses same test function as NVAE
            model_accuracies['diva'].append(diva_metrics.get('y_accuracy', 0.0))
            
            # Test DANN
            _, dann_metrics = test_dann(trained_models['dann'][seed], domain_test_loader, args.device)
            model_accuracies['dann'].append(dann_metrics.get('y_accuracy', 0.0))
            
            # Test IRM
            _, irm_metrics = test_irm(trained_models['irm'][seed], domain_test_loader, args.device)
            model_accuracies['irm'].append(irm_metrics.get('y_accuracy', 0.0))
        
        # Calculate confidence intervals for each model
        for model in model_accuracies:
            accuracies = model_accuracies[model]
            mean, std, ci_lower, ci_upper = calculate_confidence_interval(accuracies)
            
            margin_error = (ci_upper - ci_lower) / 2
            
            cross_domain_results[model][domain] = {
                'accuracies': accuracies,
                'mean_accuracy': mean,
                'std_accuracy': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence_interval': f"{mean:.4f} ± {margin_error:.4f}",
                'metrics': {'y_accuracy': mean}  # For compatibility
            }
    
    return cross_domain_results

def print_results_with_ci(results, holdout=False):
    """
    Print results for each domain and model with confidence intervals

    results is a dictionary with the following structure:
    {
        'model_name1': {
            'domain_0': {
                'mean_accuracy': 0.85,
                'std_accuracy': 0.02,
                'ci_lower': 0.83,
                'ci_upper': 0.87,
                'confidence_interval': '0.8500 ± 0.0123',
                'accuracies': [0.84, 0.86, ...]
            },
            ...
        },
        ...
    }

    holdout: True if holdout results, False if cross-domain results
    """

    print(f"--------------------------------")
    print(f"{'Holdout' if holdout else 'Cross-domain'} results with 95% Confidence Intervals:")

    for model in results:
        print(f"--------------------------------")
        print(f"Domain results for {model.upper()}:")
        for domain in results[model]:
            result = results[model][domain]
            print(f"\n{'Held out test' if holdout else 'Cross-domain test'} domain {domain} ({domain * 15}°):")
            print(f"Mean Accuracy: {result['mean_accuracy']:.4f}")
            print(f"Std Accuracy: {result['std_accuracy']:.4f}")
            print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            print(f"Confidence Interval: {result['confidence_interval']}")
            print(f"Individual accuracies: {[f'{acc:.4f}' for acc in result['accuracies']]}")
        print(f"--------------------------------")
    print(f"--------------------------------")

def analyze_domain_distribution(train_loader, test_loader, num_domains=6):
    """
    Analyze and print the distribution of images across domains in both training and test sets.
    
    Args:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        num_domains: Number of domains (default 6 for CRMNIST rotations)
    """
    # Initialize counters for each domain
    train_domain_counts = {i: 0 for i in range(num_domains)}
    test_domain_counts = {i: 0 for i in range(num_domains)}
    
    print("\nAnalyzing domain distribution...")
    
    # Count training set distribution
    print("\nCounting training set distribution...")
    for x, y, c, r in train_loader:
        # Convert one-hot encoded labels to indices if needed
        if len(r.shape) > 1:
            domain_indices = torch.argmax(r, dim=1)
        else:
            domain_indices = r
            
        for domain_idx in domain_indices:
            domain_val = domain_idx.item()
            if domain_val < num_domains:
                train_domain_counts[domain_val] += 1
    
    # Count test set distribution
    print("\nCounting test set distribution...")
    for x, y, c, r in test_loader:
        # Convert one-hot encoded labels to indices if needed
        if len(r.shape) > 1:
            domain_indices = torch.argmax(r, dim=1)
        else:
            domain_indices = r
            
        for domain_idx in domain_indices:
            domain_val = domain_idx.item()
            if domain_val < num_domains:
                test_domain_counts[domain_val] += 1
    
    # Print results
    print("\nDomain Distribution:")
    print("Domain | Training Count | Training % | Test Count | Test %")
    print("-" * 60)
    
    total_train = sum(train_domain_counts.values())
    total_test = sum(test_domain_counts.values())
    
    for domain in range(num_domains):
        train_count = train_domain_counts[domain]
        test_count = test_domain_counts[domain]
        train_percent = (train_count / total_train) * 100 if total_train > 0 else 0.0
        test_percent = (test_count / total_test) * 100 if total_test > 0 else 0.0
        print(f"{domain:6d} | {train_count:13d} | {train_percent:9.2f}% | {test_count:10d} | {test_percent:6.2f}%")
    
    return train_domain_counts, test_domain_counts

def generate_param_based_dirname(args):
    """
    Generate a directory name based on key model parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Directory name based on parameters
    """
    # Create a descriptive directory name based on key parameters
    param_parts = [
        f"alpha1-{args.alpha_1}",
        f"alpha2-{args.alpha_2}",
        f"zy{args.zy_dim}",
        f"zx{args.zx_dim}",
        f"zay{args.zay_dim}",
        f"za{args.za_dim}",
        f"b1-{args.beta_1}",
        f"b2-{args.beta_2}",
        f"b3-{args.beta_3}",
        f"b4-{args.beta_4}",
        f"ep{args.epochs}",
        f"bs{args.batch_size}",
        f"lr{args.learning_rate}"
    ]
    
    # Join with underscores and limit length
    param_dirname = "_".join(param_parts)
    
    # Add timestamp to ensure uniqueness if parameters are identical
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    param_dirname = f"{param_dirname}_{timestamp}"
    
    return param_dirname

def save_model_args(args, output_dir):
    """
    Save model arguments to a text file in the output directory.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save the arguments file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model-related arguments to save
    model_args = {
        'Model Architecture Parameters': {
            'zy_dim': args.zy_dim,
            'zx_dim': args.zx_dim,
            'zay_dim': args.zay_dim,
            'za_dim': args.za_dim,
        },
        'Loss Weight Parameters': {
            'beta_1': args.beta_1,
            'beta_2': args.beta_2,
            'beta_3': args.beta_3,
            'beta_4': args.beta_4,
            'alpha_1': args.alpha_1,
            'alpha_2': args.alpha_2,
        },
        'Training Parameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
        },
        'Data Parameters': {
            'intensity': args.intensity,
            'intensity_decay': args.intensity_decay,
        },
        'Experiment Settings': {
            'setting': args.setting,
            'mode': args.mode,
            'config': args.config,
            'device': str(args.device),
            'cuda': args.cuda,
            'num_workers': args.num_workers,
        }
    }
    
    # Save to text file
    args_file_path = os.path.join(output_dir, 'model_arguments.txt')
    
    with open(args_file_path, 'w') as f:
        f.write("CRMNIST Evaluation - Model Arguments\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for category, params in model_args.items():
            f.write(f"{category}:\n")
            f.write("-" * len(category) + "\n")
            for param_name, param_value in params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write("\n")
    
    print(f"Model arguments saved to: {args_file_path}")
    return args_file_path

def prep_domain_data(spec_data):
    # Prepare domain data
    domain_data = {int(key): value for key, value in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data
    class_map = spec_data['class_map']
    
    # Choose labels subset if not already chosen
    y_c, subsets = core.CRMNIST.utils.choose_label_subset(spec_data)
    spec_data['y_c'] = y_c

    return spec_data, class_map

def run_experiment(args):
    global device
    device = args.device
    print(f"Using device: {device}")
    
    # Validate arguments
    if not hasattr(args, 'mode') or args.mode is None:
        raise ValueError("Mode must be specified")
    if not hasattr(args, 'setting') or args.setting is None:
        raise ValueError("Setting must be specified")
    if not hasattr(args, 'config') or not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not hasattr(args, 'out') or args.out is None:
        raise ValueError("Output directory must be specified")
    
    # Generate parameter-based subdirectory
    param_dirname = generate_param_based_dirname(args)
    param_output_dir = os.path.join(args.out, param_dirname)
    
    print(f"Parameter-based output directory: {param_output_dir}")
    
    # Save model arguments to text file in the parameter-specific directory
    save_model_args(args, param_output_dir)
    
    # Update args.out to use the parameter-specific directory for all subsequent operations
    original_out = args.out
    args.out = param_output_dir
    
    mode = args.mode
    print(f"Running in {mode} mode")
    setting = args.setting
    print(f"Running in {setting} setting")
    
    # Load configuration from JSON
    try:
        with open(args.config, 'r') as file:
            spec_data = json.load(file)
    except Exception as e:
        raise ValueError(f"Error loading config file {args.config}: {e}")
    
    # Validate spec_data
    required_keys = ['num_r_classes', 'domain_data', 'class_map']
    for key in required_keys:
        if key not in spec_data:
            raise ValueError(f"Missing required key '{key}' in config file")
    
    models_dir = os.path.join(args.out, 'comparison_models')
    os.makedirs(models_dir, exist_ok=True)

    if setting == 'cross-domain':
        # results/comparison_models/cross-domain
        models_dir = os.path.join(models_dir, 'cross-domain')
        os.makedirs(models_dir, exist_ok=True)
    else:
        # results/comparison_models/holdout
        models_dir = os.path.join(models_dir, 'holdout')
        os.makedirs(models_dir, exist_ok=True)
    spec_data, class_map = prep_domain_data(spec_data)


    train_dataset = generate_crmnist_dataset(spec_data, train=True, 
                           transform_intensity=args.intensity,
                           transform_decay=args.intensity_decay)
    
    test_dataset = generate_crmnist_dataset(spec_data, train=False,
                          transform_intensity=args.intensity,
                          transform_decay=args.intensity_decay)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    # Analyze domain distribution
    train_domain_counts, test_domain_counts = analyze_domain_distribution(
        train_loader, test_loader, spec_data['num_r_classes']
    )
    
    if setting == 'cross-domain' and mode == 'train':
        
        nvae, training_results_nvae = train_nvae(args, spec_data, train_loader, test_loader, models_dir)
        diva, training_results_diva = train_diva(args, spec_data, train_loader, test_loader, models_dir)
        dann, training_results_dann = train_dann(args, spec_data, train_loader, test_loader, models_dir)
        irm, training_results_irm = train_irm(args, spec_data, train_loader, test_loader, models_dir)

        print("--------------------------------")
        print("NVAE training results:")
        print(f"Best model epoch: {training_results_nvae['best_model_epoch']}\nBest validation loss: {training_results_nvae['best_validation_loss']}\nBest batch metrics: {training_results_nvae['best_batch_metrics']}")

        print("--------------------------------")

        print("DIVA training results:")
        print(f"Best model epoch: {training_results_diva['best_model_epoch']}\nBest validation loss: {training_results_diva['best_validation_loss']}\nBest batch metrics: {training_results_diva['best_batch_metrics']}")

        print("--------------------------------")

        print("DANN training results:")
        print(f"Best model epoch: {training_results_dann['best_model_epoch']}\nBest validation loss: {training_results_dann['best_validation_loss']}\nBest batch metrics: {training_results_dann['best_batch_metrics']}")

        print("--------------------------------")

        print("IRM training results:")
        print(f"Best model epoch: {training_results_irm['best_model_epoch']}\nBest validation loss: {training_results_irm['best_validation_loss']}\nBest batch metrics: {training_results_irm['best_batch_metrics']}")

        print("--------------------------------")
        print("Finished training models")

        return
    
    if setting == 'holdout' and (mode == 'train' or mode == 'test'):
        # Run holdout evaluation
        holdout_results = run_holdout_evaluation_with_seeds(args, train_loader, test_loader, class_map, spec_data)
        print_results_with_ci(holdout_results, holdout=True)

        results_path = os.path.join(args.out, 'evaluation_results_holdout.json')
        with open(results_path, 'w') as f:
            json.dump(holdout_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")

        return
    
    # Load models with error handling
    checkpoint_files = {
        'nvae': os.path.join(models_dir, 'nvae_checkpoint.pt'),
        'diva': os.path.join(models_dir, 'diva_checkpoint.pt'),
        'dann': os.path.join(models_dir, 'dann_checkpoint.pt'),
        'irm': os.path.join(models_dir, 'irm_checkpoint.pt')
    }
    
    # Check if all checkpoint files exist
    for model_name, checkpoint_path in checkpoint_files.items():
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Please train the {model_name.upper()} model first.")
    
    nvae_checkpoint = torch.load(checkpoint_files['nvae'], map_location=device)
    diva_checkpoint = torch.load(checkpoint_files['diva'], map_location=device)
    dann_checkpoint = torch.load(checkpoint_files['dann'], map_location=device)
    irm_checkpoint = torch.load(checkpoint_files['irm'], map_location=device)

    # Create models with saved parameters
    nvae = VAE(**nvae_checkpoint['params']).to(device)
    diva = VAE(**diva_checkpoint['params']).to(device)
    dann = DANN(**dann_checkpoint['params']).to(device)
    irm = IRM(**irm_checkpoint['params']).to(device)

    # Load state dictionaries
    nvae.load_state_dict(nvae_checkpoint['state_dict'])
    diva.load_state_dict(diva_checkpoint['state_dict'])
    dann.load_state_dict(dann_checkpoint['state_dict'])
    irm.load_state_dict(irm_checkpoint['state_dict'])

    # Set models to eval mode
    nvae.eval()
    diva.eval()
    dann.eval()
    irm.eval()

    if setting == 'cross-domain' and mode == 'test':
        # Run cross-domain evaluation using pre-loaded models
        cross_domain_results = {'nvae': {}, 'diva': {}, 'dann': {}, 'irm': {}}
        num_domains = spec_data['num_r_classes']
        
        # Evaluate on each domain
        for domain in range(num_domains):
            print(f"\nEvaluating on domain {domain} ({domain * 15}°)")
            results = evaluate_models_with_seeds(nvae, diva, dann, irm, test_loader, test_domain=domain)
            for model in cross_domain_results:
                cross_domain_results[model][domain] = {
                    'loss': results[model]['loss'],
                    'metrics': results[model]['metrics']
                }
        
        # Print results
        print_results(cross_domain_results, holdout=False)

        results_path = os.path.join(args.out, 'evaluation_results_cross_domain.json')
        with open(results_path, 'w') as f:
            json.dump(cross_domain_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        return
    
    if mode == 'visualize':

        # visualize latent spaces

        nvae.visualize_latent_spaces(test_loader, device, os.path.join(args.out, 'nvae_latent_space'))
        # nvae.visualize_disentanglement(test_loader, device, os.path.join(args.out, 'nvae_disentanglement'))
        # nvae.visualize_latent_correlations(test_loader, device, os.path.join(args.out, 'nvae_latent_correlations'))

        diva.visualize_latent_spaces(test_loader, device, os.path.join(args.out, 'diva_latent_space'))
        # diva.visualize_disentanglement(test_loader, device, os.path.join(args.out, 'diva_disentanglement'))
        # diva.visualize_latent_correlations(test_loader, device, os.path.join(args.out, 'diva_latent_correlations'))

        dann.visualize_latent_space(test_loader, device, os.path.join(args.out, 'dann_latent_space'))
        
        irm.visualize_latent_space(test_loader, device, os.path.join(args.out, 'irm_latent_space'))

        # visualize reconstruction

        # nvae.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'nvae_reconstruction'))
        # diva.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'diva_reconstruction'))
        # dann.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'dann_reconstruction'))
        # irm.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'irm_reconstruction'))
    
    if mode == 'confidence_interval' or mode == 'all':
        print("\n" + "="*80)
        print("RUNNING CONFIDENCE INTERVAL EVALUATION WITH 10 SEEDS")
        print("="*80)
        
        if setting == 'holdout':
            print("Running holdout evaluation with confidence intervals...")
            holdout_results = run_holdout_evaluation_with_seeds(args, train_loader, test_loader, class_map, spec_data)
            print_results_with_ci(holdout_results, holdout=True)
            
            # Save results
            results_path = os.path.join(args.out, 'evaluation_results_holdout_ci.json')
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model in holdout_results:
                json_results[model] = {}
                for domain in holdout_results[model]:
                    result = holdout_results[model][domain].copy()
                    result['accuracies'] = [float(acc) for acc in result['accuracies']]
                    result['mean_accuracy'] = float(result['mean_accuracy'])
                    result['std_accuracy'] = float(result['std_accuracy'])
                    result['ci_lower'] = float(result['ci_lower'])
                    result['ci_upper'] = float(result['ci_upper'])
                    json_results[model][domain] = result
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nHoldout CI results saved to {results_path}")
            
        elif setting == 'cross-domain':
            print("Running cross-domain evaluation with confidence intervals...")
            cross_domain_results = run_cross_domain_evaluation_with_seeds(args, train_loader, test_loader, spec_data)
            print_results_with_ci(cross_domain_results, holdout=False)
            
            # Save results
            results_path = os.path.join(args.out, 'evaluation_results_cross_domain_ci.json')
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model in cross_domain_results:
                json_results[model] = {}
                for domain in cross_domain_results[model]:
                    result = cross_domain_results[model][domain].copy()
                    result['accuracies'] = [float(acc) for acc in result['accuracies']]
                    result['mean_accuracy'] = float(result['mean_accuracy'])
                    result['std_accuracy'] = float(result['std_accuracy'])
                    result['ci_lower'] = float(result['ci_lower'])
                    result['ci_upper'] = float(result['ci_upper'])
                    json_results[model][domain] = result
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nCross-domain CI results saved to {results_path}")

if __name__ == "__main__":
    parser = core.utils.get_parser('CRMNIST')
    parser.add_argument('--intensity', '-i', type=float, default=1.5)
    parser.add_argument('--intensity_decay', '-d', type=float, default=1.0)
    parser.add_argument('--config', type=str, default='../conf/crmnist.json')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--zy_dim', type=int, default=12)
    parser.add_argument('--zx_dim', type=int, default=12)
    parser.add_argument('--zay_dim', type=int, default=12)
    parser.add_argument('--za_dim', type=int, default=12)
    parser.add_argument('--beta_1', type=float, default=1.0)
    parser.add_argument('--beta_2', type=float, default=1.0)
    parser.add_argument('--beta_3', type=float, default=1.0)
    parser.add_argument('--beta_4', type=float, default=1.0)
    parser.add_argument('--alpha_1', type=float, default=1.0, help='Weight for y (digit) classifier loss')
    parser.add_argument('--alpha_2', type=float, default=2.0, help='Weight for a (domain) classifier loss')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--setting', type=str, default='cross-domain', choices=['holdout', 'cross-domain'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize', 'confidence_interval', 'all'])
    
    args = parser.parse_args()
    
    # Set up CUDA if available
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.cuda:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    run_experiment(args)
from core.train import train
from core.test import test_nvae, test_dann
from core.CRMNIST.model import VAE
import torch
from core.CRMNIST.data_generation import generate_crmnist_dataset
from core.CRMNIST.comparison.train import train_nvae, train_diva, train_dann
import core.CRMNIST.utils
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import os

"""TODO:
    - divide dimensions evenly for zy, za, zx when removing zay, (also try splitting only among zy and za)
    - test against domain adversarial network
    - compute metrics for each model using each rotation as a test domain, get mean and std of metrics
    - run each method with multiple random seeds, reporting mean ± standard-deviation.
    - visualize latent space
"""

"""
    Training and evaluation of NVAE and DIVA models

    To run the script, use the following command:

    python -m core.CRMNIST.comparison.evaluation --config conf/crmnist.json --out results/ --cuda
"""

device = None 

def evaluate_models(nvae, diva, dann, test_loader, test_domain=None, path=None):
    """
    Evaluate models on the test set, optionally for a specific domain.
    
    Args:
        nvae: NVAE model
        diva: DIVA model
        dann: DANN model
        test_loader: DataLoader for test set
        test_domain: Optional domain to evaluate on (0-4 for rotations)
    """
    print("Evaluating models" + (f" on domain {test_domain}" if test_domain is not None else ""))
    
    # Filter test loader for specific domain if requested
    if test_domain is not None:
        domain_test_loader = filter_domain_loader(test_loader, test_domain)
    else:
        domain_test_loader = test_loader
    
    test_loss_nvae, test_metrics_nvae = test_nvae(nvae, domain_test_loader, device)
    

    test_loss_diva, test_metrics_diva = test_nvae(diva, domain_test_loader, device)
    
    test_loss_dann, test_metrics_dann = test_dann(dann, domain_test_loader, device)
    

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
   
    return {
        'nvae': {'loss': test_loss_nvae, 'metrics': test_metrics_nvae},
        'diva': {'loss': test_loss_diva, 'metrics': test_metrics_diva},
        'dann': {'loss': test_loss_dann, 'metrics': test_metrics_dann}
    }

def filter_domain_loader(loader, domain, exclude=False):
    """
    Filter a DataLoader to include or exclude samples from a specific domain.
    
    Args:
        loader: Original DataLoader
        domain: Domain to filter for (0-4 for rotations)
        exclude: If True, exclude samples from the specified domain instead of including them
        
    Returns:
        Filtered DataLoader with only samples from the specified domain (or all except that domain)
    """
    domain_samples = []
    for batch in loader:
        x, y, c, r = batch
        # Get rotation domain for each sample
        rotation_indices = torch.argmax(r, dim=1) if torch.max(r) > 0 else torch.zeros(len(r))
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
        raise ValueError(f"No samples found for domain {domain}")
    
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



def run_cross_domain_evaluation(args, nvae, diva, dann, test_loader, spec_data):
    """
    Run evaluation across all domains and compute statistics.
    
    Args:
        args: Command line arguments
        nvae: NVAE model
        diva: DIVA model
        test_loader: DataLoader for test set
        
    Returns:
        Dictionary containing mean and std of metrics across domains
    """
    num_domains = spec_data['num_r_classes'] 
    cross_domain_results = {'nvae': {}, 'diva': {}, 'dann': {}}
    
    # Evaluate on each domain
    for domain in range(num_domains):
        print(f"\nEvaluating on domain {domain} ({domain * 15}°)")
        results = evaluate_models(nvae, diva, dann, test_loader, test_domain=domain, path=os.path.join(args.out, f'cross_domain_domain_{domain}'))
        for model in cross_domain_results:
            cross_domain_results[model][domain] = {
                'loss': results[model]['loss'],
                'metrics': results[model]['metrics']
            }
    
    return cross_domain_results



def run_holdout_evaluation(args, train_loader, test_loader, class_map, spec_data):
    """
    Run evaluation where we hold out one domain at a time for testing.
    
    Args:
        args: Command line arguments
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        
    Returns:
        Dictionary containing results for each holdout domain
    """
    num_domains = 5  # 5 rotation domains
    holdout_results = {'nvae': {}, 'diva': {}, 'dann': {}}
    
    # For each domain, hold it out and train/test
    for holdout_domain in range(num_domains):
        print(f"\nHolding out domain {holdout_domain} ({holdout_domain * 15}°)")
        
        # Filter training data to exclude holdout domain
        filtered_train_loader = filter_domain_loader(train_loader, holdout_domain, exclude=True)
        
        # Train models on filtered training data

        
        nvae, training_results_nvae = train_nvae(args, spec_data,
                                                filtered_train_loader, test_loader, args.out)
        print(f"NVAE training results: \n"
              f"Best model epoch: {training_results_nvae['best_model_epoch']}\n"
              f"Best validation loss: {training_results_nvae['best_validation_loss']}\n"
              f"Best batch metrics: {training_results_nvae['best_batch_metrics']}")  
          
        diva, training_results_diva = train_diva(args, spec_data, 
                                               filtered_train_loader, test_loader, args.out)
        print(f"DIVA training results: \n"
              f"Best model epoch: {training_results_diva['best_model_epoch']}\n"
              f"Best validation loss: {training_results_diva['best_validation_loss']}\n"
              f"Best batch metrics: {training_results_diva['best_batch_metrics']}")
        
        dann, training_results_dann = train_dann(args, spec_data,
                                               filtered_train_loader, test_loader, args.out)
        print(f"DANN training results: \n"
              f"Best model epoch: {training_results_dann['best_model_epoch']}\n"
              f"Best validation loss: {training_results_dann['best_validation_loss']}\n"
              f"Best batch metrics: {training_results_dann['best_batch_metrics']}")
        
        # Test on holdout domain
        holdout_test_loader = filter_domain_loader(test_loader, holdout_domain)
        results = evaluate_models(nvae, diva, dann, holdout_test_loader, path=os.path.join(args.out, f'holdout_domain_{holdout_domain}'))
        
        # Store results
        for model in holdout_results:
            holdout_results[model][holdout_domain] = {
                'loss': results[model]['loss'],
                'metrics': results[model]['metrics']
            }
    
    
    return holdout_results

def analyze_domain_distribution(train_loader, test_loader):
    """
    Analyze and print the distribution of images across domains in both training and test sets.
    """
    # Initialize counters for each domain (0-5)
    train_domain_counts = {i: 0 for i in range(6)}  # Changed from 5 to 6 domains
    test_domain_counts = {i: 0 for i in range(6)}   # Changed from 5 to 6 domains
    
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
            train_domain_counts[domain_idx.item()] += 1
    
    # Count test set distribution
    print("\nCounting test set distribution...")
    for x, y, c, r in test_loader:
        # Convert one-hot encoded labels to indices if needed
        if len(r.shape) > 1:
            domain_indices = torch.argmax(r, dim=1)
        else:
            domain_indices = r
            
        for domain_idx in domain_indices:
            test_domain_counts[domain_idx.item()] += 1
    
    # Print results
    print("\nDomain Distribution:")
    print("Domain | Training Count | Training % | Test Count | Test %")
    print("-" * 60)
    
    total_train = sum(train_domain_counts.values())
    total_test = sum(test_domain_counts.values())
    
    for domain in range(6):  # Changed from 5 to 6 domains
        train_count = train_domain_counts[domain]
        test_count = test_domain_counts[domain]
        train_percent = (train_count / total_train) * 100
        test_percent = (test_count / total_test) * 100
        print(f"{domain:6d} | {train_count:13d} | {train_percent:9.2f}% | {test_count:10d} | {test_percent:6.2f}%")
    
    return train_domain_counts, test_domain_counts

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
    mode = args.mode
    print(f"Running in {mode} mode")
    setting = args.setting
    print(f"Running in {setting} setting")
    
    # Load configuration from JSON
    with open(args.config, 'r') as file:
        spec_data = json.load(file)
    
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
    train_domain_counts, test_domain_counts = analyze_domain_distribution(train_loader, test_loader)
    
    if setting == 'cross-domain' and mode == 'train':
        
        nvae, training_results_nvae = train_nvae(args, spec_data, train_loader, test_loader, models_dir)
        diva, training_results_diva = train_diva(args, spec_data, train_loader, test_loader, models_dir)
        dann, training_results_dann = train_dann(args, spec_data, train_loader, test_loader, models_dir)

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
        print("Finished training models")

        return
    
    if setting == 'holdout' and (mode == 'train' or mode == 'test'):
        # Run holdout evaluation
        holdout_results = run_holdout_evaluation(args, train_loader, test_loader, class_map, spec_data)
        print_results(holdout_results, holdout=True)

        results_path = os.path.join(args.out, 'evaluation_results_holdout.json')
        with open(results_path, 'w') as f:
            json.dump(holdout_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")

        return
    
    # Load models
    nvae_state = torch.load(os.path.join(models_dir, 'nvae_checkpoint.pt'))
    diva_state = torch.load(os.path.join(models_dir, 'diva_checkpoint.pt'))
    dann_state = torch.load(os.path.join(models_dir, 'dann_checkpoint.pt'))

    # Create model instances
    nvae = NVAE(spec_data, z_dim=args.z_dim).to(device)
    diva = DIVA(spec_data, z_dim=args.z_dim).to(device)
    dann = DANN(spec_data, z_dim=args.z_dim).to(device)

    # Load state dictionaries
    nvae.load_state_dict(nvae_state)
    diva.load_state_dict(diva_state)
    dann.load_state_dict(dann_state)

    # Set models to eval mode
    nvae.eval()
    diva.eval()
    dann.eval()

    if setting == 'cross-domain' and mode == 'test':
        # Run cross-domain evaluation
        cross_domain_results = run_cross_domain_evaluation(args, nvae, diva, dann, test_loader, spec_data)
        print_results(cross_domain_results, holdout=False)

        results_path = os.path.join(args.out, 'evaluation_results_cross_domain.json')
        with open(results_path, 'w') as f:
            json.dump(cross_domain_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        return
    
    if mode == 'visualize':

        # visualize latent spaces

        nvae.visualize_latent_spaces(test_loader, device, os.path.join(args.out, 'nvae_latent_space'))
        nvae.visualize_disentanglement(test_loader, device, os.path.join(args.out, 'nvae_disentanglement'))
        nvae.visualize_latent_correlations(test_loader, device, os.path.join(args.out, 'nvae_latent_correlations'))

        diva.visualize_latent_spaces(test_loader, device, os.path.join(args.out, 'diva_latent_space'))
        diva.visualize_disentanglement(test_loader, device, os.path.join(args.out, 'diva_disentanglement'))
        diva.visualize_latent_correlations(test_loader, device, os.path.join(args.out, 'diva_latent_correlations'))

        dann.visualize_latent_space(test_loader, device, os.path.join(args.out, 'dann_latent_space'))

        # visualize reconstruction

        # nvae.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'nvae_reconstruction'))
        # diva.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'diva_reconstruction'))
        # dann.visualize_reconstruction(test_loader, device, os.path.join(args.out, 'dann_reconstruction'))

    

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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--setting', type=str, default='cross-domain', choices=['holdout', 'cross-domain'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize', 'all'])
    
    args = parser.parse_args()
    
    # Set up CUDA if available
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.cuda:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    run_experiment(args)
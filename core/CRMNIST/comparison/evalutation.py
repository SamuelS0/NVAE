from core.train import train
from core.test import test
from core.CRMNIST.model import VAE
import torch
from core.CRMNIST.data_generation import generate_crmnist_dataset, CRMNIST
import core.CRMNIST.utils
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import os

"""
    Training and evaluation of NVAE and DIVA models

    To run the script, use the following command:

    python -m core.CRMNIST.comparison.evalutation --config conf/crmnist.json --out results/ --cuda
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_nvae(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir):
    print("Training NVAE...")
    nvae = VAE(class_map=class_map,
               zy_dim=args.zy_dim,
               zx_dim=args.zx_dim,
               zay_dim=args.zay_dim,
               za_dim=args.za_dim,
               y_dim=num_y_classes,
               a_dim=num_r_classes,
               beta_1=args.beta_1,
               beta_2=args.beta_2,
               beta_3=args.beta_3,
               beta_4=args.beta_4,
               diva=False)
    
    optimizer = optim.Adam(nvae.parameters(), lr=args.learning_rate)
    patience = 5
    training_metrics = train(args, nvae, optimizer, train_loader, test_loader, args.device, patience)

    if training_metrics['best_model_state'] is not None:
        nvae.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    final_model_path = os.path.join(models_dir, f"nvae_model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(nvae.state_dict(), final_model_path)

    return nvae, training_metrics

def train_diva(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir):
    print("Training DIVA...")
    diva = VAE(class_map=class_map,
               zy_dim=args.zy_dim,
               zx_dim=args.zx_dim,
               zay_dim=args.zay_dim,
               za_dim=args.za_dim,
               y_dim=num_y_classes,
               a_dim=num_r_classes,
               beta_1=args.beta_1,
               beta_2=args.beta_2,
               beta_3=args.beta_3,
               beta_4=args.beta_4,
               diva=True)
    
    optimizer = optim.Adam(diva.parameters(), lr=args.learning_rate)
    patience = 5
    training_metrics = train(args, diva, optimizer, train_loader, test_loader, args.device, patience)

    if training_metrics['best_model_state'] is not None:
        diva.load_state_dict(training_metrics['best_model_state'])
        print("Loaded best model for final evaluation")
    
    final_model_path = os.path.join(models_dir, f"diva_model_checkpoint_epoch_{training_metrics['best_model_epoch']}.pt")
    torch.save(diva.state_dict(), final_model_path)

    return diva, training_metrics

def evaluate_models(nvae, diva, test_loader):
    print("Evaluating models")
    
    test_loss_nvae, test_metrics_nvae = test(nvae, test_loader, device)
    test_loss_diva, test_metrics_diva = test(diva, test_loader, device)

    print("--------------------------------")
    print(f"NVAE test results:")
    print(f"Test loss: {test_loss_nvae}")
    print(f"Test metrics: {test_metrics_nvae}")
    
    print("--------------------------------")
    print(f"DIVA test results:")
    print(f"Test loss: {test_loss_diva}")
    print(f"Test metrics: {test_metrics_diva}")
 


def run_experiment(args):
    # Load configuration from JSON
    with open(args.config, 'r') as file:
        spec_data = json.load(file)
    
    models_dir = os.path.join(args.out, 'comparison_models')
    os.makedirs(models_dir, exist_ok=True)
        
    # Prepare domain data
    domain_data = {int(key): value for key, value in spec_data['domain_data'].items()}
    spec_data['domain_data'] = domain_data
    class_map = spec_data['class_map']
    
    # Choose labels subset if not already chosen
    y_c, subsets = core.CRMNIST.utils.choose_label_subset(spec_data)
    spec_data['y_c'] = y_c

    train_dataset = CRMNIST(spec_data, train=True, 
                           transform_intensity=args.intensity,
                           transform_decay=args.intensity_decay)
    
    test_dataset = CRMNIST(spec_data, train=False,
                          transform_intensity=args.intensity,
                          transform_decay=args.intensity_decay)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    num_y_classes = train_dataset.dataset.num_y_classes
    num_r_classes = train_dataset.dataset.num_r_classes

    nvae, training_results_nvae = train_nvae(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir)
    diva, training_results_diva = train_diva(args, num_y_classes, num_r_classes, class_map, train_loader, test_loader, models_dir)

    print("NVAE training results:")
    print(f"Best model epoch: {training_results_nvae['best_model_epoch']}\nBest validation loss: {training_results_nvae['best_validation_loss']}\nBest batch metrics: {training_results_nvae['best_batch_metrics']}")
    print("DIVA training results:")
    print(f"Best model epoch: {training_results_diva['best_model_epoch']}\nBest validation loss: {training_results_diva['best_validation_loss']}\nBest batch metrics: {training_results_diva['best_batch_metrics']}")

    evaluate_models(nvae, diva, test_loader)
    
if __name__ == "__main__":
    parser = core.utils.get_parser('CRMNIST')
    parser.add_argument('--intensity', '-i', type=float, default=1.5)
    parser.add_argument('--intensity_decay', '-d', type=float, default=1.0)
    parser.add_argument('--config', type=str, default='../conf/crmnist.json')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--zy_dim', type=int, default=32)
    parser.add_argument('--zx_dim', type=int, default=32)
    parser.add_argument('--zay_dim', type=int, default=32)
    parser.add_argument('--za_dim', type=int, default=32)
    parser.add_argument('--beta_1', type=float, default=1.0)
    parser.add_argument('--beta_2', type=float, default=1.0)
    parser.add_argument('--beta_3', type=float, default=1.0)
    parser.add_argument('--beta_4', type=float, default=1.0)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    
    args = parser.parse_args()
    
    # Set up CUDA if available
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.cuda:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    run_experiment(args)
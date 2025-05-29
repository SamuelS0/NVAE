from disentanglement_visualization import visualize_disentanglement
from torch.utils.data import DataLoader
from run_wild import initialize_model, get_args
# After training your model
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torch
from utils_wild import (
    prepare_data, 
    visualize_reconstructions,
    select_diverse_sample_batch,
    generate_images_latent
)

dataset = get_dataset(
            dataset="camelyon17", 
            download=False, 
            root_dir='/midtier/cocolab/scratch/ofn9004/WILD',
            unlabeled=False
        )
args = get_args()
args.cuda = args.cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

num_classes = 2
num_domains = 5
model = initialize_model(args, num_classes, num_domains)
model.load_state_dict(torch.load('/home/ofn9004/NVAE/core/vae_recon10_a100_b2_big_latent/models/model_best.pt'))

model.eval()
train_loader, val_loader, test_loader = prepare_data(dataset, args)

# Run disentanglement analysis
visualize_disentanglement(
    model=model,
    dataloader=val_loader,
    device=args.device,
    save_path='./results/disentanglement.png',
    num_variations=7,
    num_examples=3
)
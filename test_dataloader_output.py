#!/usr/bin/env python3
"""Test what the CRMNIST dataloader actually returns"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Test if Subset preserves tuple length
class TestDS(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return (1, 2, 3, 4)  # 4 values like CRMNIST

ds = TestDS()
subset = Subset(ds, [0, 1, 2])
loader = DataLoader(subset, batch_size=2)
batch = next(iter(loader))
print(f'Subset test: Batch has {len(batch)} elements')
print(f'Expected: 4, Got: {len(batch)}')

# Now test with actual CRMNIST if available
try:
    from core.CRMNIST.data_generation import generate_crmnist_dataset, load_crmnist_spec_data
    import os

    print("\nTesting actual CRMNIST dataset...")

    # Load spec data
    spec_data_path = os.path.join('core', 'CRMNIST', 'spec_data_default.json')
    if os.path.exists(spec_data_path):
        spec_data = load_crmnist_spec_data(spec_data_path)

        # Generate a small test dataset
        print("Generating test dataset...")
        train_dataset = generate_crmnist_dataset(spec_data, train=True, use_cache=True)

        # Create subset and loader
        print("Creating subset and dataloader...")
        train_subset, _ = torch.utils.data.random_split(
            train_dataset, [100, len(train_dataset) - 100],
            generator=torch.Generator().manual_seed(42)
        )

        loader = DataLoader(train_subset, batch_size=4, shuffle=False)

        # Get one batch
        batch = next(iter(loader))
        print(f'\nCRMNIST dataloader output:')
        print(f'  Number of elements in batch: {len(batch)}')
        if len(batch) == 4:
            print(f'  x shape: {batch[0].shape}')
            print(f'  y shape: {batch[1].shape}')
            print(f'  c shape: {batch[2].shape}')
            print(f'  r shape: {batch[3].shape}')
            print('\n✅ CRMNIST dataloader correctly returns 4 values')
        else:
            print(f'\n❌ ERROR: CRMNIST dataloader returns {len(batch)} values instead of 4!')
            for i, item in enumerate(batch):
                if hasattr(item, 'shape'):
                    print(f'  Element {i} shape: {item.shape}')
    else:
        print(f"Spec data not found at {spec_data_path}")

except Exception as e:
    print(f"Error testing CRMNIST: {e}")
    import traceback
    traceback.print_exc()

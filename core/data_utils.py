"""
Data utilities for domain filtering and OOD dataset creation.

This module provides utilities for:
- Filtering datasets by domain/hospital indices
- Creating in-distribution (ID) and out-of-distribution (OOD) splits
- Supporting both CRMNIST and WILD dataset formats
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset


class FilteredDataset(Dataset):
    """
    Wrapper dataset that filters samples based on domain indices.

    Args:
        dataset: Original dataset (CRMNISTDataset or WILD)
        domain_indices: List of domain indices to include
        dataset_type: 'crmnist' or 'wild'
        exclude: If True, exclude the specified domains instead of including them
    """
    def __init__(self, dataset, domain_indices, dataset_type='crmnist', exclude=False):
        self.dataset = dataset
        self.domain_indices = set(domain_indices)
        self.dataset_type = dataset_type
        self.exclude = exclude

        # Build index mapping
        self.valid_indices = []
        for idx in range(len(dataset)):
            domain = self._get_domain_from_index(idx)
            if exclude:
                # Include if domain NOT in domain_indices
                if domain not in self.domain_indices:
                    self.valid_indices.append(idx)
            else:
                # Include if domain IS in domain_indices
                if domain in self.domain_indices:
                    self.valid_indices.append(idx)

        print(f"FilteredDataset: {len(self.valid_indices)}/{len(dataset)} samples "
              f"({'excluding' if exclude else 'including'} domains {sorted(domain_indices)})")

    def _get_domain_from_index(self, idx):
        """Extract domain index from a sample."""
        sample = self.dataset[idx]

        if self.dataset_type == 'crmnist':
            # CRMNIST: (img, y_label, c_label, r_label) where r_label is one-hot
            r_label = sample[3]
            if isinstance(r_label, torch.Tensor):
                domain = torch.argmax(r_label).item()
            else:
                domain = np.argmax(r_label)
        elif self.dataset_type == 'wild':
            # WILD: (x, y, metadata) where metadata[0] is hospital ID
            metadata = sample[2]
            if isinstance(metadata, torch.Tensor):
                domain = metadata[0].item()
            else:
                domain = metadata[0]
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        return domain

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        return self.dataset[original_idx]


def filter_dataset_by_domains(dataset, domain_indices, dataset_type='crmnist', include=True):
    """
    Filter dataset to include or exclude specific domains.

    Args:
        dataset: CRMNISTDataset or WILD dataset
        domain_indices: List of domain indices to filter by
        dataset_type: 'crmnist' or 'wild'
        include: If True, keep only these domains; if False, exclude them

    Returns:
        FilteredDataset containing only the specified samples
    """
    return FilteredDataset(
        dataset,
        domain_indices,
        dataset_type=dataset_type,
        exclude=not include
    )


def create_ood_split(dataset, ood_domain_idx, dataset_type='crmnist'):
    """
    Split dataset into in-distribution (ID) and out-of-distribution (OOD) portions.

    Args:
        dataset: Full dataset containing all domains
        ood_domain_idx: Index of domain to withhold for OOD testing
        dataset_type: 'crmnist' or 'wild'

    Returns:
        (id_dataset, ood_dataset) tuple of FilteredDataset objects
    """
    # Determine total number of domains based on dataset type
    if dataset_type == 'crmnist':
        num_domains = 6  # Domains 0-5
    elif dataset_type == 'wild':
        num_domains = 5  # Hospitals 0-4
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Validate OOD domain index
    if ood_domain_idx < 0 or ood_domain_idx >= num_domains:
        raise ValueError(f"OOD domain index {ood_domain_idx} out of range [0, {num_domains-1}]")

    # Create ID dataset (all domains except OOD)
    id_domains = [i for i in range(num_domains) if i != ood_domain_idx]
    id_dataset = filter_dataset_by_domains(
        dataset,
        id_domains,
        dataset_type=dataset_type,
        include=True
    )

    # Create OOD dataset (only the withheld domain)
    ood_dataset = filter_dataset_by_domains(
        dataset,
        [ood_domain_idx],
        dataset_type=dataset_type,
        include=True
    )

    return id_dataset, ood_dataset


def get_domain_from_sample(sample, dataset_type='crmnist'):
    """
    Extract domain label from a sample tuple.

    Args:
        sample: Dataset sample (format depends on dataset_type)
        dataset_type: 'crmnist' or 'wild'

    Returns:
        Domain index as integer
    """
    if dataset_type == 'crmnist':
        # CRMNIST: (img, y_label, c_label, r_label)
        r_label = sample[3]
        if isinstance(r_label, torch.Tensor):
            return torch.argmax(r_label).item()
        else:
            return np.argmax(r_label)
    elif dataset_type == 'wild':
        # WILD: (x, y, metadata)
        metadata = sample[2]
        if isinstance(metadata, torch.Tensor):
            return metadata[0].item()
        else:
            return metadata[0]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def print_domain_distribution(dataset, dataset_type='crmnist', name='Dataset'):
    """
    Print the distribution of samples across domains.

    Args:
        dataset: Dataset to analyze
        dataset_type: 'crmnist' or 'wild'
        name: Name for display purposes
    """
    from collections import Counter

    domains = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        domain = get_domain_from_sample(sample, dataset_type)
        domains.append(domain)

    domain_counts = Counter(domains)
    print(f"\n{name} Domain Distribution:")
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        percentage = 100 * count / len(dataset)
        print(f"  Domain {domain}: {count:5d} samples ({percentage:5.2f}%)")
    print(f"  Total: {len(dataset)} samples across {len(domain_counts)} domains")

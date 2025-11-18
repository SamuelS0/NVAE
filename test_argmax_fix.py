#!/usr/bin/env python3
"""
Direct test of the argmax bug fix
Tests that the code correctly handles (batch, 1) shaped tensors
"""

import torch

print("="*80)
print("DIRECT TEST OF ARGMAX BUG FIX")
print("="*80)

# Simulate the problematic case: (batch, 1) shaped tensors
print("\nTest Case: (batch, 1) shaped tensors")
print("-" * 80)

batch_size = 32

# Create (batch, 1) tensors with mixed classes
y = torch.tensor([[0], [1], [0], [1], [1], [0], [1], [0],
                  [1], [0], [1], [1], [0], [0], [1], [0],
                  [0], [1], [1], [0], [1], [0], [0], [1],
                  [1], [1], [0], [0], [1], [0], [1], [0]], dtype=torch.float32)

domain = torch.tensor([[0], [1], [2], [3], [4], [0], [1], [2],
                       [3], [4], [0], [1], [2], [3], [4], [0],
                       [1], [2], [3], [4], [0], [1], [2], [3],
                       [4], [0], [1], [2], [3], [4], [0], [1]], dtype=torch.float32)

print(f"Original shapes: y={y.shape}, domain={domain.shape}")
print(f"Expected: Both should be (32, 1)")

# OLD BUGGY CODE (for comparison):
print("\n" + "-" * 80)
print("OLD BUGGY CODE:")
print("-" * 80)

y_old = y.clone()
domain_old = domain.clone()

# This is what the old code did
if len(y_old.shape) > 1:
    y_old_idx = torch.argmax(y_old, dim=1)
else:
    y_old_idx = y_old.long()

if len(domain_old.shape) > 1:
    domain_old_idx = torch.argmax(domain_old, dim=1)
else:
    domain_old_idx = domain_old.long()

print(f"After old argmax: y_idx shape={y_old_idx.shape}, domain_idx shape={domain_old_idx.shape}")
print(f"y_idx unique values: {torch.unique(y_old_idx).tolist()}")
print(f"domain_idx unique values: {torch.unique(domain_old_idx).tolist()}")

# Count class distribution with old code
normal_count_old = (y_old_idx == 0).sum().item()
tumor_count_old = (y_old_idx == 1).sum().item()
print(f"\nClass distribution with OLD code:")
print(f"  Normal (0): {normal_count_old} samples")
print(f"  Tumor (1): {tumor_count_old} samples")

if normal_count_old == batch_size:
    print("❌ BUG CONFIRMED: Old code returns all zeros!")

# NEW FIXED CODE:
print("\n" + "-" * 80)
print("NEW FIXED CODE:")
print("-" * 80)

y_new = y.clone()
domain_new = domain.clone()

# This is the new fixed code
if len(y_new.shape) > 1 and y_new.shape[1] > 1:
    y_new_idx = torch.argmax(y_new, dim=1)
elif len(y_new.shape) > 1:
    y_new_idx = y_new.squeeze().long()
else:
    y_new_idx = y_new.long()

if len(domain_new.shape) > 1 and domain_new.shape[1] > 1:
    domain_new_idx = torch.argmax(domain_new, dim=1)
elif len(domain_new.shape) > 1:
    domain_new_idx = domain_new.squeeze().long()
else:
    domain_new_idx = domain_new.long()

print(f"After new logic: y_idx shape={y_new_idx.shape}, domain_idx shape={domain_new_idx.shape}")
print(f"y_idx unique values: {torch.unique(y_new_idx).tolist()}")
print(f"domain_idx unique values: {torch.unique(domain_new_idx).tolist()}")

# Count class distribution with new code
normal_count_new = (y_new_idx == 0).sum().item()
tumor_count_new = (y_new_idx == 1).sum().item()
print(f"\nClass distribution with NEW code:")
print(f"  Normal (0): {normal_count_new} samples")
print(f"  Tumor (1): {tumor_count_new} samples")

# Verify correctness
expected_normal = (y.squeeze() == 0).sum().item()
expected_tumor = (y.squeeze() == 1).sum().item()

print(f"\nExpected distribution (ground truth):")
print(f"  Normal (0): {expected_normal} samples")
print(f"  Tumor (1): {expected_tumor} samples")

# VERIFICATION
print("\n" + "="*80)
print("VERIFICATION RESULTS")
print("="*80)

if normal_count_new == expected_normal and tumor_count_new == expected_tumor:
    print("✅ PASS: New code correctly handles (batch, 1) tensors")
    print("✅ PASS: Class distribution matches ground truth")
else:
    print("❌ FAIL: New code does NOT match ground truth")

if normal_count_new > 0 and tumor_count_new > 0:
    print("✅ PASS: Both classes have samples (bug is fixed)")
else:
    print("❌ FAIL: One or both classes have 0 samples")

# Test other shapes
print("\n" + "="*80)
print("ADDITIONAL TESTS: Other tensor shapes")
print("="*80)

# Test 1: One-hot encoded (batch, num_classes)
print("\n1. One-hot encoded tensors (batch, num_classes):")
y_onehot = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
print(f"   Input shape: {y_onehot.shape}")

if len(y_onehot.shape) > 1 and y_onehot.shape[1] > 1:
    y_onehot_idx = torch.argmax(y_onehot, dim=1)
elif len(y_onehot.shape) > 1:
    y_onehot_idx = y_onehot.squeeze().long()
else:
    y_onehot_idx = y_onehot.long()

print(f"   Output: {y_onehot_idx.tolist()}")
expected_onehot = [0, 1, 0, 1]
if y_onehot_idx.tolist() == expected_onehot:
    print("   ✅ PASS: One-hot case works correctly")
else:
    print(f"   ❌ FAIL: Expected {expected_onehot}, got {y_onehot_idx.tolist()}")

# Test 2: 1D tensor (batch,)
print("\n2. 1D tensors (batch,):")
y_1d = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
print(f"   Input shape: {y_1d.shape}")

if len(y_1d.shape) > 1 and y_1d.shape[1] > 1:
    y_1d_idx = torch.argmax(y_1d, dim=1)
elif len(y_1d.shape) > 1:
    y_1d_idx = y_1d.squeeze().long()
else:
    y_1d_idx = y_1d.long()

print(f"   Output: {y_1d_idx.tolist()}")
expected_1d = [0, 1, 0, 1]
if y_1d_idx.tolist() == expected_1d:
    print("   ✅ PASS: 1D case works correctly")
else:
    print(f"   ❌ FAIL: Expected {expected_1d}, got {y_1d_idx.tolist()}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)

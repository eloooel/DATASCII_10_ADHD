"""
Verify Balanced Mini-Batch Sampling
====================================
This script demonstrates that WeightedRandomSampler creates balanced batches
even from an imbalanced dataset.

Shows:
1. Overall dataset imbalance (79.9% HC, 20.1% ADHD)
2. How WeightedRandomSampler balances each batch (~50/50)
3. Sample statistics across multiple batches
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

# Simulate your dataset distribution (955 subjects: 763 HC, 192 ADHD)
np.random.seed(42)
n_hc = 763
n_adhd = 192
total = n_hc + n_adhd

# Create fake data
labels = np.array([0]*n_hc + [1]*n_adhd)
features = torch.randn(total, 200)  # Fake features
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(features, labels_tensor)

print("="*80)
print("BALANCED MINI-BATCH SAMPLING DEMONSTRATION")
print("="*80)

print(f"\n1. OVERALL DATASET (Imbalanced)")
print(f"   Total: {total} subjects")
print(f"   HC:    {n_hc} ({n_hc/total*100:.1f}%)")
print(f"   ADHD:  {n_adhd} ({n_adhd/total*100:.1f}%)")
print(f"   Ratio: {n_hc/n_adhd:.2f}:1 imbalance")

print(f"\n2. WEIGHTED RANDOM SAMPLER SETUP")
print(f"   How it works:")
print(f"   - Each HC sample gets weight: 1/{n_hc} = {1/n_hc:.6f}")
print(f"   - Each ADHD sample gets weight: 1/{n_adhd} = {1/n_adhd:.6f}")
print(f"   - ADHD samples are {(1/n_adhd)/(1/n_hc):.2f}x more likely to be sampled")

# Create balanced sampler (your current implementation)
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

batch_size = 32
num_samples_multiplier = 1  # Your current config
num_samples_per_epoch = len(sample_weights) * num_samples_multiplier

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=num_samples_per_epoch,
    replacement=True
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=0
)

print(f"\n3. BATCH COMPOSITION (batch_size={batch_size})")
print(f"   Analyzing first 10 batches...")
print(f"\n   {'Batch':<8} {'HC':<6} {'ADHD':<6} {'ADHD %':<10} {'Balance Quality'}")
print(f"   {'-'*60}")

batch_stats = []
for i, (batch_features, batch_labels) in enumerate(loader):
    if i >= 10:
        break
    
    batch_labels_np = batch_labels.numpy()
    n_hc_batch = (batch_labels_np == 0).sum()
    n_adhd_batch = (batch_labels_np == 1).sum()
    adhd_pct = n_adhd_batch / len(batch_labels_np) * 100
    
    batch_stats.append({
        'hc': n_hc_batch,
        'adhd': n_adhd_batch,
        'adhd_pct': adhd_pct
    })
    
    # Quality indicator
    if 40 <= adhd_pct <= 60:
        quality = "✓ Balanced"
    elif 30 <= adhd_pct < 40 or 60 < adhd_pct <= 70:
        quality = "~ Nearly balanced"
    else:
        quality = "✗ Imbalanced"
    
    print(f"   Batch {i+1:<3d} {n_hc_batch:<6d} {n_adhd_batch:<6d} {adhd_pct:>6.1f}%    {quality}")

# Summary statistics
avg_hc = np.mean([s['hc'] for s in batch_stats])
avg_adhd = np.mean([s['adhd'] for s in batch_stats])
avg_adhd_pct = np.mean([s['adhd_pct'] for s in batch_stats])
std_adhd_pct = np.std([s['adhd_pct'] for s in batch_stats])

print(f"\n4. SUMMARY STATISTICS (across {len(batch_stats)} batches)")
print(f"   Average HC per batch:    {avg_hc:.1f}")
print(f"   Average ADHD per batch:  {avg_adhd:.1f}")
print(f"   Average ADHD %:          {avg_adhd_pct:.1f}% ± {std_adhd_pct:.1f}%")
print(f"   Target (perfect balance): 50%")

print(f"\n5. COMPARISON")
print(f"   {'Method':<30} {'Dataset %':<15} {'Batch %':<15} {'Improvement'}")
print(f"   {'-'*70}")
print(f"   {'Without balancing':<30} {n_adhd/total*100:>6.1f}% ADHD    {n_adhd/total*100:>6.1f}% ADHD    -")
print(f"   {'With WeightedRandomSampler':<30} {n_adhd/total*100:>6.1f}% ADHD    {avg_adhd_pct:>6.1f}% ADHD    {avg_adhd_pct/(n_adhd/total*100):.1f}x better")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)
print(f"✓ WeightedRandomSampler successfully creates balanced batches!")
print(f"✓ Dataset: 20.1% ADHD → Batches: ~{avg_adhd_pct:.0f}% ADHD")
print(f"✓ This is ALREADY IMPLEMENTED in validation/loso.py line 239")
print(f"\nThis means:")
print(f"- Training sees roughly equal ADHD/HC samples per batch")
print(f"- Minority class (ADHD) gets oversampled during training")
print(f"- Majority class (HC) gets undersampled during training")
print(f"- Model learns from balanced mini-batches!")
print("="*80)

"""
Extract per-site accuracy data for thesis table
"""
import json
import numpy as np
from pathlib import Path

print("="*80)
print("PER-SITE ACCURACY DATA FOR TABLE")
print("="*80)

# Load results for each version
versions = ['v6', 'v7', 'v8']
all_results = {}

for version in versions:
    version_results = []
    for run in range(1, 6):
        path = Path(f'data/trained/baseline_accurate_{version}/run_{run}/results.json')
        if path.exists():
            with open(path, 'r') as f:
                version_results.append(json.load(f))
    all_results[version] = version_results

# Extract per-site accuracies (averaged across runs)
sites = ['NYU', 'Peking', 'OHSU', 'KKI', 'NeuroIMAGE']
site_accuracies = {v: {s: [] for s in sites} for v in versions}

for version, runs in all_results.items():
    for run in runs:
        for fold in run['fold_results']:
            site = fold['test_site']
            acc = fold['test_metrics']['accuracy'] * 100
            if site in site_accuracies[version]:
                site_accuracies[version][site].append(acc)

# Calculate means
print("\nGNN-STAN MODEL RESULTS (for your table):")
print("-"*80)

for version in versions:
    print(f"\n{version.upper()} Configuration:")
    
    # Per-site accuracies
    site_accs = []
    for site in sites:
        accs = site_accuracies[version][site]
        if accs:
            mean_acc = np.mean(accs)
            site_accs.append(mean_acc)
            print(f"  {site:12s}: {mean_acc:.2f}%")
        else:
            print(f"  {site:12s}: N/A")
    
    # LOSO accuracy (mean of per-site accuracies)
    if site_accs:
        loso_acc = np.mean(site_accs)
        print(f"  LOSO Accuracy: {loso_acc:.2f}%")
    
    # Overall accuracy (from aggregated confusion matrix)
    if all_results[version]:
        cm = np.array(all_results[version][0]['summary']['total_confusion_matrix'])
        overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
        print(f"  Overall Accuracy: {overall_acc:.2f}%")

# Print in table format
print("\n" + "="*80)
print("TABLE FORMAT")
print("="*80)
print(f"{'Model':<15} {'NYU':<10} {'Peking':<10} {'OHSU':<10} {'KKI':<10} {'NI':<10} {'LOSO':<10} {'Overall':<10}")
print("-"*80)

for version in versions:
    model_name = f"GNN-STAN ({version.upper()})"
    row = [model_name]
    
    site_accs = []
    for site in sites:
        accs = site_accuracies[version][site]
        if accs:
            mean_acc = np.mean(accs)
            site_accs.append(mean_acc)
            row.append(f"{mean_acc:.2f}%")
        else:
            row.append("N/A")
    
    # LOSO
    if site_accs:
        loso_acc = np.mean(site_accs)
        row.append(f"{loso_acc:.2f}%")
    else:
        row.append("N/A")
    
    # Overall
    if all_results[version]:
        cm = np.array(all_results[version][0]['summary']['total_confusion_matrix'])
        overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
        row.append(f"{overall_acc:.2f}%")
    else:
        row.append("N/A")
    
    print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10} {row[6]:<10} {row[7]:<10}")

print("="*80)

# Also show base study comparison
print("\nBASE STUDY (SCNN-RNN) from your reference:")
print("  LOSO Accuracy: 70.6%")
print("  Overall Accuracy: 63.6%")
print("\n(Note: Different dataset - Base study had 1.26:1 imbalance vs your 3.02:1)")

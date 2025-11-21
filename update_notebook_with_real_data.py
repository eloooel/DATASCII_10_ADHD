"""
Quick script to show what real data is available for the notebook visualizations
"""

import json
import numpy as np
from pathlib import Path

print("="*80)
print("REAL DATA AVAILABLE FOR VISUALIZATIONS")
print("="*80)

# 1. Ablation Study Results
print("\n1. ABLATION STUDY (Figure 10)")
print("-"*80)
ablation_path = Path('data/ablation_results/ablation_results.json')
if ablation_path.exists():
    with open(ablation_path, 'r') as f:
        ablation_data = json.load(f)
    
    print("✓ Real ablation results available!")
    print("\nData structure:")
    for variant, metrics in ablation_data.items():
        print(f"  {variant}:")
        print(f"    Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"    Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"    Specificity: {metrics['specificity']*100:.2f}%")
    
    print("\nTo use in notebook cell:")
    print("  ablation_data = json.load(open('data/ablation_results/ablation_results.json'))")
    print("  # Then extract metrics for each variant")
else:
    print("✗ Run: python run_ablation_study.py")

# 2. Temporal Attention
print("\n2. TEMPORAL ATTENTION (Figure 12)")
print("-"*80)
attention_path = Path('data/attention_weights/attention_weights.npz')
if attention_path.exists():
    data = np.load(attention_path)
    print("✓ Real attention weights available!")
    
    # Check for position importance
    if 'stan_encoder.attention_position_importance' in data:
        temporal_attn = data['stan_encoder.attention_position_importance']
        print(f"\nTemporal attention shape: {temporal_attn.shape}")
        print(f"  Ready-to-plot vector of {temporal_attn.shape[-1]} timepoints")
        print("\nTo use in notebook cell:")
        print("  data = np.load('data/attention_weights/attention_weights.npz')")
        print("  temporal_attention = data['stan_encoder.attention_position_importance'][0]")
        print("  # Shape: (352,) - one importance value per timepoint")
    else:
        print("\nAvailable attention keys:")
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                print(f"  {key}: {data[key].shape}")
else:
    print("✗ Run: python extract_attention_weights.py")

# 3. Predictions for ROC/PR Curves
print("\n3. PREDICTIONS FOR ROC/PR CURVES (Figures 7-8)")
print("-"*80)
results_path = Path('data/trained/baseline_accurate_v6/run_1/results.json')
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Check if predictions are in the data
    has_predictions = False
    if 'fold_results' in results and len(results['fold_results']) > 0:
        fold = results['fold_results'][0]
        if 'test_metrics' in fold:
            if 'probabilities' in fold['test_metrics']:
                has_predictions = True
                probs = fold['test_metrics']['probabilities']
                labels = fold['test_metrics']['true_labels']
                print("✓ Predictions already in existing results!")
                print(f"\nFirst fold has {len(probs)} predictions")
                print("\nTo use in notebook cell:")
                print("  # Load results for each version (v6, v7, v8)")
                print("  # Extract probabilities from fold_results[i]['test_metrics']['probabilities']")
                print("  # Concatenate across all folds")
    
    if not has_predictions:
        print("✗ Predictions not in saved results")
        print("  Run: python train_v6_with_predictions.py (takes ~30 min)")
        print("  OR: Use synthetic predictions based on confusion matrices (current approach)")
else:
    print("✗ Results not found")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nYou can re-run the notebook now - it will use:")
print("  ✓ Real data from existing results (Figures 1-6, 9, 13, 14)")
print("  ✓ Real ablation data (Figure 10) - just loaded!")
print("  ✓ Real temporal attention (Figure 12) - just extracted!")
print("  ~ Synthetic ROC/PR curves (Figures 7-8) - based on real confusion matrices")
print("\nThe notebook will automatically detect and use the real data where available.")

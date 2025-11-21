"""
Training script for baseline_accurate_v8 configuration
Aggressive class weighting: [1.0, 5.0] (increased from v6's [1.0, 4.0])
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.baseline_accurate_v8_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE
from validation.loso import LeaveOneSiteOutValidator
from utils.data_loader import load_features_and_labels
import torch
import numpy as np
import json
import pandas as pd

def print_results(run_results, run_idx):
    """Print results for a single run with per-site breakdown"""
    print(f"\n{'='*80}")
    print(f"RUN {run_idx} RESULTS")
    print(f"{'='*80}")
    
    # Per-site accuracy
    print("\nPer-Site Accuracy:")
    for fold in run_results['fold_results']:
        test_site = fold['test_site']
        acc = fold['test_metrics']['accuracy'] * 100
        sens = fold['test_metrics']['sensitivity'] * 100
        spec = fold['test_metrics']['specificity'] * 100
        print(f"  - {test_site:12s}: {acc:.2f}% (Sens: {sens:.2f}%, Spec: {spec:.2f}%)")
    
    # LOSO accuracy (mean across folds)
    loso_acc = run_results['summary']['accuracy_mean'] * 100
    loso_std = run_results['summary']['accuracy_std'] * 100
    print(f"\nLOSO Accuracy: {loso_acc:.2f}% ± {loso_std:.2f}%")
    
    # Overall accuracy (from aggregated confusion matrix)
    cm = np.array(run_results['summary']['total_confusion_matrix'])
    overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    overall_sens = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    overall_spec = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"\nOverall Metrics (Aggregated):")
    print(f"  Accuracy:    {overall_acc:.2f}%")
    print(f"  Sensitivity: {overall_sens:.2f}%")
    print(f"  Specificity: {overall_spec:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"  [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
    print(f"   [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
    print(f"  (Predicted {cm[1,1]} out of {cm[1,0] + cm[1,1]} ADHD cases)")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Train baseline accurate model v8 (AGGRESSIVE WEIGHTING)')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--output-dir', type=str, default='data/trained/baseline_accurate_v8')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BASELINE V8: AGGRESSIVE CLASS WEIGHTING")
    print("="*80)
    print("Configuration:")
    print("  - Class weights: [1.0, 5.0] (5x emphasis on ADHD)")
    print("  - Label smoothing: 0.05")
    print("  - Balanced mini-batch sampling")
    print("  - Binary cross-entropy loss")
    print("\nExpected: Higher sensitivity (50-60%) than v6, may sacrifice specificity")
    print("Purpose: Test if more aggressive weighting improves minority class detection")
    print("="*80)
    
    # Load data with merged Peking sites
    print("\nLoading data...")
    feature_manifest_path = Path('data/features/feature_manifest.csv')
    manifest_df = pd.read_csv(feature_manifest_path)
    
    # Filter to baseline sites
    manifest_df = manifest_df[manifest_df['site'].isin(TRAINING_CONFIG_BASELINE['sites'])]
    
    print(f"\nDataset Summary:")
    print(f"  Total subjects: {len(manifest_df)}")
    print(f"  HC: {(manifest_df['diagnosis'] == 0).sum()} ({(manifest_df['diagnosis'] == 0).sum()/len(manifest_df)*100:.1f}%)")
    print(f"  ADHD: {(manifest_df['diagnosis'] == 1).sum()} ({(manifest_df['diagnosis'] == 1).sum()/len(manifest_df)*100:.1f}%)")
    print(f"  Sites: {TRAINING_CONFIG_BASELINE['sites']}")
    
    # Load features
    fc_matrices, roi_timeseries, labels, sites = load_features_and_labels(manifest_df)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Run training
    seeds = TRAINING_CONFIG_BASELINE['seeds'][:args.num_runs]
    
    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx} - Seed: {seed}")
        print(f"{'='*80}")
        
        # Update seed in config
        config = TRAINING_CONFIG_BASELINE.copy()
        config['seed'] = seed
        
        # Create validator
        validator = LeaveOneSiteOutValidator(
            model_config=MODEL_CONFIG_BASELINE,
            training_config=config
        )
        
        # Run LOSO validation
        results = validator.validate(
            fc_matrices=fc_matrices,
            roi_timeseries=roi_timeseries,
            labels=labels,
            sites=sites
        )
        
        # Print results
        print_results(results, run_idx)
        
        # Save results
        output_path = Path(args.output_dir) / f'run_{run_idx}'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_serializable = {
            'seed': seed,
            'summary': results['summary'],
            'fold_results': [
                {
                    'test_site': fold['test_site'],
                    'fold_id': fold['fold_id'],
                    'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                    for k, v in fold['test_metrics'].items()},
                    'final_epoch': fold['final_epoch'],
                    'training_time': fold['training_time']
                }
                for fold in results['fold_results']
            ]
        }
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save summary
        cm = np.array(results['summary']['total_confusion_matrix'])
        summary = {
            'run': run_idx,
            'seed': seed,
            'loso_accuracy_mean': float(results['summary']['accuracy_mean']),
            'loso_accuracy_std': float(results['summary']['accuracy_std']),
            'overall_accuracy': float((cm[0,0] + cm[1,1]) / cm.sum()),
            'overall_sensitivity': float(cm[1,1] / (cm[1,0] + cm[1,1])),
            'overall_specificity': float(cm[0,0] / (cm[0,0] + cm[0,1]))
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"ALL RUNS COMPLETED")
    print(f"{'='*80}")
    print("\nNext Step: Compare v6 (weights=4.0) vs v8 (weights=5.0)")
    print("Determine optimal class weight for best sensitivity/specificity balance.")

if __name__ == '__main__':
    main()

"""
Training script for baseline_accurate_v6 configuration
Merges Peking_1/2/3 into single site for proper LOSO validation
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.baseline_accurate_v6_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE
from validation.loso import LOSOValidator
from utils.data_loader import load_all_sites
import torch
import numpy as np
import json

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
    loso_acc = run_results['test_metrics']['accuracy']['mean'] * 100
    loso_std = run_results['test_metrics']['accuracy']['std'] * 100
    print(f"\nLOSO Accuracy: {loso_acc:.2f}% ± {loso_std:.2f}%")
    
    # Overall accuracy (from aggregated confusion matrix)
    cm = np.array(run_results['overall_confusion_matrix'])
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
    parser = argparse.ArgumentParser(description='Train baseline accurate model v6')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--output-dir', type=str, default='data/trained/baseline_accurate_v6')
    args = parser.parse_args()
    
    # Load data with merged Peking sites
    print("Loading data...")
    features, labels, sites, subject_ids = load_all_sites(
        TRAINING_CONFIG_BASELINE['sites'],
        max_rois=TRAINING_CONFIG_BASELINE.get('max_rois')
    )
    
    print(f"\nDataset Summary:")
    print(f"  Total subjects: {len(labels)}")
    print(f"  HC: {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
    print(f"  ADHD: {(labels == 1).sum()} ({(labels == 1).sum()/len(labels)*100:.1f}%)")
    print(f"  Sites: {TRAINING_CONFIG_BASELINE['sites']}")
    
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
        validator = LOSOValidator(
            model_config=MODEL_CONFIG_BASELINE,
            training_config=config,
            device=device
        )
        
        # Run LOSO validation
        results = validator.validate(
            features=features,
            labels=labels,
            sites=sites,
            subject_ids=subject_ids
        )
        
        # Print results
        print_results(results, run_idx)
        
        # Save results
        output_path = Path(args.output_dir) / f'run_{run_idx}'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_serializable = {
            'seed': seed,
            'test_metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                for kk, vv in v.items()} 
                            for k, v in results['test_metrics'].items()},
            'overall_confusion_matrix': results['overall_confusion_matrix'].tolist(),
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
        summary = {
            'run': run_idx,
            'seed': seed,
            'loso_accuracy_mean': float(results['test_metrics']['accuracy']['mean']),
            'loso_accuracy_std': float(results['test_metrics']['accuracy']['std']),
            'overall_accuracy': float((results['overall_confusion_matrix'][0,0] + results['overall_confusion_matrix'][1,1]) / results['overall_confusion_matrix'].sum()),
            'overall_sensitivity': float(results['overall_confusion_matrix'][1,1] / (results['overall_confusion_matrix'][1,0] + results['overall_confusion_matrix'][1,1])),
            'overall_specificity': float(results['overall_confusion_matrix'][0,0] / (results['overall_confusion_matrix'][0,0] + results['overall_confusion_matrix'][0,1]))
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"ALL RUNS COMPLETED")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

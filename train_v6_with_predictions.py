"""
Re-train V6 with save_predictions=True to capture probabilities for ROC/PR curves
Runs just 1 run to save time, since we already have the full results
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.baseline_accurate_v6_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE
from validation.loso import LeaveOneSiteOutValidator
from utils.data_loader import load_features_and_labels
import torch
import numpy as np
import json
import pandas as pd

def main():
    print("="*80)
    print("TRAINING V6 WITH PREDICTION SAVING ENABLED")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    feature_manifest_path = Path('data/features/feature_manifest.csv')
    manifest_df = pd.read_csv(feature_manifest_path)
    manifest_df = manifest_df[manifest_df['site'].isin(TRAINING_CONFIG_BASELINE['sites'])]
    
    print(f"\nDataset Summary:")
    print(f"  Total subjects: {len(manifest_df)}")
    print(f"  HC: {(manifest_df['diagnosis'] == 0).sum()} ({(manifest_df['diagnosis'] == 0).sum()/len(manifest_df)*100:.1f}%)")
    print(f"  ADHD: {(manifest_df['diagnosis'] == 1).sum()} ({(manifest_df['diagnosis'] == 1).sum()/len(manifest_df)*100:.1f}%)")
    
    # Load features
    fc_matrices, roi_timeseries, labels, sites = load_features_and_labels(manifest_df)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Update config to save predictions
    config = TRAINING_CONFIG_BASELINE.copy()
    config['save_predictions'] = True  # Enable prediction saving
    config['seed'] = 42  # Use first seed
    
    print("\n✓ save_predictions=True enabled")
    print("  This will save per-subject probabilities for ROC/PR curve generation")
    
    # Create validator
    validator = LeaveOneSiteOutValidator(
        model_config=MODEL_CONFIG_BASELINE,
        training_config=config
    )
    
    # Run LOSO validation
    print("\nRunning LOSO validation...")
    results = validator.validate(
        fc_matrices=fc_matrices,
        roi_timeseries=roi_timeseries,
        labels=labels,
        sites=sites
    )
    
    # Save results with predictions
    output_path = Path('data/trained/baseline_accurate_v6_with_predictions')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if predictions were saved
    predictions_saved = False
    for fold in results['fold_results']:
        if fold.get('predictions_data') is not None:
            predictions_saved = True
            break
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nPredictions saved: {predictions_saved}")
    
    if predictions_saved:
        print("\nPrediction data structure per fold:")
        for fold in results['fold_results']:
            if fold.get('predictions_data'):
                pred_data = fold['predictions_data']
                print(f"  Fold {fold['fold_id']} ({fold['test_site']}):")
                print(f"    - Probabilities: {len(pred_data['probabilities'])} subjects")
                print(f"    - Shape: {np.array(pred_data['probabilities']).shape}")
    
    # Save detailed results
    results_serializable = {
        'seed': config['seed'],
        'summary': results['summary'],
        'fold_results': [
            {
                'test_site': fold['test_site'],
                'fold_id': fold['fold_id'],
                'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                for k, v in fold['test_metrics'].items()},
                'final_epoch': fold['final_epoch'],
                'training_time': fold['training_time'],
                'predictions_data': fold.get('predictions_data', None)
            }
            for fold in results['fold_results']
        ]
    }
    
    with open(output_path / 'results_with_predictions.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Print summary
    cm = np.array(results['summary']['total_confusion_matrix'])
    overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    overall_sens = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    overall_spec = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {overall_acc:.2f}%")
    print(f"  Sensitivity: {overall_sens:.2f}%")
    print(f"  Specificity: {overall_spec:.2f}%")
    
    print(f"\n✓ Results saved to: {output_path / 'results_with_predictions.json'}")
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print("\nYou can now use this data to generate real ROC/PR curves in the notebook!")

if __name__ == '__main__':
    main()

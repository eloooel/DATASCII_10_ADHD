"""
Baseline-Accurate Training Script
==================================
Reproduces the base study methodology with multiple runs for statistical robustness.

This script:
1. Loads the baseline-accurate configuration
2. Runs LOSO cross-validation multiple times with different random seeds
3. Saves results for each run
4. Computes and reports mean±std statistics across all runs

Usage:
    python scripts/train_baseline_accurate.py --num-runs 5 --output-dir data/trained/baseline_accurate
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.baseline_accurate_config import (
    MODEL_CONFIG_BASELINE,
    TRAINING_CONFIG_BASELINE,
    EXPECTED_RESULTS_BASELINE
)
from validation.loso import LeaveOneSiteOutValidator
from utils.data_loader import load_features_and_labels


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_baseline_data(feature_manifest_path: Path, sites: List[str]):
    """Load data for baseline sites only"""
    print(f"\nLoading data from: {feature_manifest_path}")
    
    # Load feature manifest
    manifest_df = pd.read_csv(feature_manifest_path)
    
    # Filter to baseline sites
    manifest_df = manifest_df[manifest_df['site'].isin(sites)]
    
    print(f"Total subjects after site filtering: {len(manifest_df)}")
    print(f"Sites: {manifest_df['site'].unique()}")
    print(f"Class distribution: {manifest_df['diagnosis'].value_counts().to_dict()}")
    
    # Load features and labels
    fc_matrices, roi_timeseries, labels, site_ids = load_features_and_labels(manifest_df)
    
    return fc_matrices, roi_timeseries, labels, site_ids, manifest_df


def run_single_training(
    run_id: int,
    seed: int,
    fc_matrices: np.ndarray,
    roi_timeseries: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Run a single training experiment with LOSO validation"""
    
    print(f"\n{'='*80}")
    print(f"RUN {run_id + 1} - Seed: {seed}")
    print(f"{'='*80}")
    
    # Set random seed
    set_random_seed(seed)
    
    # Create validator
    validator = LeaveOneSiteOutValidator(
        model_config=model_config,
        training_config=training_config
    )
    
    # Run LOSO validation
    results = validator.validate(
        fc_matrices=fc_matrices,
        roi_timeseries=roi_timeseries,
        labels=labels,
        sites=sites
    )
    
    # Save results for this run
    run_dir = output_dir / f"run_{run_id + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary = results['summary']
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    
    print(f"\n✅ Run {run_id + 1} completed!")
    print(f"Accuracy: {summary.get('accuracy_mean', summary.get('overall_accuracy', 0)):.4f}")
    print(f"Sensitivity: {summary.get('sensitivity_mean', summary.get('overall_sensitivity', 0)):.4f}")
    print(f"Specificity: {summary.get('specificity_mean', summary.get('overall_specificity', 0)):.4f}")
    print(f"AUC: {summary.get('auc_mean', 0):.4f}")
    
    return summary


def aggregate_runs(all_summaries: List[Dict[str, Any]], output_dir: Path):
    """Aggregate results across all runs and compute statistics"""
    
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS ALL RUNS")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easy statistics
    df = pd.DataFrame(all_summaries)
    
    # Compute mean and std for each metric - handle both _mean suffix and direct keys
    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1', 'precision', 'recall']
    
    statistics = {}
    for metric in metrics:
        # Try metric_mean first, then overall_metric, then metric
        mean_key = f"{metric}_mean"
        overall_key = f"overall_{metric}"
        
        if mean_key in df.columns:
            mean = df[mean_key].mean()
            std = df[mean_key].std()
        elif overall_key in df.columns:
            mean = df[overall_key].mean()
            std = df[overall_key].std()
        elif metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
        else:
            continue
            
        statistics[metric] = {
            'mean': mean,
            'std': std,
            'formatted': f"{mean*100:.2f}±{std*100:.2f}%"
        }
    
    # Save statistics
    with open(output_dir / "aggregate_statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # Save detailed results table
    df.to_csv(output_dir / "all_runs_summary.csv", index=False)
    
    # Print results
    print("\n📊 FINAL RESULTS (Mean±Std across all runs):")
    print("-" * 60)
    for metric in ['accuracy', 'sensitivity', 'specificity', 'auc']:
        if metric in statistics:
            print(f"{metric.capitalize():15s}: {statistics[metric]['formatted']}")
    print("-" * 60)
    
    # Compare with expected results
    print("\n🎯 Comparison with Expected Results:")
    print("-" * 60)
    for metric, expected in EXPECTED_RESULTS_BASELINE.items():
        if metric in statistics:
            achieved = statistics[metric]['formatted']
            print(f"{metric.capitalize():15s}: {achieved} (Expected: {expected})")
    print("-" * 60)
    
    return statistics


def main():
    parser = argparse.ArgumentParser(description='Run baseline-accurate training with multiple runs')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of independent training runs (default: 5)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/trained/baseline_accurate'),
                        help='Output directory for results')
    parser.add_argument('--feature-manifest', type=Path, default=Path('data/features/feature_manifest.csv'),
                        help='Path to feature manifest CSV')
    parser.add_argument('--start-run', type=int, default=0,
                        help='Starting run index (for resuming)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_info = {
        'model_config': MODEL_CONFIG_BASELINE,
        'training_config': TRAINING_CONFIG_BASELINE,
        'expected_results': EXPECTED_RESULTS_BASELINE,
        'num_runs': args.num_runs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(args.output_dir / "configuration.json", 'w') as f:
        json.dump(config_info, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("BASELINE-ACCURATE TRAINING")
    print(f"{'='*80}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sites: {', '.join(TRAINING_CONFIG_BASELINE['sites'])}")
    print(f"Max ROIs: {TRAINING_CONFIG_BASELINE['max_rois']}")
    print(f"Validation: {TRAINING_CONFIG_BASELINE['validation_strategy'].upper()}")
    print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*80}")
    
    # Load data
    fc_matrices, roi_timeseries, labels, sites, manifest_df = load_baseline_data(
        feature_manifest_path=args.feature_manifest,
        sites=TRAINING_CONFIG_BASELINE['sites']
    )
    
    # Run multiple experiments
    all_summaries = []
    seeds = TRAINING_CONFIG_BASELINE['seeds'][:args.num_runs]
    
    for run_id in range(args.start_run, args.num_runs):
        seed = seeds[run_id]
        
        summary = run_single_training(
            run_id=run_id,
            seed=seed,
            fc_matrices=fc_matrices,
            roi_timeseries=roi_timeseries,
            labels=labels,
            sites=sites,
            model_config=MODEL_CONFIG_BASELINE,
            training_config=TRAINING_CONFIG_BASELINE,
            output_dir=args.output_dir
        )
        
        all_summaries.append(summary)
    
    # Aggregate results
    statistics = aggregate_runs(all_summaries, args.output_dir)
    
    print(f"\n✅ All runs completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nTo compare with other experiments:")
    print(f"  python scripts/compare_experiments.py --exp1 {args.output_dir} --exp2 <other_experiment>")


if __name__ == "__main__":
    main()

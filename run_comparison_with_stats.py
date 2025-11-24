"""
Run experiment comparison with statistical treatment
Compares GNN-STAN variants (V6, V7, V8) with baseline SCNN-RNN
"""

from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from compare_experiments import ExperimentComparer

def main():
    # Define experiment paths
    data_dir = Path("data/trained")
    
    experiments = {
        "baseline_accurate_v6": {
            "path": data_dir / "baseline_accurate_v6" / "run_1" / "results.json",
            "description": "GNN-STAN V6 (class_weights=1:4, LOSO)"
        },
        "baseline_accurate_v7": {
            "path": data_dir / "baseline_accurate_v7" / "run_1" / "results.json",
            "description": "GNN-STAN V7 (no class weights, LOSO)"
        },
        "baseline_accurate_v8": {
            "path": data_dir / "baseline_accurate_v8" / "run_1" / "results.json",
            "description": "GNN-STAN V8 (class_weights=1:5, LOSO)"
        }
    }
    
    # Create comparer
    output_dir = Path("experiments/statistical_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparer = ExperimentComparer(output_dir=output_dir)
    
    # Add experiments
    print("\n" + "="*70)
    print("Loading experiments...")
    print("="*70)
    
    for exp_name, exp_info in experiments.items():
        if exp_info["path"].exists():
            comparer.add_experiment(
                name=exp_name,
                results_path=exp_info["path"],
                description=exp_info["description"]
            )
            print(f"✓ Loaded: {exp_name}")
        else:
            print(f"⚠️  Not found: {exp_info['path']}")
    
    # Generate full report with statistical tests
    print("\n" + "="*70)
    print("Generating comparison report with statistical treatment...")
    print("="*70)
    
    # Use V7 as baseline (best performing model)
    comparer.generate_full_report(baseline_experiment="baseline_accurate_v7")
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files include:")
    print("  - Paired t-tests for metrics (accuracy, sensitivity, specificity, AUC, F1)")
    print("  - McNemar's test for classification agreement")
    print("  - Descriptive statistics (mean, std, improvement)")
    print("  - Effect sizes (Cohen's d)")
    print("  - Per-site accuracy comparisons")
    print("\nNote: McNemar's test requires prediction data to be available")
    print("      in the results.json files (requires save_predictions=True)")

if __name__ == '__main__':
    main()

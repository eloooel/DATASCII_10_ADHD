"""
Experiment Comparison Tool

Compare results across multiple experiments for thesis tables and analysis.
Generates comparison tables, statistical tests, and publication-ready summaries.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy import stats
from collections import defaultdict


class ExperimentComparer:
    """Compare multiple experiments and generate thesis-ready tables"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.experiments = {}
        self.output_dir = output_dir or Path('./experiments/comparisons')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_experiment(
        self,
        name: str,
        results_path: Path,
        description: str = ""
    ) -> None:
        """
        Add experiment results for comparison
        
        Args:
            name: Experiment name
            results_path: Path to results JSON file
            description: Optional description
        """
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        self.experiments[name] = {
            'description': description,
            'results': results,
            'results_path': str(results_path)
        }
        
        print(f"✅ Added experiment: {name}")
    
    def add_experiment_from_dict(
        self,
        name: str,
        results: Dict[str, Any],
        description: str = ""
    ) -> None:
        """
        Add experiment from results dictionary
        
        Args:
            name: Experiment name
            results: Results dictionary
            description: Optional description
        """
        self.experiments[name] = {
            'description': description,
            'results': results,
            'results_path': 'in-memory'
        }
        
        print(f"✅ Added experiment: {name}")
    
    def create_comparison_table(
        self,
        metrics: List[str] = None,
        include_ci: bool = True,
        format_style: str = 'latex'
    ) -> pd.DataFrame:
        """
        Create comparison table of all experiments
        
        Args:
            metrics: Metrics to include (default: all)
            include_ci: Include confidence intervals
            format_style: 'latex', 'markdown', or 'csv'
        
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1', 'precision', 'recall']
        
        rows = []
        
        for exp_name, exp_data in self.experiments.items():
            results = exp_data['results']
            
            # Extract summary metrics
            if 'summary' in results:
                summary = results['summary']
            elif 'validation_results' in results and 'summary' in results['validation_results']:
                summary = results['validation_results']['summary']
            else:
                print(f"⚠️  Warning: No summary found for {exp_name}")
                continue
            
            row = {'Experiment': exp_name}
            
            for metric in metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                ci_lower_key = f'{metric}_ci_lower'
                ci_upper_key = f'{metric}_ci_upper'
                
                if mean_key in summary:
                    mean_val = summary[mean_key]
                    std_val = summary.get(std_key, 0)
                    
                    if include_ci and ci_lower_key in summary:
                        ci_lower = summary[ci_lower_key]
                        ci_upper = summary[ci_upper_key]
                        row[metric.capitalize()] = f"{mean_val:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
                    else:
                        row[metric.capitalize()] = f"{mean_val:.4f} ± {std_val:.4f}"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save in different formats
        if format_style == 'latex':
            latex_path = self.output_dir / 'comparison_table.tex'
            with open(latex_path, 'w') as f:
                f.write(df.to_latex(index=False, float_format="%.4f"))
            print(f"📄 LaTeX table saved to: {latex_path}")
        
        elif format_style == 'markdown':
            markdown_path = self.output_dir / 'comparison_table.md'
            with open(markdown_path, 'w') as f:
                f.write(df.to_markdown(index=False))
            print(f"📄 Markdown table saved to: {markdown_path}")
        
        # Always save CSV
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV table saved to: {csv_path}")
        
        return df
    
    def statistical_comparison(
        self,
        baseline_experiment: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison against baseline
        
        Args:
            baseline_experiment: Name of baseline experiment
            alpha: Significance level
        
        Returns:
            Dictionary with statistical test results
        """
        if baseline_experiment not in self.experiments:
            raise ValueError(f"Baseline experiment '{baseline_experiment}' not found")
        
        baseline_results = self.experiments[baseline_experiment]['results']
        
        # Extract fold-level metrics from baseline
        baseline_fold_metrics = self._extract_fold_metrics(baseline_results)
        
        comparisons = {}
        
        for exp_name, exp_data in self.experiments.items():
            if exp_name == baseline_experiment:
                continue
            
            exp_results = exp_data['results']
            exp_fold_metrics = self._extract_fold_metrics(exp_results)
            
            # Perform t-tests for each metric
            metric_tests = {}
            
            for metric in ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1']:
                if metric in baseline_fold_metrics and metric in exp_fold_metrics:
                    baseline_values = baseline_fold_metrics[metric]
                    exp_values = exp_fold_metrics[metric]
                    
                    # Paired t-test (if same folds) or independent t-test
                    if len(baseline_values) == len(exp_values):
                        t_stat, p_value = stats.ttest_rel(exp_values, baseline_values)
                        test_type = 'paired'
                    else:
                        t_stat, p_value = stats.ttest_ind(exp_values, baseline_values)
                        test_type = 'independent'
                    
                    # Effect size (Cohen's d)
                    mean_diff = np.mean(exp_values) - np.mean(baseline_values)
                    pooled_std = np.sqrt((np.std(exp_values)**2 + np.std(baseline_values)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    metric_tests[metric] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'cohens_d': float(cohens_d),
                        'effect_size': self._interpret_cohens_d(cohens_d),
                        'test_type': test_type,
                        'baseline_mean': float(np.mean(baseline_values)),
                        'experiment_mean': float(np.mean(exp_values)),
                        'improvement': float(mean_diff)
                    }
            
            comparisons[exp_name] = {
                'description': exp_data['description'],
                'metric_tests': metric_tests
            }
        
        # Save statistical comparison
        stats_path = self.output_dir / f'statistical_comparison_vs_{baseline_experiment}.json'
        with open(stats_path, 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"📊 Statistical comparison saved to: {stats_path}")
        
        return comparisons
    
    def _extract_fold_metrics(self, results: Dict) -> Dict[str, List[float]]:
        """Extract fold-level metrics from results"""
        metrics = defaultdict(list)
        
        # Handle different result structures
        fold_results = None
        if 'fold_results' in results:
            fold_results = results['fold_results']
        elif 'validation_results' in results and 'fold_results' in results['validation_results']:
            fold_results = results['validation_results']['fold_results']
        
        if fold_results:
            for fold in fold_results:
                if 'test_metrics' in fold:
                    for metric, value in fold['test_metrics'].items():
                        if isinstance(value, (int, float)):
                            metrics[metric].append(value)
        
        return dict(metrics)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def create_ablation_summary(self) -> pd.DataFrame:
        """
        Create ablation study summary table
        
        Assumes experiments follow naming: baseline, gnn_only, stan_only, etc.
        """
        rows = []
        
        # Define expected ablation experiments
        ablation_order = [
            'gnn_only',
            'stan_only',
            'hybrid_no_attention',
            'hybrid_no_fusion',
            'full_hybrid'
        ]
        
        for exp_name in ablation_order:
            if exp_name in self.experiments:
                exp_data = self.experiments[exp_name]
                results = exp_data['results']
                
                # Extract metrics
                if 'summary_metrics' in results:
                    metrics = results['summary_metrics']
                elif 'summary' in results:
                    metrics = {
                        'accuracy': results['summary'].get('accuracy_mean', 0),
                        'sensitivity': results['summary'].get('sensitivity_mean', 0),
                        'specificity': results['summary'].get('specificity_mean', 0),
                        'auc': results['summary'].get('auc_mean', 0),
                        'f1': results['summary'].get('f1_mean', 0)
                    }
                else:
                    continue
                
                rows.append({
                    'Model Variant': exp_name.replace('_', ' ').title(),
                    'Description': exp_data.get('description', ''),
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Sensitivity': f"{metrics['sensitivity']:.4f}",
                    'Specificity': f"{metrics['specificity']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}"
                })
        
        df = pd.DataFrame(rows)
        
        # Save
        csv_path = self.output_dir / 'ablation_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"📊 Ablation summary saved to: {csv_path}")
        
        # LaTeX version
        latex_path = self.output_dir / 'ablation_summary.tex'
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False))
        print(f"📄 LaTeX ablation table saved to: {latex_path}")
        
        return df
    
    def create_site_comparison_table(self) -> pd.DataFrame:
        """
        Create site-wise performance comparison (for LOSO experiments)
        """
        site_data = defaultdict(dict)
        
        for exp_name, exp_data in self.experiments.items():
            results = exp_data['results']
            
            # Check if LOSO results
            if 'fold_results' in results:
                for fold in results['fold_results']:
                    if 'test_site' in fold and 'test_metrics' in fold:
                        site = fold['test_site']
                        site_data[site][exp_name] = fold['test_metrics'].get('accuracy', 0)
        
        if not site_data:
            print("⚠️  No site-specific results found (LOSO validation required)")
            return pd.DataFrame()
        
        # Create DataFrame
        rows = []
        for site, exp_results in site_data.items():
            row = {'Site': site}
            row.update({f"{exp}: Acc": f"{acc:.4f}" for exp, acc in exp_results.items()})
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save
        csv_path = self.output_dir / 'site_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"📊 Site comparison saved to: {csv_path}")
        
        return df
    
    def generate_full_report(
        self,
        baseline_experiment: Optional[str] = None
    ) -> None:
        """
        Generate comprehensive comparison report
        
        Args:
            baseline_experiment: Optional baseline for statistical tests
        """
        print("\n" + "="*70)
        print("📊 GENERATING COMPREHENSIVE EXPERIMENT COMPARISON REPORT")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f'report_{timestamp}'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Comparison table
        print("\n1️⃣  Creating comparison table...")
        comparison_df = self.create_comparison_table(format_style='latex')
        comparison_df.to_csv(report_dir / 'comparison_table.csv', index=False)
        
        # 2. Statistical comparison (if baseline provided)
        if baseline_experiment and baseline_experiment in self.experiments:
            print(f"\n2️⃣  Running statistical tests (baseline: {baseline_experiment})...")
            stats_results = self.statistical_comparison(baseline_experiment)
            
            # Create significance table
            sig_rows = []
            for exp_name, exp_stats in stats_results.items():
                for metric, test_result in exp_stats['metric_tests'].items():
                    sig_rows.append({
                        'Experiment': exp_name,
                        'Metric': metric.capitalize(),
                        'Baseline Mean': f"{test_result['baseline_mean']:.4f}",
                        'Experiment Mean': f"{test_result['experiment_mean']:.4f}",
                        'Improvement': f"{test_result['improvement']:.4f}",
                        'p-value': f"{test_result['p_value']:.4f}",
                        'Significant': '✓' if test_result['significant'] else '✗',
                        "Cohen's d": f"{test_result['cohens_d']:.3f}",
                        'Effect Size': test_result['effect_size']
                    })
            
            sig_df = pd.DataFrame(sig_rows)
            sig_df.to_csv(report_dir / 'statistical_tests.csv', index=False)
            print(f"   Saved to: {report_dir / 'statistical_tests.csv'}")
        
        # 3. Ablation summary (if applicable)
        print("\n3️⃣  Creating ablation summary...")
        try:
            ablation_df = self.create_ablation_summary()
            ablation_df.to_csv(report_dir / 'ablation_summary.csv', index=False)
        except Exception as e:
            print(f"   ⚠️  Could not create ablation summary: {e}")
        
        # 4. Site comparison (if LOSO)
        print("\n4️⃣  Creating site comparison...")
        site_df = self.create_site_comparison_table()
        if not site_df.empty:
            site_df.to_csv(report_dir / 'site_comparison.csv', index=False)
        
        # 5. Summary report (text)
        print("\n5️⃣  Writing summary report...")
        report_path = report_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT COMPARISON SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of experiments: {len(self.experiments)}\n\n")
            
            f.write("Experiments:\n")
            for exp_name, exp_data in self.experiments.items():
                f.write(f"  - {exp_name}: {exp_data['description']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("COMPARISON TABLE\n")
            f.write("="*70 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            
            if baseline_experiment:
                f.write("\n\n" + "="*70 + "\n")
                f.write(f"STATISTICAL TESTS (vs. {baseline_experiment})\n")
                f.write("="*70 + "\n\n")
                f.write(sig_df.to_string(index=False))
        
        print(f"\n✅ Full report generated in: {report_dir}")
        print(f"\nMain files:")
        print(f"  - comparison_table.csv")
        print(f"  - statistical_tests.csv")
        print(f"  - ablation_summary.csv")
        print(f"  - site_comparison.csv")
        print(f"  - summary_report.txt")


def main():
    """Command-line interface for experiment comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                       help='Paths to experiment result JSON files')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each experiment')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Baseline experiment name for statistical tests')
    parser.add_argument('--output-dir', type=str, default='./experiments/comparisons',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if len(args.experiments) != len(args.names):
        raise ValueError("Number of experiments must match number of names")
    
    # Create comparer
    comparer = ExperimentComparer(output_dir=Path(args.output_dir))
    
    # Add experiments
    for name, exp_path in zip(args.names, args.experiments):
        comparer.add_experiment(name, Path(exp_path))
    
    # Generate report
    comparer.generate_full_report(baseline_experiment=args.baseline)


if __name__ == '__main__':
    main()

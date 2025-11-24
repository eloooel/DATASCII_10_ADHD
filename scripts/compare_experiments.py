"""
Experiment Comparison Tool

Compare results across multiple experiments for thesis tables and analysis.
Generates comparison tables, statistical tests, and publication-ready summaries.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency
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
        Perform comprehensive statistical comparison against baseline
        
        Includes:
        - Paired t-tests for continuous metrics (accuracy, sensitivity, etc.)
        - McNemar's test for paired classification results
        - Effect size calculations (Cohen's d)
        - Descriptive statistics (mean, std)
        
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
        baseline_predictions = self._extract_predictions(baseline_results)
        
        comparisons = {}
        
        for exp_name, exp_data in self.experiments.items():
            if exp_name == baseline_experiment:
                continue
            
            exp_results = exp_data['results']
            exp_fold_metrics = self._extract_fold_metrics(exp_results)
            exp_predictions = self._extract_predictions(exp_results)
            
            comparison_results = {
                'description': exp_data['description'],
                'metric_tests': {},
                'mcnemar_test': None,
                'descriptive_stats': {}
            }
            
            # 1. PAIRED T-TESTS for each metric
            for metric in ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1']:
                if metric in baseline_fold_metrics and metric in exp_fold_metrics:
                    baseline_values = np.array(baseline_fold_metrics[metric])
                    exp_values = np.array(exp_fold_metrics[metric])
                    
                    # Ensure same length for paired test
                    min_len = min(len(baseline_values), len(exp_values))
                    baseline_values = baseline_values[:min_len]
                    exp_values = exp_values[:min_len]
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(exp_values, baseline_values)
                    
                    # Effect size (Cohen's d for paired samples)
                    mean_diff = np.mean(exp_values - baseline_values)
                    std_diff = np.std(exp_values - baseline_values)
                    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                    
                    # Descriptive statistics
                    comparison_results['descriptive_stats'][metric] = {
                        'baseline_mean': float(np.mean(baseline_values)),
                        'baseline_std': float(np.std(baseline_values)),
                        'experiment_mean': float(np.mean(exp_values)),
                        'experiment_std': float(np.std(exp_values)),
                        'mean_difference': float(mean_diff),
                        'std_difference': float(std_diff)
                    }
                    
                    comparison_results['metric_tests'][metric] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < alpha),
                        'cohens_d': float(cohens_d),
                        'effect_size': self._interpret_cohens_d(cohens_d),
                        'test_type': 'paired_t_test',
                        'baseline_mean': float(np.mean(baseline_values)),
                        'baseline_std': float(np.std(baseline_values)),
                        'experiment_mean': float(np.mean(exp_values)),
                        'experiment_std': float(np.std(exp_values)),
                        'improvement': float(mean_diff),
                        'improvement_percent': float((mean_diff / np.mean(baseline_values) * 100) if np.mean(baseline_values) != 0 else 0)
                    }
            
            # 2. McNEMAR'S TEST for classification agreement
            if baseline_predictions and exp_predictions:
                mcnemar_result = self._perform_mcnemar_test(
                    baseline_predictions, 
                    exp_predictions
                )
                comparison_results['mcnemar_test'] = mcnemar_result
            
            comparisons[exp_name] = comparison_results
        
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
    
    def _extract_predictions(self, results: Dict) -> Optional[List[Tuple[int, int]]]:
        """
        Extract prediction pairs (y_true, y_pred) from fold results
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            List of (true_label, predicted_label) tuples, or None if unavailable
        """
        predictions = []
        
        if 'fold_results' not in results:
            return None
        
        for fold in results['fold_results']:
            if 'test_metrics' in fold:
                test_metrics = fold['test_metrics']
                
                # Try to get predictions and true labels
                if 'predictions' in test_metrics and 'true_labels' in test_metrics:
                    y_true = test_metrics['true_labels']
                    y_pred = test_metrics['predictions']
                    
                    # Add to prediction pairs
                    for true_label, pred_label in zip(y_true, y_pred):
                        predictions.append((int(true_label), int(pred_label)))
                
                # Alternative: reconstruct from probabilities if available
                elif 'probabilities' in test_metrics and 'true_labels' in test_metrics:
                    y_true = test_metrics['true_labels']
                    probs = test_metrics['probabilities']
                    
                    # Convert probabilities to predictions (threshold = 0.5)
                    y_pred = [1 if p >= 0.5 else 0 for p in probs]
                    
                    for true_label, pred_label in zip(y_true, y_pred):
                        predictions.append((int(true_label), int(pred_label)))
        
        return predictions if predictions else None
    
    def _perform_mcnemar_test(
        self, 
        baseline_predictions: List[Tuple[int, int]], 
        exp_predictions: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Perform McNemar's test for paired nominal data
        
        Tests whether two classifiers disagree in the same way on paired samples
        
        Args:
            baseline_predictions: List of (y_true, y_pred) for baseline
            exp_predictions: List of (y_true, y_pred) for experiment
            
        Returns:
            Dictionary with McNemar's test results
        """
        # Match predictions by aligning indices (assuming same order)
        min_len = min(len(baseline_predictions), len(exp_predictions))
        baseline_predictions = baseline_predictions[:min_len]
        exp_predictions = exp_predictions[:min_len]
        
        # Build contingency table:
        # [[both_correct, baseline_correct_exp_wrong],
        #  [baseline_wrong_exp_correct, both_wrong]]
        
        both_correct = 0
        baseline_correct_exp_wrong = 0
        baseline_wrong_exp_correct = 0
        both_wrong = 0
        
        for (y_true_b, y_pred_b), (y_true_e, y_pred_e) in zip(baseline_predictions, exp_predictions):
            # Ensure same ground truth
            assert y_true_b == y_true_e, "Mismatched ground truth in prediction pairs"
            y_true = y_true_b
            
            baseline_correct = (y_pred_b == y_true)
            exp_correct = (y_pred_e == y_true)
            
            if baseline_correct and exp_correct:
                both_correct += 1
            elif baseline_correct and not exp_correct:
                baseline_correct_exp_wrong += 1
            elif not baseline_correct and exp_correct:
                baseline_wrong_exp_correct += 1
            else:
                both_wrong += 1
        
        # McNemar's test focuses on disagreements (b and c in 2x2 table)
        # Null hypothesis: b = c (models disagree equally)
        b = baseline_correct_exp_wrong
        c = baseline_wrong_exp_correct
        
        # Contingency table for chi-square test
        # Using continuity correction for small samples
        contingency_table = np.array([[both_correct, b], [c, both_wrong]])
        
        # Calculate McNemar's statistic with continuity correction
        if b + c == 0:
            # No disagreements - models are identical
            mcnemar_statistic = 0.0
            p_value = 1.0
        else:
            mcnemar_statistic = ((abs(b - c) - 1) ** 2) / (b + c)
            # Chi-square with 1 degree of freedom
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(mcnemar_statistic, df=1)
        
        # Interpretation
        total_samples = min_len
        agreement_rate = (both_correct + both_wrong) / total_samples if total_samples > 0 else 0
        disagreement_rate = (b + c) / total_samples if total_samples > 0 else 0
        
        return {
            'mcnemar_statistic': float(mcnemar_statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'contingency_table': {
                'both_correct': int(both_correct),
                'baseline_correct_exp_wrong': int(b),
                'baseline_wrong_exp_correct': int(c),
                'both_wrong': int(both_wrong)
            },
            'summary': {
                'total_samples': int(total_samples),
                'agreement_rate': float(agreement_rate),
                'disagreement_rate': float(disagreement_rate),
                'baseline_advantage': int(b),
                'experiment_advantage': int(c),
                'net_advantage': int(c - b),  # Positive favors experiment
                'interpretation': self._interpret_mcnemar(p_value, c, b)
            }
        }
    
    def _interpret_mcnemar(self, p_value: float, exp_advantage: int, baseline_advantage: int) -> str:
        """
        Interpret McNemar's test result
        
        Args:
            p_value: P-value from McNemar's test
            exp_advantage: Cases where experiment was correct but baseline was wrong
            baseline_advantage: Cases where baseline was correct but experiment was wrong
            
        Returns:
            Human-readable interpretation
        """
        if p_value >= 0.05:
            return "No significant difference in classification performance (p >= 0.05)"
        
        net_advantage = exp_advantage - baseline_advantage
        
        if net_advantage > 0:
            return f"Experiment significantly outperforms baseline (p < 0.05, +{net_advantage} net correct)"
        elif net_advantage < 0:
            return f"Baseline significantly outperforms experiment (p < 0.05, {net_advantage} net correct)"
        else:
            return "Significant difference detected but equal advantages (p < 0.05)"
    
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
        mcnemar_df = None  # Initialize outside conditional
        sig_df = None  # Initialize sig_df as well
        if baseline_experiment and baseline_experiment in self.experiments:
            print(f"\n2️⃣  Running statistical tests (baseline: {baseline_experiment})...")
            stats_results = self.statistical_comparison(baseline_experiment)
            
            # Create significance table for metric tests
            sig_rows = []
            for exp_name, exp_stats in stats_results.items():
                # Metric tests (t-tests)
                for metric, test_result in exp_stats['metric_tests'].items():
                    sig_rows.append({
                        'Experiment': exp_name,
                        'Metric': metric.capitalize(),
                        'Baseline Mean': f"{test_result['baseline_mean']:.4f}",
                        'Baseline Std': f"{test_result['baseline_std']:.4f}",
                        'Experiment Mean': f"{test_result['experiment_mean']:.4f}",
                        'Experiment Std': f"{test_result['experiment_std']:.4f}",
                        'Improvement': f"{test_result['improvement']:.4f}",
                        'Improvement %': f"{test_result['improvement_percent']:.2f}%",
                        'p-value': f"{test_result['p_value']:.4f}",
                        'Significant': '✓' if test_result['significant'] else '✗',
                        "Cohen's d": f"{test_result['cohens_d']:.3f}",
                        'Effect Size': test_result['effect_size']
                    })
            
            sig_df = pd.DataFrame(sig_rows)
            sig_df.to_csv(report_dir / 'statistical_tests.csv', index=False)
            print(f"   Saved metric tests to: {report_dir / 'statistical_tests.csv'}")
            
            # Create McNemar's test table
            mcnemar_rows = []
            for exp_name, exp_stats in stats_results.items():
                if exp_stats['mcnemar_test']:
                    mcnemar = exp_stats['mcnemar_test']
                    summary = mcnemar['summary']
                    contingency = mcnemar['contingency_table']
                    
                    mcnemar_rows.append({
                        'Experiment': exp_name,
                        'McNemar Statistic': f"{mcnemar['mcnemar_statistic']:.4f}",
                        'p-value': f"{mcnemar['p_value']:.4f}",
                        'Significant': '✓' if mcnemar['significant'] else '✗',
                        'Total Samples': summary['total_samples'],
                        'Both Correct': contingency['both_correct'],
                        'Baseline Only Correct': contingency['baseline_correct_exp_wrong'],
                        'Experiment Only Correct': contingency['baseline_wrong_exp_correct'],
                        'Both Wrong': contingency['both_wrong'],
                        'Net Advantage': summary['net_advantage'],
                        'Agreement Rate': f"{summary['agreement_rate']:.4f}",
                        'Interpretation': summary['interpretation']
                    })
            
            if mcnemar_rows:
                mcnemar_df = pd.DataFrame(mcnemar_rows)
                mcnemar_df.to_csv(report_dir / 'mcnemar_tests.csv', index=False)
                print(f"   Saved McNemar tests to: {report_dir / 'mcnemar_tests.csv'}")
            else:
                print(f"   ⚠️  No McNemar tests available (predictions not found)")
                mcnemar_df = None
        
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
        with open(report_path, 'w', encoding='utf-8') as f:
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
                
                if sig_df is not None:
                    f.write("METRIC TESTS (Paired t-tests):\n")
                    f.write("-" * 70 + "\n")
                    f.write(sig_df.to_string(index=False))
                
                if mcnemar_df is not None:
                    f.write("\n\n" + "="*70 + "\n")
                    f.write("McNEMAR'S TEST (Classification Agreement):\n")
                    f.write("-" * 70 + "\n")
                    f.write(mcnemar_df.to_string(index=False))
        
        print(f"\n✅ Full report generated in: {report_dir}")
        print(f"\nMain files:")
        print(f"  - comparison_table.csv")
        print(f"  - statistical_tests.csv (paired t-tests)")
        if mcnemar_df is not None:
            print(f"  - mcnemar_tests.csv (classification agreement)")
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

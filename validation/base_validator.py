"""
Base Validator Class

Abstract base class for all validation strategies.
Provides common functionality for:
- Model training and evaluation
- Metrics computation
- Results aggregation
- Statistical testing
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import json
import time


class BaseValidator(ABC):
    """Abstract base class for all validation strategies"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.fold_results = []
        self.validation_summary = {}
        
    @abstractmethod
    def validate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                labels: np.ndarray, sites: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Main validation method - must be implemented by subclasses
        
        Returns:
            Dictionary containing validation results and statistics
        """
        pass
    
    @abstractmethod
    def create_folds(self, fc_matrices: np.ndarray, labels: np.ndarray, 
                    sites: np.ndarray, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/test splits - must be implemented by subclasses
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        pass
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test set
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                logits = model(
                    batch['fc_matrix'],
                    batch['roi_timeseries']
                )
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Collect predictions (using argmax - no threshold adjustment)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        true_labels = np.array(all_labels)
        
        # Compute comprehensive metrics
        metrics = self._compute_metrics(true_labels, predictions, probabilities)
        
        return metrics
    
    def _compute_metrics(self, true_labels: np.ndarray, predictions: np.ndarray, 
                        probabilities: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        # ROC-AUC with comprehensive error handling
        try:
            # Check if both classes are present in true labels
            unique_labels = np.unique(true_labels)
            if len(unique_labels) < 2:
                # Only one class present - cannot compute AUC meaningfully
                auc = np.nan
            else:
                auc = roc_auc_score(true_labels, probabilities[:, 1])
        except (ValueError, IndexError) as e:
            # Edge case: prediction issues
            auc = np.nan
        
        # Confusion matrix and derived metrics
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Sensitivity (Recall) and Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Positive and Negative Predictive Values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator != 0 else 0.0
        
        # Youden's Index
        youden = sensitivity + specificity - 1
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'balanced_accuracy': float(balanced_accuracy),
            'auc': float(auc),
            'ppv': float(ppv),
            'npv': float(npv),
            'mcc': float(mcc),
            'youden': float(youden),
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'true_labels': true_labels.tolist(),
            'n_samples': len(true_labels),
            'n_positive': int(np.sum(true_labels == 1)),
            'n_negative': int(np.sum(true_labels == 0))
        }
        
        return metrics
    
    def aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across folds"""
        
        # Metrics to aggregate
        metrics_to_aggregate = [
            'accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity',
            'balanced_accuracy', 'auc', 'ppv', 'npv', 'mcc', 'youden'
        ]
        
        aggregated = {}
        
        # Compute mean, std, and confidence intervals for each metric
        for metric in metrics_to_aggregate:
            values = [fold['test_metrics'][metric] for fold in fold_results 
                     if 'test_metrics' in fold and metric in fold['test_metrics']]
            
            if values:
                # Filter out NaN values for proper statistics
                valid_values = [v for v in values if not np.isnan(v)]
                
                if valid_values:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    
                    # 95% confidence interval
                    ci_lower, ci_upper = self._compute_confidence_interval(valid_values)
                    
                    aggregated[f'{metric}_mean'] = float(mean_val)
                    aggregated[f'{metric}_std'] = float(std_val)
                    aggregated[f'{metric}_ci_lower'] = float(ci_lower)
                    aggregated[f'{metric}_ci_upper'] = float(ci_upper)
                    aggregated[f'{metric}_values'] = [float(v) for v in values]  # Keep original including NaN
                else:
                    # All values are NaN
                    aggregated[f'{metric}_mean'] = np.nan
                    aggregated[f'{metric}_std'] = np.nan
                    aggregated[f'{metric}_ci_lower'] = np.nan
                    aggregated[f'{metric}_ci_upper'] = np.nan
                    aggregated[f'{metric}_values'] = [float(v) for v in values]
        
        # Overall statistics
        aggregated['n_folds'] = len(fold_results)
        aggregated['total_subjects'] = sum(
            fold['test_metrics']['n_samples'] for fold in fold_results
            if 'test_metrics' in fold
        )
        
        # Aggregate confusion matrices
        total_cm = np.zeros((2, 2), dtype=int)
        for fold in fold_results:
            if 'test_metrics' in fold and 'confusion_matrix' in fold['test_metrics']:
                fold_cm = np.array(fold['test_metrics']['confusion_matrix'])
                if fold_cm.shape == (2, 2):
                    total_cm += fold_cm
        
        aggregated['total_confusion_matrix'] = total_cm.tolist()
        
        # Compute overall metrics from aggregated confusion matrix
        if total_cm.sum() > 0:
            tn, fp, fn, tp = total_cm.ravel()
            
            overall_metrics = {
                'overall_accuracy': (tp + tn) / (tp + tn + fp + fn),
                'overall_sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'overall_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'overall_precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'overall_f1': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            }
            
            aggregated.update({k: float(v) for k, v in overall_metrics.items()})
        
        return aggregated
    
    def _compute_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for a list of values"""
        if len(values) < 2:
            return (0.0, 0.0)
        
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)  # Standard error of the mean
        
        # t-distribution critical value
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        
        margin_error = t_critical * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def statistical_tests(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        # Extract accuracy values for testing
        accuracies = [fold['test_metrics']['accuracy'] for fold in fold_results 
                     if 'test_metrics' in fold]
        
        if len(accuracies) < 3:
            return {'error': 'Insufficient samples for statistical testing'}
        
        # One-sample t-test against chance level (50%)
        chance_level = 0.5
        t_stat, p_value = stats.ttest_1samp(accuracies, chance_level)
        
        # Effect size (Cohen's d)
        cohen_d = (np.mean(accuracies) - chance_level) / np.std(accuracies)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                [acc - chance_level for acc in accuracies]
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_p = None, None
        
        return {
            'one_sample_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'test_description': f'Testing if mean accuracy > {chance_level}'
            },
            'effect_size': {
                'cohen_d': float(cohen_d),
                'interpretation': self._interpret_effect_size(cohen_d)
            },
            'wilcoxon_test': {
                'statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
                'p_value': float(wilcoxon_p) if wilcoxon_p is not None else None,
                'significant': bool(wilcoxon_p < 0.05) if wilcoxon_p is not None else None
            } if wilcoxon_stat is not None else None
        }
    
    def _interpret_effect_size(self, cohen_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def save_results(self, output_path: Path, results: Dict[str, Any]):
        """Save validation results to files"""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated summary
        summary_path = output_path / 'validation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results.get('summary', {}), f, indent=2)
        
        # Save detailed fold results
        if 'fold_results' in results:
            for i, fold_result in enumerate(results['fold_results']):
                fold_path = output_path / f'fold_{i+1}_detailed.json'
                
                # Extract serializable data
                fold_data = {
                    'fold_id': fold_result.get('fold_id', i),
                    'test_metrics': fold_result.get('test_metrics', {}),
                    'training_time': fold_result.get('training_time', 0),
                    'n_train_samples': fold_result.get('n_train_samples', 0),
                    'n_test_samples': fold_result.get('n_test_samples', 0)
                }
                
                # Remove non-serializable embeddings for JSON
                if 'embeddings' in fold_data['test_metrics']:
                    del fold_data['test_metrics']['embeddings']
                
                with open(fold_path, 'w') as f:
                    json.dump(fold_data, f, indent=2)
        
        # Save statistical test results
        if 'statistical_tests' in results:
            stats_path = output_path / 'statistical_tests.json'
            with open(stats_path, 'w') as f:
                json.dump(results['statistical_tests'], f, indent=2)
        
        self.logger.info(f"Validation results saved to {output_path}")
    
    def log_validation_summary(self, results: Dict[str, Any]):
        """Log validation summary to console"""
        
        summary = results.get('summary', {})
        
        self.logger.info("\n" + "="*60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*60)
        
        # Main metrics
        if 'accuracy_mean' in summary:
            self.logger.info(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
            self.logger.info(f"Sensitivity: {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
            self.logger.info(f"Specificity: {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")
            self.logger.info(f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
            self.logger.info(f"F1-Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        
        # Overall metrics
        if 'overall_accuracy' in summary:
            self.logger.info(f"\nOverall Accuracy: {summary['overall_accuracy']:.4f}")
            self.logger.info(f"Overall Sensitivity: {summary['overall_sensitivity']:.4f}")
            self.logger.info(f"Overall Specificity: {summary['overall_specificity']:.4f}")
        
        # Statistical significance
        if 'statistical_tests' in results:
            stats_results = results['statistical_tests']
            if 'one_sample_t_test' in stats_results:
                t_test = stats_results['one_sample_t_test']
                self.logger.info(f"\nStatistical Significance:")
                self.logger.info(f"t-test p-value: {t_test['p_value']:.6f}")
                self.logger.info(f"Significant: {t_test['significant']}")
                
                if 'effect_size' in stats_results:
                    effect = stats_results['effect_size']
                    self.logger.info(f"Effect size (Cohen's d): {effect['cohen_d']:.3f} ({effect['interpretation']})")
        
        self.logger.info("="*60)
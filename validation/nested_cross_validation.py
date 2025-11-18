import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from typing import Dict, List, Any
from .base_validator import BaseValidator
from .loso import LeaveOneSiteOutValidator
from .kfold import StratifiedKFoldValidator


class NestedCrossValidator(BaseValidator):
    """Nested Cross-Validation with hyperparameter optimization"""

    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any],
                 param_grid: Dict[str, List], outer_cv: str = 'loso',
                 inner_cv_folds: int = 3, n_jobs: int = 1):
        super().__init__(model_config, training_config)

        self.param_grid = param_grid
        self.outer_cv = outer_cv.lower()
        self.inner_cv_folds = inner_cv_folds
        self.n_jobs = n_jobs
        self.param_combinations = list(ParameterGrid(param_grid))
        self.validation_type = f"Nested CV ({outer_cv.upper()} outer, {inner_cv_folds}-fold inner)"
        self.logger.info(f"Generated {len(self.param_combinations)} parameter combinations")

    def validate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                 labels: np.ndarray, sites: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.logger.info(f"Starting {self.validation_type}")
        self.logger.info(f"Total subjects: {len(labels)}")
        self.logger.info(f"Parameter combinations: {len(self.param_combinations)}")

        # Select outer CV validator
        if self.outer_cv == 'loso':
            outer_validator = LeaveOneSiteOutValidator(self.model_config, self.training_config)
        elif self.outer_cv == 'kfold':
            outer_validator = StratifiedKFoldValidator(self.model_config, self.training_config,
                                                       n_splits=self.inner_cv_folds)
        else:
            raise ValueError(f"Unsupported outer CV type: {self.outer_cv}")

        outer_folds = outer_validator.create_folds(fc_matrices, labels, sites)
        fold_results = []

        # Outer loop: model evaluation
        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
            self.logger.info(f"\nOuter Fold {fold_idx + 1}/{len(outer_folds)}")
            X_train_fc, X_test_fc = fc_matrices[train_idx], fc_matrices[test_idx]
            X_train_ts, X_test_ts = roi_timeseries[train_idx], roi_timeseries[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            sites_train, sites_test = sites[train_idx], sites[test_idx]

            best_score = -np.inf
            best_params = None

            # Inner loop: hyperparameter tuning
            inner_validator = StratifiedKFoldValidator(self.model_config, self.training_config,
                                                       n_splits=self.inner_cv_folds)
            for params in self.param_combinations:
                # Update model config with current hyperparameters
                model_config_copy = deepcopy(self.model_config)
                model_config_copy.update(params)

                inner_results = inner_validator.validate(X_train_fc, X_train_ts, y_train, sites_train)
                mean_acc = inner_results['summary'].get('accuracy_mean', 0.0)

                if mean_acc > best_score:
                    best_score = mean_acc
                    best_params = params

            self.logger.info(f"Best hyperparameters for outer fold {fold_idx + 1}: {best_params}")

            # Train final model on full training set with best params
            model_config_final = deepcopy(self.model_config)
            model_config_final.update(best_params)
            final_validator = StratifiedKFoldValidator(model_config_final, self.training_config, n_splits=1)
            fold_result = final_validator.validate(X_train_fc, X_train_ts, y_train, sites_train)

            # Evaluate on outer test set
            final_model = GNNSTANHybrid(model_config_final).to(self.device)
            test_loader = final_validator._create_data_loader(
                ADHDDataset(
                    fc_matrices=X_test_fc,
                    roi_timeseries=X_test_ts,
                    labels=y_test,
                    sites=sites_test,
                    augment=False
                ),
                is_train=False
            )
            test_metrics = final_validator.evaluate_model(final_model, test_loader)
            fold_result['test_metrics'] = test_metrics
            fold_results.append(fold_result)

        # Aggregate outer fold results
        summary = self.aggregate_results(fold_results)
        statistical_tests = self.statistical_tests(fold_results)

        results = {
            'validation_type': self.validation_type,
            'fold_results': fold_results,
            'summary': summary,
            'statistical_tests': statistical_tests,
            'model_config': self.model_config,
            'training_config': self.training_config
        }

        self.log_validation_summary(results)
        return results

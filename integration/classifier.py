"""
Pipeline Integration Module

Integrates data loading, model initialization, and validation.
Provides a unified interface to run LOSO, K-Fold, or Nested Cross-Validation
on the ADHD-200 dataset using preprocessed features and ROI timeseries.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from utils.data_loader import ADHDDataLoader
from models.gnn_stan_hybrid import GNNSTANHybrid
from validation.loso import LeaveOneSiteOutValidator
from validation.kfold import StratifiedKFoldValidator, KFoldValidator
from validation import NestedCrossValidator

class ADHDPipeline:
    """Unified ADHD Classification Pipeline"""
    
    def __init__(self,
                 feature_manifest_path: str,
                 demographics_path: str = None,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None,
                 validation_type: str = 'loso',
                 nested_param_grid: Dict[str, list] = None,
                 outer_cv: str = 'loso',
                 inner_cv_folds: int = 3):
        """
        Args:
            feature_manifest_path: Path to CSV listing preprocessed features
            demographics_path: Optional path to demographics CSV
            model_config: Dictionary of model hyperparameters
            training_config: Dictionary of training hyperparameters
            validation_type: 'loso', 'kfold', 'stratified_kfold', or 'nested'
            nested_param_grid: Grid of hyperparameters for nested CV
            outer_cv: Outer CV type for nested CV ('loso' or 'kfold')
            inner_cv_folds: Inner fold count for nested CV
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ADHD pipeline...")
        
        # Load data
        self.data_loader = ADHDDataLoader(feature_manifest_path, demographics_path)
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.validation_type = validation_type.lower()
        self.nested_param_grid = nested_param_grid or {}
        self.outer_cv = outer_cv
        self.inner_cv_folds = inner_cv_folds
        
        # Initialize validator placeholder
        self.validator = None
    
    def prepare_data(self):
        """Load and return all data for training/validation"""
        self.logger.info("Loading data from manifests...")
        data = self.data_loader.load_data_for_training()
        self.logger.info(f"Data shapes - FC: {data['fc_matrices'].shape}, ROI TS: {data['roi_timeseries'].shape}")
        return data
    
    def select_validator(self):
        """Initialize the appropriate validation class"""
        if self.validation_type == 'loso':
            self.validator = LeaveOneSiteOutValidator(self.model_config, self.training_config)
        elif self.validation_type == 'kfold':
            self.validator = KFoldValidator(self.model_config, self.training_config)
        elif self.validation_type == 'stratified_kfold':
            self.validator = StratifiedKFoldValidator(self.model_config, self.training_config)
        elif self.validation_type == 'nested':
            self.validator = NestedCrossValidator(
                model_config=self.model_config,
                training_config=self.training_config,
                param_grid=self.nested_param_grid,
                outer_cv=self.outer_cv,
                inner_cv_folds=self.inner_cv_folds
            )
        else:
            raise ValueError(f"Unknown validation_type: {self.validation_type}")
        
        self.logger.info(f"Validator selected: {self.validator.validation_type}")
    
    def run(self):
        """Execute the pipeline: load data, select validator, and run evaluation"""
        data = self.prepare_data()
        self.select_validator()
        
        results = self.validator.validate(
            fc_matrices=data['fc_matrices'],
            roi_timeseries=data['roi_timeseries'],
            labels=data['labels'],
            sites=data['sites']
        )
        
        return results
    
    def save_results(self, output_dir: str, results: dict):
        """Save validation results to specified directory"""
        output_path = Path(output_dir)
        self.validator.save_results(output_path, results)
        self.logger.info(f"Results saved to {output_dir}")

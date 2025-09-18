"""
Validation package for ADHD Classification Pipeline

Provides different cross-validation strategies:
- Leave-One-Site-Out (LOSO) validation for handling site effects
- K-fold cross-validation for standard evaluation
- Nested cross-validation for hyperparameter optimization
- Stratified validation for handling class imbalance
"""

from .base_validator import BaseValidator
from .loso import LeaveOneSiteOutValidator
from .kfold import KFoldValidator, StratifiedKFoldValidator
from .nested_cv import NestedCrossValidator

__all__ = [
    'BaseValidator',
    'LeaveOneSiteOutValidator', 
    'KFoldValidator',
    'StratifiedKFoldValidator',
    'NestedCrossValidator'
]
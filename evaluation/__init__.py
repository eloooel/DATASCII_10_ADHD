"""
Evaluation package for ADHD GNN-STAN pipeline

This package contains:
- evaluate.py: Evaluation module for trained models
"""

from .evaluate import ADHDModelEvaluator

__all__ = [
    "ADHDModelEvaluator"
]

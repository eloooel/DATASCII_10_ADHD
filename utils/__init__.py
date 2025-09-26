"""
Input module for ADHD preprocessing pipeline
- Handles rs-fMRI data discovery and metadata management
"""

from .data_loader import DataDiscovery
from utils.parallel_runner import run_parallel

__all__ = ["DataDiscovery", "run_parallel"]

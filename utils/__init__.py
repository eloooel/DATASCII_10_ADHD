"""
Input module for ADHD preprocessing pipeline
- Handles rs-fMRI data discovery and metadata management
"""

from .parallel_runner import run_parallel
from .data_loader import DataDiscovery, load_metadata

__all__ = ['run_parallel', 'DataDiscovery', 'load_metadata']

"""
Input module for ADHD preprocessing pipeline
- Handles rs-fMRI data discovery and metadata management
"""

from .data_loader import DataDiscovery

__all__ = ["DataDiscovery"]

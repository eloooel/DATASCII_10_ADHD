"""
Input module for ADHD preprocessing pipeline
- Handles rs-fMRI data discovery and metadata management
"""

from .data_loader import ADHDDataLoader

__all__ = ["ADHDDataLoader"]

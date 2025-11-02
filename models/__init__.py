"""
Models package for ADHD Classification using GNN-STAN Hybrid Architecture

This package contains:
- GNN branch: Graph Neural Networks for functional connectivity processing
- STAN branch: Spatio-Temporal Attention Networks for time series processing  
- Fusion: Cross-modal fusion mechanism
- Hybrid: Complete GNN-STAN hybrid model
"""

from .gnn import EnhancedGNNBranch
from .stan import EnhancedSTANBranch, MultiHeadSpatioTemporalAttention
from .fusion_layer import CrossModalFusion
from .gnn_stan_hybrid import GNNSTANHybrid
from .gnn_stan_hybrid_optimized import MemoryOptimizedGNNSTAN

__all__ = [
    'EnhancedGNNBranch',
    'EnhancedSTANBranch', 
    'MultiHeadSpatioTemporalAttention',
    'CrossModalFusion',
    'GNNSTANHybrid',
    'MemoryOptimizedGNNSTAN'
]


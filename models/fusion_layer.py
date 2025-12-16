"""
Cross-Modal Fusion Module

Implements sophisticated fusion of GNN and STAN embeddings through:
- Cross-attention mechanisms between modalities
- Projection to common embedding space
- Multi-layer fusion networks
- Residual connections and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalFusion(nn.Module):
    """Cross-modal fusion with attention mechanism"""
    
    def __init__(self, gnn_dim: int, stan_dim: int, fusion_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.gnn_dim = gnn_dim
        self.stan_dim = stan_dim
        self.fusion_dim = fusion_dim
        
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.stan_proj = nn.Linear(stan_dim, fusion_dim)
        
        self.cross_attn_gnn = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout)
        self.cross_attn_stan = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout)
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),  # Concatenated + cross-attended
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, gnn_emb: torch.Tensor, stan_emb: torch.Tensor) -> torch.Tensor:
        gnn_proj = self.gnn_proj(gnn_emb).unsqueeze(0)
        stan_proj = self.stan_proj(stan_emb).unsqueeze(0)
        
        # Cross-attention
        gnn_cross, _ = self.cross_attn_gnn(gnn_proj, stan_proj, stan_proj)
        stan_cross, _ = self.cross_attn_stan(stan_proj, gnn_proj, gnn_proj)
        
        gnn_cross = gnn_cross.squeeze(0)
        stan_cross = stan_cross.squeeze(0)
        
        # Concatenate original and cross-attended embeddings
        fused_emb = torch.cat([
            gnn_proj.squeeze(0), stan_proj.squeeze(0),
            gnn_cross, stan_cross
        ], dim=1)
        
        # Apply fusion layers
        fused_output = self.fusion_layers(fused_emb)
        
        return fused_output
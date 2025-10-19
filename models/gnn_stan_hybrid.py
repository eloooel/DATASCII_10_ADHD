"""
Complete GNN-STAN Hybrid Model for ADHD Classification

Integrates:
- Enhanced GNN branch for functional connectivity processing
- Enhanced STAN branch for temporal dynamics processing
- Cross-modal fusion mechanism
- Classification head with interpretability features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .gnn import EnhancedGNNBranch
from .stan import EnhancedSTANBranch
from .fusion_layer import CrossModalFusion


class GNNSTANHybrid(nn.Module):
    """Complete GNN-STAN Hybrid Model for ADHD Classification"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        gnn_config: dict = None,
        stan_config: dict = None,
        fusion_config: dict = None,
        classifier_dropout: float = 0.5
    ):
        super().__init__()
        
        # ROI count is inferred from input data (always 200 for Schaefer-200)
        # We'll set this in forward() on first pass
        self.n_rois = 200  # Fixed for Schaefer-200
        
        # Initialize GNN for FC matrices
        self.gnn_encoder = EnhancedGNNBranch(
            input_dim=self.n_rois,  # Number of ROIs
            **gnn_config if gnn_config else {}
        )
        
        # Initialize STAN for timeseries
        self.stan_encoder = EnhancedSTANBranch(
            input_dim=self.n_rois,  # Number of ROIs
            **stan_config if stan_config else {}
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            gnn_dim=gnn_config.get('hidden_dims', [128, 64, 32])[-1],
            stan_dim=stan_config.get('hidden_dim', 128),
            fusion_dim=fusion_config.get('fusion_dim', 128),
            dropout=fusion_config.get('dropout', 0.3)
        )
        
        # Classification head
        fusion_dim = fusion_config.get('fusion_dim', 128) if fusion_config else 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),  # Fixed: was config.get()
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, num_classes)  # Fixed: use num_classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hybrid model
        
        Args:
            fc_matrix: Functional connectivity matrix (batch, n_rois, n_rois)
            roi_timeseries: ROI time series (batch, n_timepoints, n_rois)
            
        Returns:
            logits for classification
        """
        batch_size = fc_matrix.shape[0]
        
        # Create edge_index from FC matrix (fully connected graph)
        edge_index = self._create_edge_index_from_fc(fc_matrix).to(fc_matrix.device)
        
        # Create batch indices for graph pooling
        batch = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        
        # Extract node features from FC matrix
        node_features = self._extract_node_features(fc_matrix)
        
        # GNN branch forward pass
        gnn_embedding = self.gnn_encoder(node_features, edge_index, batch)
        
        # STAN branch forward pass
        stan_embedding = self.stan_encoder(roi_timeseries)
        
        # Cross-modal fusion
        fused_embedding = self.fusion(gnn_embedding, stan_embedding)
        
        # Classification
        logits = self.classifier(fused_embedding)
        
        return logits

    def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Create edge index from FC matrix (keep strong connections)"""
        batch_size, n_rois, _ = fc_matrix.shape
        
        # Threshold to keep only strong connections (e.g., top 20%)
        threshold = torch.quantile(torch.abs(fc_matrix), 0.8, dim=(1,2), keepdim=True)
        mask = torch.abs(fc_matrix) > threshold
        
        # Get edge indices for first sample (assume same structure for batch)
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            # Add batch offset
            sources = sources + b * n_rois
            targets = targets + b * n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        
        edge_index = torch.cat(edge_index_list, dim=1)
        return edge_index
    
    def _extract_node_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Extract node features from functional connectivity matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        
        # Node features: degree centrality, clustering coefficient, betweenness (simplified)
        node_features = torch.zeros(batch_size * n_rois, 4, device=fc_matrix.device)  # 4 features per node
        
        for i in range(batch_size):
            fc = fc_matrix[i]
            
            # Degree centrality (sum of absolute connections)
            degree = torch.sum(torch.abs(fc), dim=1)
            
            # Clustering coefficient (simplified)
            clustering = torch.diagonal(torch.matmul(torch.matmul(fc, fc), fc)) / (degree + 1e-8)
            
            # Eigenvector centrality (first eigenvector component)
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(fc)
                eigen_centrality = torch.abs(eigenvecs[:, -1])  # Principal eigenvector
            except:
                eigen_centrality = degree / (torch.sum(degree) + 1e-8)
            
            # Local efficiency (simplified)
            local_eff = torch.mean(torch.abs(fc), dim=1)
            
            # Stack features
            start_idx = i * n_rois
            end_idx = (i + 1) * n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        return node_features

    def get_attention_weights(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor,
                             edge_index: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability"""
        with torch.no_grad():
            outputs = self.forward(fc_matrix, roi_timeseries, edge_index, batch)
            
            return {
                'gnn_attention': outputs['gnn_attention'],
                'stan_attention': outputs['stan_attention']
            }
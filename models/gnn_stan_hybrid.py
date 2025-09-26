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
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.n_rois = config.get('n_rois', 200)
        self.n_classes = config.get('n_classes', 2)
        
        # Extract architecture parameters
        gnn_config = config.get('gnn', {})
        stan_config = config.get('stan', {})
        fusion_config = config.get('fusion', {})
        
        # Initialize branches
        self.gnn_branch = EnhancedGNNBranch(
            input_dim=4,  # Node features: degree, clustering, eigen_centrality, local_eff
            hidden_dims=gnn_config.get('hidden_dims', [128, 64, 32]),
            dropout=gnn_config.get('dropout', 0.3),
            pool_ratios=gnn_config.get('pool_ratios', [0.8, 0.6])
        )
        
        self.stan_branch = EnhancedSTANBranch(
            input_dim=self.n_rois,
            hidden_dim=stan_config.get('hidden_dim', 128),
            num_layers=stan_config.get('num_layers', 2),
            dropout=stan_config.get('dropout', 0.3)
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            gnn_dim=gnn_config.get('hidden_dims', [128, 64, 32])[-1],
            stan_dim=stan_config.get('hidden_dim', 128),
            fusion_dim=fusion_config.get('fusion_dim', 128),
            dropout=fusion_config.get('dropout', 0.3)
        )
        
        # Classification head
        fusion_dim = fusion_config.get('fusion_dim', 128)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.5)),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, self.n_classes)
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
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor, 
                edge_index: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hybrid model
        
        Args:
            fc_matrix: Functional connectivity matrix (batch, n_rois, n_rois)
            roi_timeseries: ROI time series (batch, n_rois, n_timepoints)
            edge_index: Graph edge indices (2, n_edges)
            batch: Batch indices for graph pooling
            
        Returns:
            Dictionary containing logits, probabilities, and attention maps
        """
        
        # Extract node features from FC matrix (diagonal + upper triangle statistics)
        batch_size = fc_matrix.shape[0]
        node_features = self._extract_node_features(fc_matrix)
        
        # GNN branch forward pass
        gnn_embedding, gnn_attention = self.gnn_branch(node_features, edge_index, batch)
        
        # STAN branch forward pass
        stan_embedding, stan_attention = self.stan_branch(roi_timeseries)
        
        # Cross-modal fusion
        fused_embedding = self.fusion(gnn_embedding, stan_embedding)
        
        # Classification
        logits = self.classifier(fused_embedding)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'gnn_embedding': gnn_embedding,
            'stan_embedding': stan_embedding,
            'fused_embedding': fused_embedding,
            'gnn_attention': gnn_attention,
            'stan_attention': stan_attention
        }
    
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
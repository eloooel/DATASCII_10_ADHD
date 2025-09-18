"""
Enhanced GNN Branch for functional connectivity processing

Implements Graph Neural Networks with:
- Graph Attention Networks (GAT) for better node representation
- Hierarchical pooling for multi-scale graph features
- Skip connections for better gradient flow
- Attention mechanisms for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, Dict, Any


class EnhancedGNNBranch(nn.Module):
    """Enhanced GNN branch with hierarchical pooling and attention"""
    
    def __init__(self, input_dim: int = 200, hidden_dims: list = [128, 64, 32], 
                 dropout: float = 0.3, pool_ratios: list = [0.8, 0.6]):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Graph convolution layers with skip connections
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            # Use Graph Attention Networks for better performance
            self.gnn_layers.append(GATConv(dims[i], dims[i+1], heads=4, concat=False, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(dims[i+1]))
            
            # Skip connection for residual learning
            if dims[i] == dims[i+1]:
                self.skip_connections.append(nn.Identity())
            else:
                self.skip_connections.append(nn.Linear(dims[i], dims[i+1]))
        
        # Hierarchical pooling layers
        self.pool_layers = nn.ModuleList([
            TopKPooling(hidden_dims[0], ratio=pool_ratios[0]),
            TopKPooling(hidden_dims[1], ratio=pool_ratios[1])
        ])
        
        # Final embedding layer
        self.final_conv = GCNConv(hidden_dims[-1], hidden_dims[-1])
        self.graph_norm = nn.BatchNorm1d(hidden_dims[-1])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attention_maps = {}
        
        # Initial node features (using FC matrix as node features)
        h = x
        
        # Apply GNN layers with residual connections
        for i, (gnn, bn, skip) in enumerate(zip(self.gnn_layers, self.batch_norms, self.skip_connections)):
            h_residual = skip(h)
            
            if isinstance(gnn, GATConv):
                h, attn = gnn(h, edge_index, return_attention_weights=True)
                attention_maps[f'gnn_layer_{i}'] = attn
            else:
                h = gnn(h, edge_index)
            
            h = bn(h)
            h = F.relu(h + h_residual)  # Residual connection
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Apply hierarchical pooling
            if i < len(self.pool_layers):
                h, edge_index, _, batch, _, _ = self.pool_layers[i](h, edge_index, batch=batch)
        
        # Final convolution and global pooling
        h = self.final_conv(h, edge_index)
        h = self.graph_norm(h)
        h = F.relu(h)
        
        # Global graph representation
        graph_embedding = global_mean_pool(h, batch)
        
        return graph_embedding, attention_maps
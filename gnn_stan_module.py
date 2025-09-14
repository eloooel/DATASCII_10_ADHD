import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data, Batch
import math
from typing import Tuple, Optional, Dict, Any

class MultiHeadSpatioTemporalAttention(nn.Module):
    """Multi-head spatio-temporal attention mechanism for STAN branch"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + residual), attn_weights

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

class EnhancedSTANBranch(nn.Module):
    """Enhanced STAN branch with bidirectional processing and attention"""
    
    def __init__(self, input_dim: int = 200, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM for temporal encoding
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head spatio-temporal attention
        self.attention = MultiHeadSpatioTemporalAttention(
            d_model=hidden_dim * 2,  # bidirectional
            n_heads=8,
            dropout=dropout
        )
        
        # Temporal convolution for local pattern extraction
        self.temp_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        self.temp_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_rois, n_timepoints = x.shape
        
        # Reshape for LSTM: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, time, rois)
        
        # Bidirectional LSTM encoding
        lstm_out, _ = self.temporal_encoder(x)  # (batch, time, 2*hidden)
        
        # Multi-head attention over temporal dimension
        attended_out, attn_weights = self.attention(lstm_out)
        
        # Temporal convolution for local patterns
        conv_input = attended_out.transpose(1, 2)  # (batch, features, time)
        conv_out = self.temp_conv(conv_input)
        conv_out = self.temp_norm(conv_out)
        conv_out = F.relu(conv_out)
        
        # Global temporal pooling
        temporal_embedding = torch.mean(conv_out, dim=2)  # (batch, hidden)
        temporal_embedding = self.dropout(temporal_embedding)
        
        return temporal_embedding, attn_weights

class CrossModalFusion(nn.Module):
    """Cross-modal fusion with attention mechanism"""
    
    def __init__(self, gnn_dim: int, stan_dim: int, fusion_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.gnn_dim = gnn_dim
        self.stan_dim = stan_dim
        self.fusion_dim = fusion_dim
        
        # Project embeddings to common dimension
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.stan_proj = nn.Linear(stan_dim, fusion_dim)
        
        # Cross-attention mechanism
        self.cross_attn_gnn = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout)
        self.cross_attn_stan = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout)
        
        # Fusion layers
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
        # Project to common dimension
        gnn_proj = self.gnn_proj(gnn_emb).unsqueeze(0)  # (1, batch, dim)
        stan_proj = self.stan_proj(stan_emb).unsqueeze(0)  # (1, batch, dim)
        
        # Cross-attention
        gnn_cross, _ = self.cross_attn_gnn(gnn_proj, stan_proj, stan_proj)
        stan_cross, _ = self.cross_attn_stan(stan_proj, gnn_proj, gnn_proj)
        
        # Remove sequence dimension
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
            input_dim=self.n_rois,
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
        node_features = torch.zeros(batch_size * n_rois, 4)  # 4 features per node
        
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
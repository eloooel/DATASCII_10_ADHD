"""
Memory-Optimized GNN-STAN Hybrid Model with Advanced Optimization Techniques

Integrates all optimization techniques:
1. Active Gradient Offloading
2. Activation Swapping
3. Hybrid Attention Blocks
4. Dynamic Gating
5. Hot/Cold Neuron Preloading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .gnn import EnhancedGNNBranch
from .stan import EnhancedSTANBranch
from .fusion_layer import CrossModalFusion
from optimization.advanced_memory_optimization import (
    HybridAttentionBlock,
    DynamicGatingModule,
    MemoryOptimizationManager
)


class MemoryOptimizedGNNSTAN(nn.Module):
    """
    Memory-Optimized GNN-STAN Hybrid with advanced optimization techniques
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        gnn_config: dict = None,
        stan_config: dict = None,
        fusion_config: dict = None,
        classifier_dropout: float = 0.5,
        # Optimization settings
        use_hybrid_attention: bool = True,
        use_dynamic_gating: bool = True,
        attention_chunk_size: int = 64,
        attention_sparsity: float = 0.2,
        gating_threshold: float = 0.5
    ):
        super().__init__()
        
        self.n_rois = 200  # Schaefer-200
        self.use_hybrid_attention = use_hybrid_attention
        self.use_dynamic_gating = use_dynamic_gating
        
        # GNN branch with memory optimization
        gnn_cfg = gnn_config or {}
        # Node feature dimension is 4 (from _extract_node_features)
        self.gnn_encoder = EnhancedGNNBranch(
            input_dim=4,  # Node features: degree, clustering, eigenvector centrality, local efficiency
            **gnn_cfg
        )
        
        # STAN branch with hybrid attention
        stan_cfg = stan_config or {}
        stan_hidden = stan_cfg.get('hidden_dim', 128)
        
        if use_hybrid_attention:
            # Replace standard attention with Hybrid Attention Blocks
            self.stan_attention = HybridAttentionBlock(
                embed_dim=self.n_rois,
                num_heads=num_heads,
                dropout=dropout,
                chunk_size=attention_chunk_size,
                sparsity_ratio=attention_sparsity,
                use_low_rank=True
            )
        else:
            self.stan_attention = None
        
        # STAN temporal encoder (LSTM)
        self.stan_lstm = nn.LSTM(
            input_size=self.n_rois,
            hidden_size=stan_hidden,
            num_layers=stan_cfg.get('num_layers', 2),
            dropout=dropout if stan_cfg.get('num_layers', 2) > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection after bidirectional LSTM
        self.stan_proj = nn.Linear(stan_hidden * 2, stan_hidden)
        
        # Dynamic gating for GNN and STAN branches
        if use_dynamic_gating:
            self.gating_module = DynamicGatingModule(
                input_dim=hidden_dim,
                num_layers=2,  # GNN and STAN branches
                gating_threshold=gating_threshold,
                use_learnable_gates=True
            )
        else:
            self.gating_module = None
        
        # Cross-modal fusion
        fusion_cfg = fusion_config or {}
        gnn_out_dim = gnn_cfg.get('hidden_dims', [128, 64, 32])[-1]
        
        self.fusion = CrossModalFusion(
            gnn_dim=gnn_out_dim,
            stan_dim=stan_hidden,
            fusion_dim=fusion_cfg.get('fusion_dim', 128),
            dropout=fusion_cfg.get('dropout', 0.3)
        )
        
        # Lightweight classifier
        fusion_dim = fusion_cfg.get('fusion_dim', 128)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Optimization manager (will be set by training loop)
        self.optimization_manager = None
    
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
    
    def setup_optimization(
        self,
        device: str = 'cuda',
        enable_gradient_offload: bool = True,
        enable_activation_swap: bool = True,
        enable_neuron_preload: bool = False,
        **kwargs
    ):
        """Setup memory optimization manager"""
        self.optimization_manager = MemoryOptimizationManager(
            model=self,
            device=device,
            enable_gradient_offload=enable_gradient_offload,
            enable_activation_swap=enable_activation_swap,
            enable_neuron_preload=enable_neuron_preload,
            enable_dynamic_gating=self.use_dynamic_gating,
            **kwargs
        )
        self.optimization_manager.setup()
        return self.optimization_manager
    
    def forward(
        self,
        fc_matrix: torch.Tensor,
        roi_timeseries: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-optimized forward pass
        
        Args:
            fc_matrix: Functional connectivity (batch, n_rois, n_rois)
            roi_timeseries: ROI time series (batch, n_timepoints, n_rois)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits and optional attention weights
        """
        batch_size = fc_matrix.shape[0]
        
        # Store activations for swapping if enabled
        if self.optimization_manager and self.optimization_manager.activation_manager:
            act_mgr = self.optimization_manager.activation_manager
        else:
            act_mgr = None
        
        # ==================== GNN Branch ====================
        # Create edge index from FC matrix
        edge_index = self._create_edge_index_from_fc(fc_matrix).to(fc_matrix.device)
        
        # Create batch indices
        batch_idx = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        
        # Extract node features
        node_features = self._extract_node_features(fc_matrix)
        
        if act_mgr:
            act_mgr.store_activation('node_features', node_features, keep_on_gpu=True)
        
        # GNN forward with optional dynamic gating
        if self.use_dynamic_gating and self.gating_module:
            gnn_embedding = self.gating_module.forward(
                node_features,
                lambda x: self.gnn_encoder(x, edge_index, batch_idx),
                layer_idx=0
            )
        else:
            gnn_embedding = self.gnn_encoder(node_features, edge_index, batch_idx)
        
        if act_mgr:
            act_mgr.store_activation('gnn_embedding', gnn_embedding, keep_on_gpu=False)
        
        # ==================== STAN Branch ====================
        # Apply hybrid attention if enabled
        if self.use_hybrid_attention and self.stan_attention:
            result = self.stan_attention(
                roi_timeseries,
                return_attention=return_attention
            )
            # Handle return value - can be tensor or tuple
            if isinstance(result, tuple):
                roi_timeseries_attn, attn_weights = result
            else:
                roi_timeseries_attn = result
                attn_weights = None
        else:
            roi_timeseries_attn = roi_timeseries
            attn_weights = None
        
        if act_mgr:
            act_mgr.store_activation('roi_timeseries_attn', roi_timeseries_attn, keep_on_gpu=False)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.stan_lstm(roi_timeseries_attn)
        
        # Use final hidden state (both directions)
        stan_embedding = torch.cat([h_n[-2], h_n[-1]], dim=1)
        stan_embedding = self.stan_proj(stan_embedding)
        
        if act_mgr:
            act_mgr.store_activation('stan_embedding', stan_embedding, keep_on_gpu=True)
        
        # Retrieve GNN embedding if it was swapped
        if act_mgr:
            gnn_embedding = act_mgr.get_activation('gnn_embedding', device=fc_matrix.device)
        
        # ==================== Fusion & Classification ====================
        # Cross-modal fusion with optional dynamic gating
        if self.use_dynamic_gating and self.gating_module:
            fused_embedding = self.gating_module.forward(
                torch.cat([gnn_embedding, stan_embedding], dim=1),
                lambda x: self.fusion(
                    x[:, :gnn_embedding.size(1)],
                    x[:, gnn_embedding.size(1):]
                ),
                layer_idx=1
            )
        else:
            fused_embedding = self.fusion(gnn_embedding, stan_embedding)
        
        # Classification
        logits = self.classifier(fused_embedding)
        
        # Prepare output
        output = {'logits': logits}
        
        if return_attention and attn_weights is not None:
            output['attention_weights'] = attn_weights
        
        if self.use_dynamic_gating and self.gating_module:
            output['gate_skip_rates'] = self.gating_module.get_skip_rate()
        
        return output
    
    def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Create edge index from FC matrix (sparse connections)"""
        batch_size, n_rois, _ = fc_matrix.shape
        
        # Threshold to keep only strong connections (top 20%)
        # Flatten for quantile calculation (PyTorch 2.6.0 compatibility)
        fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
        threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
        mask = torch.abs(fc_matrix) > threshold
        
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            sources = sources + b * n_rois
            targets = targets + b * n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        
        edge_index = torch.cat(edge_index_list, dim=1)
        return edge_index
    
    def _extract_node_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Extract graph features from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        node_features = torch.zeros(batch_size * n_rois, 4, device=fc_matrix.device)
        
        for i in range(batch_size):
            fc = fc_matrix[i]
            
            # Degree centrality
            degree = torch.sum(torch.abs(fc), dim=1)
            
            # Clustering coefficient (simplified)
            clustering = torch.diagonal(torch.matmul(torch.matmul(fc, fc), fc)) / (degree + 1e-8)
            
            # Eigenvector centrality (approximation)
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(fc)
                eigen_centrality = torch.abs(eigenvecs[:, -1])
            except:
                eigen_centrality = degree / (torch.sum(degree) + 1e-8)
            
            # Local efficiency
            local_eff = torch.mean(torch.abs(fc), dim=1)
            
            # Stack features
            start_idx = i * n_rois
            end_idx = (i + 1) * n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        return node_features
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if self.optimization_manager:
            return self.optimization_manager.get_memory_stats()
        
        stats = {}
        if torch.cuda.is_available():
            stats['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
        return stats
    
    def print_optimization_stats(self):
        """Print optimization statistics"""
        print("\n" + "="*60)
        print("MEMORY OPTIMIZATION STATISTICS")
        print("="*60)
        
        # Memory stats
        stats = self.get_memory_stats()
        if stats:
            print(f"\nMemory Usage:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Gating stats
        if self.use_dynamic_gating and self.gating_module:
            skip_rates = self.gating_module.get_skip_rate()
            if skip_rates:
                print(f"\nDynamic Gating Skip Rates:")
                for layer_idx, rate in skip_rates.items():
                    print(f"  Layer {layer_idx}: {rate*100:.1f}% skipped")
        
        print("="*60 + "\n")

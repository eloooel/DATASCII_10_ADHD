"""
Enhanced STAN Branch for temporal processing of ROI time series

Implements Spatio-Temporal Attention Networks with:
- Multi-head attention mechanisms
- Bidirectional LSTM for temporal encoding
- Temporal convolution for local pattern extraction
- Layer normalization and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
"""
Advanced Memory Optimization Techniques for GNN-STAN Hybrid Model

Implements:
1. Active Gradient Offloading (AGO)
2. Holistic Traffic-Aware Activation Swapping (HTAAS)
3. Hybrid Attention Blocks (HAB) for VRAM-Efficient Attention
4. "Hot" and "Cold" Neuron Preloading
5. Dynamic Gating of Attention/Graph Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from collections import deque
import threading
import gc


# ============================================================================
# 1. Active Gradient Offloading (AGO)
# ============================================================================

class ActiveGradientOffloader:
    """
    Offload gradients to CPU during backward pass to reduce VRAM usage.
    Automatically detects which gradients to offload based on memory pressure.
    """
    
    def __init__(self, model: nn.Module, offload_threshold: float = 0.7, device: str = 'cuda'):
        """
        Args:
            model: PyTorch model
            offload_threshold: Fraction of VRAM usage to trigger offloading (0.0-1.0)
            device: Device for primary compute
        """
        self.model = model
        self.offload_threshold = offload_threshold
        self.device = device
        self.cpu_device = torch.device('cpu')
        
        # Track which parameters are offloaded
        self.offloaded_grads = {}
        self.grad_hooks = []
        
        # Memory statistics
        self.peak_memory = 0
        self.current_memory = 0
        
    def get_memory_usage(self) -> float:
        """Get current VRAM usage as fraction of total"""
        if self.device == 'cpu':
            return 0.0
        
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return allocated
    
    def should_offload(self) -> bool:
        """Check if we should offload gradients based on memory pressure"""
        usage = self.get_memory_usage()
        return usage > self.offload_threshold
    
    def offload_gradient(self, param_name: str, grad: torch.Tensor):
        """Offload gradient to CPU"""
        if grad is not None and grad.is_cuda:
            self.offloaded_grads[param_name] = grad.cpu()
            return True
        return False
    
    def restore_gradient(self, param_name: str, param: nn.Parameter):
        """Restore gradient from CPU to GPU"""
        if param_name in self.offloaded_grads:
            param.grad = self.offloaded_grads[param_name].to(self.device)
            del self.offloaded_grads[param_name]
    
    def register_hooks(self):
        """Register backward hooks for gradient offloading"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._grad_hook(n, grad)
                )
                self.grad_hooks.append(hook)
    
    def _grad_hook(self, param_name: str, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """Hook called during backward pass"""
        if self.should_offload():
            self.offload_gradient(param_name, grad)
            # Return None to clear GPU gradient
            return None
        return grad
    
    def restore_all_gradients(self):
        """Restore all offloaded gradients before optimizer step"""
        for name, param in self.model.named_parameters():
            self.restore_gradient(name, param)
    
    def clear(self):
        """Clear offloaded gradients and remove hooks"""
        self.offloaded_grads.clear()
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks.clear()


# ============================================================================
# 2. Holistic Traffic-Aware Activation Swapping (HTAAS)
# ============================================================================

class ActivationSwapManager:
    """
    Intelligently swap activations between GPU and CPU based on:
    - Access frequency (traffic awareness)
    - Layer dependencies
    - Memory pressure
    """
    
    def __init__(self, max_gpu_activations: int = 10, prefetch_window: int = 2):
        """
        Args:
            max_gpu_activations: Maximum activations to keep on GPU
            prefetch_window: Number of layers ahead to prefetch
        """
        self.max_gpu_activations = max_gpu_activations
        self.prefetch_window = prefetch_window
        
        # Activation storage
        self.gpu_activations = {}  # layer_id -> activation
        self.cpu_activations = {}  # layer_id -> activation
        
        # Access tracking for LRU-like policy
        self.access_counts = {}  # layer_id -> count
        self.access_order = deque(maxlen=max_gpu_activations)
        
        # Prefetch queue
        self.prefetch_queue = deque()
        self.prefetch_thread = None
        
    def store_activation(self, layer_id: str, activation: torch.Tensor, keep_on_gpu: bool = False):
        """Store activation with intelligent placement"""
        self.access_counts[layer_id] = self.access_counts.get(layer_id, 0) + 1
        
        if keep_on_gpu or len(self.gpu_activations) < self.max_gpu_activations:
            # Keep on GPU
            self.gpu_activations[layer_id] = activation
            self.access_order.append(layer_id)
        else:
            # Swap to CPU
            self.cpu_activations[layer_id] = activation.cpu()
            
            # Evict least recently used from GPU if over capacity
            if len(self.gpu_activations) >= self.max_gpu_activations:
                lru_layer = self.access_order.popleft()
                if lru_layer in self.gpu_activations:
                    self.cpu_activations[lru_layer] = self.gpu_activations[lru_layer].cpu()
                    del self.gpu_activations[lru_layer]
    
    def get_activation(self, layer_id: str, device: str = 'cuda') -> Optional[torch.Tensor]:
        """Retrieve activation, moving to GPU if needed"""
        # Update access tracking
        self.access_counts[layer_id] = self.access_counts.get(layer_id, 0) + 1
        
        # Check GPU first
        if layer_id in self.gpu_activations:
            if layer_id not in self.access_order:
                self.access_order.append(layer_id)
            return self.gpu_activations[layer_id]
        
        # Check CPU and move to GPU
        if layer_id in self.cpu_activations:
            activation = self.cpu_activations[layer_id].to(device)
            del self.cpu_activations[layer_id]
            self.store_activation(layer_id, activation, keep_on_gpu=True)
            return activation
        
        return None
    
    def prefetch_activations(self, layer_ids: List[str], device: str = 'cuda'):
        """Asynchronously prefetch activations to GPU"""
        for layer_id in layer_ids:
            if layer_id in self.cpu_activations and layer_id not in self.gpu_activations:
                # Move to GPU asynchronously
                activation = self.cpu_activations[layer_id].to(device, non_blocking=True)
                self.gpu_activations[layer_id] = activation
                del self.cpu_activations[layer_id]
    
    def clear(self):
        """Clear all stored activations"""
        self.gpu_activations.clear()
        self.cpu_activations.clear()
        self.access_counts.clear()
        self.access_order.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 3. Hybrid Attention Block (HAB) for VRAM-Efficient Attention
# ============================================================================

class HybridAttentionBlock(nn.Module):
    """
    Memory-efficient attention using:
    - Sparse attention patterns
    - Low-rank approximations
    - Chunked computation
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        chunk_size: int = 64,
        sparsity_ratio: float = 0.2,
        use_low_rank: bool = True,
        rank_ratio: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.sparsity_ratio = sparsity_ratio
        self.use_low_rank = use_low_rank
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Low-rank projections if enabled
        if use_low_rank:
            self.rank = max(1, int(embed_dim * rank_ratio))
            self.q_proj = nn.Sequential(
                nn.Linear(embed_dim, self.rank),
                nn.Linear(self.rank, embed_dim)
            )
            self.k_proj = nn.Sequential(
                nn.Linear(embed_dim, self.rank),
                nn.Linear(self.rank, embed_dim)
            )
            self.v_proj = nn.Sequential(
                nn.Linear(embed_dim, self.rank),
                nn.Linear(self.rank, embed_dim)
            )
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tensor
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Project Q, K, V with low-rank approximation
        Q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Chunked attention computation to save memory
        if seq_len > self.chunk_size:
            output, attn_weights = self._chunked_attention(Q, K, V, mask)
        else:
            output, attn_weights = self._sparse_attention(Q, K, V, mask)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        if return_attention:
            return output, attn_weights
        return output
    
    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse attention (keep top-k connections)"""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply sparsity: keep only top-k connections
        k = max(1, int(scores.size(-1) * self.sparsity_ratio))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(scores)
        sparse_mask.scatter_(-1, topk_indices, 1.0)
        
        # Apply sparse mask
        scores = scores * sparse_mask
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def _chunked_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention in chunks to reduce memory"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Split into chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        output_chunks = []
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, seq_len)
            
            Q_chunk = Q[:, :, start:end, :]
            
            # Compute attention for this chunk
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / (head_dim ** 0.5)
            
            if mask is not None:
                mask_chunk = mask[:, :, start:end, :]
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output_chunk = torch.matmul(attn_weights, V)
            output_chunks.append(output_chunk)
            
            # Clear intermediate tensors
            del scores, attn_weights
        
        # Concatenate chunks
        output = torch.cat(output_chunks, dim=2)
        
        return output, None  # Don't return attention weights for chunked mode


# ============================================================================
# 4. "Hot" and "Cold" Neuron Preloading
# ============================================================================

class NeuronPreloader:
    """
    Analyze neuron activation patterns and preload frequently used neurons
    for faster inference.
    """
    
    def __init__(self, model: nn.Module, warmup_steps: int = 100):
        """
        Args:
            model: PyTorch model
            warmup_steps: Number of forward passes to analyze activation patterns
        """
        self.model = model
        self.warmup_steps = warmup_steps
        
        # Activation statistics
        self.activation_stats = {}  # layer_name -> {'mean': ..., 'std': ..., 'frequency': ...}
        self.hot_neurons = {}  # layer_name -> indices of hot neurons
        self.cold_neurons = {}  # layer_name -> indices of cold neurons
        
        # Tracking
        self.warmup_count = 0
        self.is_warmed_up = False
        
    def register_hooks(self):
        """Register forward hooks to track activations"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                module.register_forward_hook(
                    lambda mod, inp, out, n=name: self._activation_hook(n, out)
                )
    
    def _activation_hook(self, layer_name: str, activation: torch.Tensor):
        """Track activation statistics during warmup"""
        if self.warmup_count >= self.warmup_steps:
            return
        
        with torch.no_grad():
            # Flatten activation
            act_flat = activation.detach().abs().mean(dim=0)
            
            if layer_name not in self.activation_stats:
                self.activation_stats[layer_name] = {
                    'sum': act_flat,
                    'sum_sq': act_flat ** 2,
                    'count': 1
                }
            else:
                stats = self.activation_stats[layer_name]
                stats['sum'] += act_flat
                stats['sum_sq'] += act_flat ** 2
                stats['count'] += 1
    
    def finalize_warmup(self, hot_percentile: float = 0.8):
        """Analyze warmup data and identify hot/cold neurons"""
        self.is_warmed_up = True
        
        for layer_name, stats in self.activation_stats.items():
            # Compute mean and std
            count = stats['count']
            mean = stats['sum'] / count
            std = torch.sqrt(stats['sum_sq'] / count - mean ** 2)
            
            # Identify hot neurons (high activation)
            threshold_hot = torch.quantile(mean, hot_percentile)
            self.hot_neurons[layer_name] = (mean > threshold_hot).nonzero(as_tuple=True)[0]
            
            # Identify cold neurons (low activation)
            threshold_cold = torch.quantile(mean, 1 - hot_percentile)
            self.cold_neurons[layer_name] = (mean < threshold_cold).nonzero(as_tuple=True)[0]
            
            print(f"Layer {layer_name}: {len(self.hot_neurons[layer_name])} hot neurons, "
                  f"{len(self.cold_neurons[layer_name])} cold neurons")
    
    def get_pruning_mask(self, layer_name: str, prune_cold: bool = True) -> Optional[torch.Tensor]:
        """Get mask to prune cold neurons during inference"""
        if layer_name not in self.cold_neurons:
            return None
        
        if prune_cold:
            # Create mask that zeros out cold neurons
            cold_indices = self.cold_neurons[layer_name]
            # Implementation depends on layer type
            return cold_indices
        
        return None


# ============================================================================
# 5. Dynamic Gating for Attention/Graph Layers
# ============================================================================

class DynamicGatingModule(nn.Module):
    """
    Dynamically gate attention and graph layers based on input complexity.
    Skips expensive computations when input is "easy".
    """
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        gating_threshold: float = 0.5,
        use_learnable_gates: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.gating_threshold = gating_threshold
        self.use_learnable_gates = use_learnable_gates
        
        if use_learnable_gates:
            # Learnable gating network
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, num_layers),
                nn.Sigmoid()
            )
        
        # Track gating statistics
        self.gate_stats = {'total': 0, 'skipped': [0] * num_layers}
        
    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating decisions for each layer
        
        Args:
            x: Input tensor (batch, ...)
            
        Returns:
            gates: Binary tensor (batch, num_layers) indicating which layers to execute
        """
        if self.use_learnable_gates:
            # Use learnable gating network
            # Reshape to (batch, -1) for pooling
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_pooled = x_flat.mean(dim=1, keepdim=True)  # (batch, 1)
            # Expand to input_dim
            x_pooled = x_pooled.expand(batch_size, self.input_dim)  # (batch, input_dim)
            gate_probs = self.gate_network(x_pooled)  # (batch, num_layers)
            gates = (gate_probs > self.gating_threshold).float()
        else:
            # Simple heuristic: based on input variance
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_var = x_flat.var(dim=1)  # (batch,)
            threshold = x_var.median()
            # High variance = complex input = execute all layers
            gates = (x_var.unsqueeze(-1) > threshold).float().expand(-1, self.num_layers)
        
        return gates
    
    def forward(self, x: torch.Tensor, layer_fn: callable, layer_idx: int) -> torch.Tensor:
        """
        Apply gating to a layer
        
        Args:
            x: Input tensor
            layer_fn: Function that executes the layer
            layer_idx: Index of the layer
            
        Returns:
            output: Layer output (or input if skipped)
        """
        gates = self.compute_gates(x)
        
        # Track statistics
        self.gate_stats['total'] += x.size(0)
        skipped = (gates[:, layer_idx] == 0).sum().item()
        self.gate_stats['skipped'][layer_idx] += skipped
        
        # Execute layer only for samples with gate=1
        if gates[:, layer_idx].sum() == 0:
            # Skip entire batch for this layer
            return x
        elif gates[:, layer_idx].sum() == gates.size(0):
            # Execute for entire batch
            return layer_fn(x)
        else:
            # Mixed batch: execute only for gated samples
            output = x.clone()
            mask = gates[:, layer_idx].bool()
            output[mask] = layer_fn(x[mask])
            return output
    
    def get_skip_rate(self) -> Dict[int, float]:
        """Get skip rate for each layer"""
        if self.gate_stats['total'] == 0:
            return {}
        
        skip_rates = {}
        for layer_idx, skipped in enumerate(self.gate_stats['skipped']):
            skip_rates[layer_idx] = skipped / self.gate_stats['total']
        
        return skip_rates
    
    def reset_stats(self):
        """Reset gating statistics"""
        self.gate_stats = {'total': 0, 'skipped': [0] * self.num_layers}


# ============================================================================
# Unified Optimization Manager
# ============================================================================

class MemoryOptimizationManager:
    """
    Unified manager for all memory optimization techniques
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        enable_gradient_offload: bool = True,
        enable_activation_swap: bool = True,
        enable_neuron_preload: bool = True,
        enable_dynamic_gating: bool = True,
        gradient_offload_threshold: float = 0.7,
        max_gpu_activations: int = 10,
        warmup_steps: int = 100
    ):
        """
        Args:
            model: PyTorch model to optimize
            device: Primary compute device
            enable_*: Flags to enable specific optimizations
        """
        self.model = model
        self.device = device
        
        # Initialize optimization components
        self.gradient_offloader = ActiveGradientOffloader(
            model, gradient_offload_threshold, device
        ) if enable_gradient_offload else None
        
        self.activation_manager = ActivationSwapManager(
            max_gpu_activations
        ) if enable_activation_swap else None
        
        self.neuron_preloader = NeuronPreloader(
            model, warmup_steps
        ) if enable_neuron_preload else None
        
        self.dynamic_gating = enable_dynamic_gating
        
    def setup(self):
        """Setup all optimization components"""
        if self.gradient_offloader:
            self.gradient_offloader.register_hooks()
            print("✓ Active Gradient Offloading enabled")
        
        if self.neuron_preloader:
            self.neuron_preloader.register_hooks()
            print("✓ Neuron Preloading enabled (warmup required)")
        
        if self.activation_manager:
            print("✓ Activation Swapping enabled")
        
        if self.dynamic_gating:
            print("✓ Dynamic Gating enabled")
    
    def pre_backward(self):
        """Call before backward pass"""
        pass
    
    def post_backward(self):
        """Call after backward pass"""
        if self.gradient_offloader:
            self.gradient_offloader.restore_all_gradients()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.gradient_offloader:
            self.gradient_offloader.clear()
        
        if self.activation_manager:
            self.activation_manager.clear()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            stats['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        return stats

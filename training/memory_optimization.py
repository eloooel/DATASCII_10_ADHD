"""
GPU Memory Optimization Utilities
Implements gradient checkpointing, mixed precision training, and memory-efficient attention
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable
import gc


class MemoryEfficientTrainer:
    """Wrapper for memory-efficient training techniques"""
    
    def __init__(
        self,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """
        Args:
            use_amp: Use automatic mixed precision (FP16)
            use_gradient_checkpointing: Use gradient checkpointing to trade compute for memory
            gradient_accumulation_steps: Accumulate gradients over multiple batches
        """
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        print(f"Memory Optimization Settings:")
        print(f"  - Mixed Precision (AMP): {self.use_amp}")
        print(f"  - Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"  - Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
    
    def enable_gradient_checkpointing(self, model: nn.Module):
        """Enable gradient checkpointing for the model"""
        if not self.use_gradient_checkpointing:
            return
        
        # Enable for GNN components
        if hasattr(model, 'gnn_encoder'):
            for module in model.gnn_encoder.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
        
        # Enable for STAN components
        if hasattr(model, 'stan_encoder'):
            for module in model.stan_encoder.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
    
    def train_step(
        self,
        model: nn.Module,
        batch: dict,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> tuple:
        """
        Memory-efficient training step with optional AMP and gradient accumulation
        
        Returns:
            (loss, predictions)
        """
        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast():
                outputs = model(batch['fc_matrix'], batch['timeseries'])
                loss = criterion(outputs, batch['label'])
                loss = loss / self.gradient_accumulation_steps
        else:
            outputs = model(batch['fc_matrix'], batch['timeseries'])
            loss = criterion(outputs, batch['label'])
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Get predictions
        _, predicted = outputs.max(1)
        
        return loss.item() * self.gradient_accumulation_steps, predicted
    
    @staticmethod
    def clear_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_memory_stats() -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    
    @staticmethod
    def print_memory_stats(prefix: str = ""):
        """Print current GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        stats = MemoryEfficientTrainer.get_memory_stats()
        print(f"{prefix}GPU Memory: "
              f"Allocated={stats['allocated']:.2f}GB, "
              f"Reserved={stats['reserved']:.2f}GB, "
              f"Peak={stats['max_allocated']:.2f}GB")


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention mechanism using chunking"""
    
    def __init__(self, chunk_size: int = 32):
        super().__init__()
        self.chunk_size = chunk_size
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage
        
        Args:
            query: (batch, seq_len, dim)
            key: (batch, seq_len, dim)
            value: (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = query.shape
        
        # Compute attention in chunks
        output = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            query_chunk = query[:, i:end_idx, :]
            
            # Attention scores
            scores = torch.matmul(query_chunk, key.transpose(-2, -1)) / (dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Weighted values
            chunk_output = torch.matmul(attn_weights, value)
            output.append(chunk_output)
        
        return torch.cat(output, dim=1)
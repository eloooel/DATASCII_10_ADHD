"""
Training Integration with Memory Optimization

Integrates all optimization techniques into the training loop.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, TYPE_CHECKING
from pathlib import Path

# Import only for type hints to avoid circular import
if TYPE_CHECKING:
    from models.gnn_stan_hybrid_optimized import MemoryOptimizedGNNSTAN

from optimization.advanced_memory_optimization import MemoryOptimizationManager


class OptimizedTrainer:
    """
    Training wrapper with integrated memory optimization
    """
    
    def __init__(
        self,
        model_config: dict,
        device: str = 'cuda',
        # Optimization settings
        enable_gradient_offload: bool = True,
        enable_activation_swap: bool = True,
        enable_neuron_preload: bool = False,
        gradient_offload_threshold: float = 0.75,
        max_gpu_activations: int = 8,
        warmup_steps: int = 50
    ):
        """
        Args:
            model_config: Model configuration dict
            device: Training device
            enable_*: Optimization flags
        """
        # Import here to avoid circular import
        from models.gnn_stan_hybrid_optimized import MemoryOptimizedGNNSTAN
        
        self.device = device
        self.model_config = model_config
        
        # Create optimized model
        self.model = MemoryOptimizedGNNSTAN(
            hidden_dim=model_config.get('hidden_dim', 128),
            num_classes=model_config.get('num_classes', 2),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.3),
            gnn_config=model_config.get('gnn', {}),
            stan_config=model_config.get('stan', {}),
            fusion_config=model_config.get('fusion', {}),
            classifier_dropout=model_config.get('classifier_dropout', 0.5),
            # Optimization specific
            use_hybrid_attention=True,
            use_dynamic_gating=True,
            attention_chunk_size=64,
            attention_sparsity=0.2,
            gating_threshold=0.5
        ).to(device)
        
        # Setup optimization manager
        self.opt_manager = self.model.setup_optimization(
            device=device,
            enable_gradient_offload=enable_gradient_offload,
            enable_activation_swap=enable_activation_swap,
            enable_neuron_preload=enable_neuron_preload,
            gradient_offload_threshold=gradient_offload_threshold,
            max_gpu_activations=max_gpu_activations,
            warmup_steps=warmup_steps
        )
        
        self.warmup_count = 0
        self.warmup_steps = warmup_steps
        self.is_warmed_up = False
        
    def train_step(
        self,
        fc_matrix: torch.Tensor,
        roi_timeseries: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Single training step with optimization
        
        Args:
            fc_matrix: Functional connectivity (batch, 200, 200)
            roi_timeseries: ROI timeseries (batch, time, 200)
            labels: Ground truth labels (batch,)
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        
        # Move data to device
        fc_matrix = fc_matrix.to(self.device)
        roi_timeseries = roi_timeseries.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(fc_matrix, roi_timeseries, return_attention=False)
        logits = outputs['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass with gradient offloading
        if self.opt_manager and self.opt_manager.gradient_offloader:
            self.opt_manager.pre_backward()
        
        loss.backward()
        
        if self.opt_manager and self.opt_manager.gradient_offloader:
            self.opt_manager.post_backward()
        
        # Optimizer step
        optimizer.step()
        
        # Warmup neuron preloader
        if self.opt_manager and self.opt_manager.neuron_preloader:
            if not self.is_warmed_up:
                self.warmup_count += 1
                if self.warmup_count >= self.warmup_steps:
                    self.opt_manager.neuron_preloader.finalize_warmup()
                    self.is_warmed_up = True
                    print("Success: Neuron preloading warmup complete")
        
        # Compute metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': acc.item(),
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
    
    @torch.no_grad()
    def eval_step(
        self,
        fc_matrix: torch.Tensor,
        roi_timeseries: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Single evaluation step
        
        Args:
            fc_matrix: Functional connectivity
            roi_timeseries: ROI timeseries
            labels: Ground truth labels
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.eval()
        
        fc_matrix = fc_matrix.to(self.device)
        roi_timeseries = roi_timeseries.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(fc_matrix, roi_timeseries, return_attention=False)
        logits = outputs['logits']
        
        loss = criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': acc.item()
        }
    
    def get_model(self) -> nn.Module:
        """Get the underlying model"""
        return self.model
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            **kwargs
        }
        torch.save(checkpoint, path)
        print(f"Success: Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Success: Checkpoint loaded from {path}")
        return checkpoint
    
    def print_stats(self):
        """Print optimization statistics"""
        self.model.print_optimization_stats()
    
    def cleanup(self):
        """Cleanup optimization resources"""
        if self.opt_manager:
            self.opt_manager.cleanup()


def create_optimized_trainer(
    model_config: dict,
    device: str = 'cuda',
    optimization_level: str = 'high'
) -> OptimizedTrainer:
    """
    Factory function to create trainer with predefined optimization levels
    
    Args:
        model_config: Model configuration
        device: Training device
        optimization_level: 'low', 'medium', or 'high'
        
    Returns:
        OptimizedTrainer instance
    """
    # Import here to avoid circular import
    from models.gnn_stan_hybrid_optimized import MemoryOptimizedGNNSTAN
    
    optimization_presets = {
        'low': {
            'enable_gradient_offload': False,
            'enable_activation_swap': False,
            'enable_neuron_preload': False,
            'gradient_offload_threshold': 0.9,
            'max_gpu_activations': 20
        },
        'medium': {
            'enable_gradient_offload': True,
            'enable_activation_swap': True,
            'enable_neuron_preload': False,
            'gradient_offload_threshold': 0.75,
            'max_gpu_activations': 10
        },
        'high': {
            'enable_gradient_offload': True,
            'enable_activation_swap': True,
            'enable_neuron_preload': True,
            'gradient_offload_threshold': 0.65,
            'max_gpu_activations': 6
        }
    }
    
    preset = optimization_presets.get(optimization_level, optimization_presets['medium'])
    
    return OptimizedTrainer(
        model_config=model_config,
        device=device,
        **preset
    )


# Example usage:
if __name__ == "__main__":
    # Example configuration
    model_config = {
        'hidden_dim': 128,
        'num_classes': 2,
        'num_heads': 4,
        'dropout': 0.3,
        'gnn': {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3,
            'pool_ratios': [0.8, 0.6]
        },
        'stan': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'fusion': {
            'fusion_dim': 128,
            'dropout': 0.3
        },
        'classifier_dropout': 0.5
    }
    
    # Create trainer with high optimization
    trainer = create_optimized_trainer(
        model_config=model_config,
        device='cuda',
        optimization_level='high'
    )
    
    print("Success: Optimized trainer created")
    print("\nOptimization features:")
    print("  - Active Gradient Offloading: Enabled")
    print("  - Holistic Activation Swapping: Enabled")
    print("  - Hybrid Attention Blocks: Enabled")
    print("  - Hot/Cold Neuron Preloading: Enabled")
    print("  - Dynamic Layer Gating: Enabled")

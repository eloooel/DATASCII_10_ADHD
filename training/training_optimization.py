"""
Training and Optimization Module for GNN-STAN Hybrid Model

Provides complete training pipeline with:
- ADHD Dataset class with data augmentation
- Focal Loss for class imbalance handling
- Early stopping and learning rate scheduling
- Leave-One-Site-Out cross-validation
- Comprehensive metrics tracking and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..models import GNNSTANHybrid


class ADHDDataset(Dataset):
    """Dataset class for ADHD rs-fMRI data"""
    
    def __init__(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray, 
                 labels: np.ndarray, sites: np.ndarray, demographics: Optional[np.ndarray] = None,
                 augment: bool = False):
        self.fc_matrices = torch.FloatTensor(fc_matrices)
        self.roi_timeseries = torch.FloatTensor(roi_timeseries)
        self.labels = torch.LongTensor(labels)
        self.sites = sites
        self.demographics = demographics
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fc_matrix = self.fc_matrices[idx]
        roi_ts = self.roi_timeseries[idx]
        label = self.labels[idx]
        site = self.sites[idx]
        
        # Data augmentation for training
        if self.augment and self.training:
            fc_matrix, roi_ts = self._augment_data(fc_matrix, roi_ts)
        
        # Create graph data structure
        edge_index, edge_weights = self._create_graph_structure(fc_matrix)
        
        return {
            'fc_matrix': fc_matrix,
            'roi_timeseries': roi_ts,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'label': label,
            'site': site
        }
    
    def _augment_data(self, fc_matrix: torch.Tensor, roi_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation techniques"""
        
        # 1. Add small amount of Gaussian noise to time series
        noise_std = 0.01
        roi_ts = roi_ts + torch.randn_like(roi_ts) * noise_std
        
        # 2. Temporal jittering (shift time series slightly)
        if torch.rand(1) < 0.3:
            shift = torch.randint(-5, 6, (1,)).item()
            if shift != 0:
                roi_ts = torch.roll(roi_ts, shift, dims=1)
        
        # 3. ROI dropout (randomly set some ROIs to zero)
        if torch.rand(1) < 0.2:
            dropout_mask = torch.rand(fc_matrix.shape[0]) > 0.05  # Keep 95% of ROIs
            fc_matrix = fc_matrix * dropout_mask.unsqueeze(1) * dropout_mask.unsqueeze(0)
            roi_ts = roi_ts * dropout_mask.unsqueeze(1)
        
        # 4. Connectivity thresholding variation
        if torch.rand(1) < 0.3:
            threshold_factor = 1.0 + torch.randn(1).item() * 0.1  # ±10% variation
            fc_matrix = fc_matrix * threshold_factor
        
        return fc_matrix, roi_ts
    
    def _create_graph_structure(self, fc_matrix: torch.Tensor, threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph edge structure from functional connectivity matrix"""
        
        # Apply threshold to create sparse graph
        abs_fc = torch.abs(fc_matrix)
        edge_mask = abs_fc > threshold
        
        # Get edge indices
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        edge_weights = fc_matrix[edge_mask]
        
        return edge_index, edge_weights


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self._save_checkpoint(model)
            return False
        
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self._save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False
    
    def _save_checkpoint(self, model: nn.Module):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class TrainingOptimizationModule:
    """Complete training and optimization module for GNN-STAN hybrid model"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Training history
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': []
        }
        
        # Cross-validation results
        self.cv_results = []
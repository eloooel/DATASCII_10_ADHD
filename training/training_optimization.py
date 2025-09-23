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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from tqdm import tqdm

from models import GNNSTANHybrid


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
        self.training = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fc_matrix = self.fc_matrices[idx]
        roi_ts = self.roi_timeseries[idx]
        label = self.labels[idx]
        site = self.sites[idx]

        if self.augment and self.training:
            fc_matrix, roi_ts = self._augment_data(fc_matrix, roi_ts)

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
        noise_std = 0.01
        roi_ts = roi_ts + torch.randn_like(roi_ts) * noise_std

        if torch.rand(1) < 0.3:
            shift = torch.randint(-5, 6, (1,)).item()
            if shift != 0:
                roi_ts = torch.roll(roi_ts, shift, dims=1)

        if torch.rand(1) < 0.2:
            dropout_mask = torch.rand(fc_matrix.shape[0]) > 0.05
            fc_matrix = fc_matrix * dropout_mask.unsqueeze(1) * dropout_mask.unsqueeze(0)
            roi_ts = roi_ts * dropout_mask.unsqueeze(1)

        if torch.rand(1) < 0.3:
            threshold_factor = 1.0 + torch.randn(1).item() * 0.1
            fc_matrix = fc_matrix * threshold_factor

        return fc_matrix, roi_ts

    def _create_graph_structure(self, fc_matrix: torch.Tensor, threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_fc = torch.abs(fc_matrix)
        edge_mask = abs_fc > threshold
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        edge_weights = fc_matrix[edge_mask]
        return edge_index, edge_weights

    @staticmethod
    def collate_fn(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}


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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_training(self, fc_matrices, roi_timeseries, labels, sites):
        dataset = ADHDDataset(fc_matrices, roi_timeseries, labels, sites, augment=True)

        logo = LeaveOneGroupOut()
        all_results = []

        for fold, (train_idx, val_idx) in enumerate(logo.split(np.arange(len(labels)), labels, sites)):
            self.logger.info(f"Fold {fold+1}/{logo.get_n_splits()} - Training on {len(train_idx)}, Validating on {len(val_idx)}")

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)

            dataset.training = True
            train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True,
                                      collate_fn=ADHDDataset.collate_fn)
            dataset.training = False
            val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'], shuffle=False,
                                    collate_fn=ADHDDataset.collate_fn)

            model = GNNSTANHybrid(self.model_config).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.training_config['lr'])
            criterion = FocalLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            early_stopping = EarlyStopping(patience=self.training_config['patience'])

            best_val_acc = 0
            for epoch in range(self.training_config['epochs']):
                model.train()
                train_losses = []
                for batch in train_loader:
                    optimizer.zero_grad()
                    fc = torch.stack(batch['fc_matrix']).to(self.device)
                    ts = torch.stack(batch['roi_timeseries']).to(self.device)
                    labels_batch = torch.stack(batch['label']).to(self.device)
                    edge_index = [ei.to(self.device) for ei in batch['edge_index']]
                    edge_weights = [ew.to(self.device) for ew in batch['edge_weights']]

                    outputs = model(fc, ts, edge_index, None)
                    loss = criterion(outputs['logits'], labels_batch)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                val_acc, val_auc = self._evaluate(model, val_loader)
                scheduler.step(val_acc)

                self.logger.info(f"Epoch {epoch+1}: Train Loss {np.mean(train_losses):.4f}, Val Acc {val_acc:.4f}, Val AUC {val_auc:.4f}")

                if early_stopping(val_acc, model):
                    self.logger.info("Early stopping triggered")
                    break

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint_path = Path(self.training_config['output_dir']) / f"best_model_fold{fold}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    self.logger.info(f"Saved best model for fold {fold+1} at {checkpoint_path}")

            all_results.append({'fold': fold, 'val_acc': best_val_acc})

        return all_results

    def _evaluate(self, model, dataloader):
        model.eval()
        preds, probs, true = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                fc = torch.stack(batch['fc_matrix']).to(self.device)
                ts = torch.stack(batch['roi_timeseries']).to(self.device)
                labels_batch = torch.stack(batch['label']).to(self.device)
                edge_index = [ei.to(self.device) for ei in batch['edge_index']]
                outputs = model(fc, ts, edge_index, None)
                logits = outputs['logits']
                p = torch.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().numpy())
                probs.extend(p.cpu().numpy()[:, 1])
                true.extend(labels_batch.cpu().numpy())
        acc = accuracy_score(true, preds)
        try:
            auc = roc_auc_score(true, probs)
        except:
            auc = 0.0
        return acc, auc

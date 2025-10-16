import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from .dataset import ADHDDataset
from .data_splitter import DataSplitter
from models import GNNSTANHybrid
from optimization import FocalLoss, EarlyStopping



class TrainingOptimizationModule:
    """Training pipeline for GNN-STAN hybrid model"""

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
            train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'],
                                      shuffle=True, collate_fn=ADHDDataset.collate_fn)
            dataset.training = False
            val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'],
                                    shuffle=False, collate_fn=ADHDDataset.collate_fn)

            model = GNNSTANHybrid(self.model_config).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.training_config['lr'])
            criterion = FocalLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
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

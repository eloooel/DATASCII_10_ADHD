"""
K-Fold Cross-Validation

Standard k-fold and stratified k-fold cross-validation implementations.
Provides alternative validation strategies when site information is not
available or when standard statistical validation is desired.
"""

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Any, Optional
import time
from tqdm import tqdm

from .base_validator import BaseValidator
from models import GNNSTANHybrid
from training.dataset import ADHDDataset
from optimization import FocalLoss, EarlyStopping


class KFoldValidator(BaseValidator):
    """Standard K-Fold Cross-Validation"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any], 
                 n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = 42):
        super().__init__(model_config, training_config)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.validation_type = f"{n_splits}-Fold CV"
    
    def validate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                labels: np.ndarray, sites: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation
        
        Args:
            fc_matrices: Functional connectivity matrices
            roi_timeseries: ROI time series
            labels: Subject labels
            sites: Site information (used for statistics only)
            
        Returns:
            Complete validation results
        """
        
        self.logger.info(f"Starting {self.validation_type}")
        self.logger.info(f"Total subjects: {len(labels)}")
        self.logger.info(f"Class distribution: {np.bincount(labels)}")
        
        # Create folds
        folds = self.create_folds(fc_matrices, labels, sites)
        self.logger.info(f"Created {len(folds)} folds")
        
        # Perform cross-validation
        fold_results = []
        total_start_time = time.time()
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"FOLD {fold_idx + 1}/{len(folds)}")
            self.logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
            
            # Log fold composition
            self._log_fold_composition(labels, sites, train_idx, test_idx, fold_idx)
            
            # Train and evaluate
            fold_result = self._train_and_evaluate_fold(
                fc_matrices, roi_timeseries, labels, sites,
                train_idx, test_idx, fold_idx
            )
            
            fold_results.append(fold_result)
            self._log_fold_results(fold_result)
        
        total_time = time.time() - total_start_time
        
        # Aggregate results
        summary = self.aggregate_results(fold_results)
        statistical_tests = self.statistical_tests(fold_results)
        
        results = {
            'validation_type': self.validation_type,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'summary': summary,
            'statistical_tests': statistical_tests,
            'total_validation_time': total_time,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        self.log_validation_summary(results)
        return results
    
    def create_folds(self, fc_matrices: np.ndarray, labels: np.ndarray, 
                    sites: np.ndarray, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create K-Fold splits"""
        
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        return list(kfold.split(fc_matrices))
    
    def _train_and_evaluate_fold(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                               labels: np.ndarray, sites: np.ndarray,
                               train_idx: np.ndarray, test_idx: np.ndarray,
                               fold_idx: int) -> Dict[str, Any]:
        """Train and evaluate model for a single fold"""
        
        fold_start_time = time.time()
        
        # Create datasets
        train_dataset = ADHDDataset(
            fc_matrices=fc_matrices[train_idx],
            roi_timeseries=roi_timeseries[train_idx],
            labels=labels[train_idx],
            sites=sites[train_idx],
            augment=True
        )
        
        test_dataset = ADHDDataset(
            fc_matrices=fc_matrices[test_idx],
            roi_timeseries=roi_timeseries[test_idx],
            labels=labels[test_idx],
            sites=sites[test_idx],
            augment=False
        )
        
        # Create data loaders
        train_loader = self._create_data_loader(train_dataset, is_train=True, labels=labels[train_idx])
        test_loader = self._create_data_loader(test_dataset, is_train=False)
        
        # Initialize and train model
        model = GNNSTANHybrid(**self.model_config).to(self.device)
        training_history = self._train_model(model, train_loader, test_loader, fold_idx)
        
        # Final evaluation
        test_metrics = self.evaluate_model(model, test_loader)
        
        training_time = time.time() - fold_start_time
        
        return {
            'fold_id': fold_idx,
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            'train_class_distribution': np.bincount(labels[train_idx]).tolist(),
            'test_class_distribution': np.bincount(labels[test_idx]).tolist(),
            'training_time': training_time,
            'training_history': training_history,
            'test_metrics': test_metrics
        }
    
    def _create_data_loader(self, dataset: ADHDDataset, is_train: bool, labels: np.ndarray = None):
        """Create data loader with optional balanced sampling"""
        from torch.utils.data import DataLoader, WeightedRandomSampler
        
        if is_train and labels is not None:
            # Create balanced sampler for training
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                dataset,
                batch_size=self.training_config.get('batch_size', 16),
                sampler=sampler,
                collate_fn=self._collate_batch,
                num_workers=self.training_config.get('num_workers', 4)
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.training_config.get('batch_size', 16),
                shuffle=False,
                collate_fn=self._collate_batch,
                num_workers=self.training_config.get('num_workers', 4)
            )
    
    def _train_model(self, model, train_loader, val_loader, fold_idx):
        """Train model with early stopping"""
        
        # Setup training components
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)
        criterion = self._setup_criterion()
        early_stopping = EarlyStopping(
            patience=self.training_config.get('patience', 15)
        )
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        epochs = self.training_config.get('epochs', 100)
        
        with tqdm(range(epochs), desc=f"Fold {fold_idx + 1} Training", leave=False) as pbar:
            for epoch in pbar:
                # Training phase
                train_metrics = self._train_epoch(model, train_loader, criterion, optimizer)
                val_metrics = self._validate_epoch(model, val_loader, criterion)
                
                # Update scheduler
                if scheduler:
                    scheduler.step(val_metrics['loss'])
                
                # Record history
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Update progress
                pbar.set_postfix({
                    'Train_Acc': f"{train_metrics['accuracy']:.3f}",
                    'Val_Acc': f"{val_metrics['accuracy']:.3f}"
                })
                
                # Early stopping
                if early_stopping(val_metrics['accuracy'], model):
                    break
        
        return history
    
    def _log_fold_composition(self, labels, sites, train_idx, test_idx, fold_idx):
        """Log composition of train/test sets"""
        
        # Class distribution
        train_dist = np.bincount(labels[train_idx])
        test_dist = np.bincount(labels[test_idx])
        
        self.logger.info(f"  Train class dist: ADHD={train_dist[1] if len(train_dist) > 1 else 0}, "
                        f"Control={train_dist[0]}")
        self.logger.info(f"  Test class dist: ADHD={test_dist[1] if len(test_dist) > 1 else 0}, "
                        f"Control={test_dist[0]}")
        
        # Site distribution if available
        if sites is not None:
            train_sites = np.unique(sites[train_idx])
            test_sites = np.unique(sites[test_idx])
            
            self.logger.info(f"  Train sites ({len(train_sites)}): {train_sites}")
            self.logger.info(f"  Test sites ({len(test_sites)}): {test_sites}")
    
    # Helper methods (similar to LOSO implementation)
    def _setup_optimizer(self, model):
        optimizer_name = self.training_config.get('optimizer', 'adamw')
        lr = self.training_config.get('learning_rate', 1e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        
        if optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    def _setup_scheduler(self, optimizer):
        if self.training_config.get('use_scheduler', True):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
        return None
    
    def _setup_criterion(self):
        if self.training_config.get('use_focal_loss', True):
            return FocalLoss(
                alpha=self.training_config.get('focal_alpha', 0.8),
                gamma=self.training_config.get('focal_gamma', 2.0)
            )
        else:
            return torch.nn.CrossEntropyLoss()
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in train_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(batch['fc_matrix'], batch['roi_timeseries'], 
                          batch['edge_index'], batch['batch'])
            loss = criterion(outputs['logits'], batch['label'])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
        
        return {'loss': total_loss / len(train_loader), 'accuracy': correct / total}
    
    def _validate_epoch(self, model, val_loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = model(batch['fc_matrix'], batch['roi_timeseries'],
                              batch['edge_index'], batch['batch'])
                loss = criterion(outputs['logits'], batch['label'])
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()
        
        return {'loss': total_loss / len(val_loader), 'accuracy': correct / total}
    
    def _collate_batch(self, batch_list):
        """Custom collate function for graph batching"""
        fc_matrices = torch.stack([item['fc_matrix'] for item in batch_list])
        roi_timeseries = torch.stack([item['roi_timeseries'] for item in batch_list])
        labels = torch.stack([item['label'] for item in batch_list])
        
        # Handle graph structure
        batch_indices = []
        edge_indices = []
        edge_weights = []
        
        for i, item in enumerate(batch_list):
            edge_index = item['edge_index'] + i * fc_matrices.shape[1]
            edge_indices.append(edge_index)
            edge_weights.append(item['edge_weights'])
            batch_indices.extend([i] * fc_matrices.shape[1])
        
        return {
            'fc_matrix': fc_matrices,
            'roi_timeseries': roi_timeseries,
            'edge_index': torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
            'edge_weights': torch.cat(edge_weights) if edge_weights else torch.empty(0),
            'batch': torch.tensor(batch_indices, dtype=torch.long),
            'label': labels
        }
    
    def _log_fold_results(self, fold_result):
        """Log fold results"""
        metrics = fold_result['test_metrics']
        self.logger.info(f"Fold {fold_result['fold_id'] + 1} Results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        self.logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        self.logger.info(f"  AUC: {metrics['auc']:.4f}")


class StratifiedKFoldValidator(KFoldValidator):
    """Stratified K-Fold Cross-Validation for balanced class distribution"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any],
                 n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = 42):
        super().__init__(model_config, training_config, n_splits, shuffle, random_state)
        self.validation_type = f"Stratified {n_splits}-Fold CV"
    
    def create_folds(self, fc_matrices: np.ndarray, labels: np.ndarray, 
                    sites: np.ndarray, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified K-Fold splits maintaining class balance"""
        
        skfold = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        return list(skfold.split(fc_matrices, labels))
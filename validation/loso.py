"""
Leave-One-Site-Out (LOSO) Cross-Validation

Specialized validation for neuroimaging data to handle site effects.
Each fold uses one site as test set and all other sites for training.
This approach helps evaluate model generalizability across different 
scanning sites and acquisition protocols.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Tuple, Any, Optional
import time
from tqdm import tqdm

from .base_validator import BaseValidator
from ..models import GNNSTANHybrid
from ..training.train import ADHDDataset, FocalLoss, EarlyStopping


class LeaveOneSiteOutValidator(BaseValidator):
    """Leave-One-Site-Out Cross-Validation for handling site effects"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        super().__init__(model_config, training_config)
        self.validation_type = "Leave-One-Site-Out"
    
    def validate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                labels: np.ndarray, sites: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform Leave-One-Site-Out cross-validation
        
        Args:
            fc_matrices: Functional connectivity matrices (n_subjects, n_rois, n_rois)
            roi_timeseries: ROI time series (n_subjects, n_rois, n_timepoints)
            labels: Subject labels (n_subjects,)
            sites: Site information (n_subjects,)
            
        Returns:
            Dictionary containing complete validation results
        """
        
        self.logger.info(f"Starting {self.validation_type} Cross-Validation")
        self.logger.info(f"Total subjects: {len(labels)}")
        
        # Get unique sites and log information
        unique_sites = np.unique(sites)
        self.logger.info(f"Number of sites: {len(unique_sites)}")
        self.logger.info(f"Sites: {unique_sites}")
        
        # Log site-wise statistics
        self._log_site_statistics(labels, sites)
        
        # Create folds
        folds = self.create_folds(fc_matrices, labels, sites)
        self.logger.info(f"Created {len(folds)} folds")
        
        # Perform cross-validation
        fold_results = []
        total_start_time = time.time()
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            test_site = sites[test_idx[0]]  # All test samples are from same site
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FOLD {fold_idx + 1}/{len(folds)}: Testing on site '{test_site}'")
            self.logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
            
            # Train and evaluate for this fold
            fold_result = self._train_and_evaluate_fold(
                fc_matrices, roi_timeseries, labels, sites,
                train_idx, test_idx, fold_idx, test_site
            )
            
            fold_results.append(fold_result)
            
            # Log fold results
            self._log_fold_results(fold_result)
        
        total_time = time.time() - total_start_time
        self.logger.info(f"\nTotal validation time: {total_time:.2f} seconds")
        
        # Aggregate results
        summary = self.aggregate_results(fold_results)
        statistical_tests = self.statistical_tests(fold_results)
        
        # Create final results dictionary
        results = {
            'validation_type': self.validation_type,
            'fold_results': fold_results,
            'summary': summary,
            'statistical_tests': statistical_tests,
            'site_information': self._get_site_information(labels, sites),
            'total_validation_time': total_time,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        # Log final summary
        self.log_validation_summary(results)
        
        return results
    
    def create_folds(self, fc_matrices: np.ndarray, labels: np.ndarray, 
                    sites: np.ndarray, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create LOSO folds - one fold per site"""
        
        logo = LeaveOneGroupOut()
        folds = list(logo.split(fc_matrices, labels, sites))
        
        self.logger.info(f"Created {len(folds)} LOSO folds")
        
        # Validate folds
        for i, (train_idx, test_idx) in enumerate(folds):
            test_sites = np.unique(sites[test_idx])
            train_sites = np.unique(sites[train_idx])
            
            if len(test_sites) > 1:
                self.logger.warning(f"Fold {i}: Test set contains multiple sites: {test_sites}")
            
            # Check that test site is not in training set
            overlap = set(test_sites) & set(train_sites)
            if overlap:
                self.logger.error(f"Fold {i}: Site overlap between train and test: {overlap}")
        
        return folds
    
    def _train_and_evaluate_fold(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                               labels: np.ndarray, sites: np.ndarray,
                               train_idx: np.ndarray, test_idx: np.ndarray,
                               fold_idx: int, test_site: str) -> Dict[str, Any]:
        """Train and evaluate model for a single fold"""
        
        fold_start_time = time.time()
        
        # Create datasets
        train_dataset = ADHDDataset(
            fc_matrices[train_idx], roi_timeseries[train_idx],
            labels[train_idx], sites[train_idx], augment=True
        )
        
        test_dataset = ADHDDataset(
            fc_matrices[test_idx], roi_timeseries[test_idx],
            labels[test_idx], sites[test_idx], augment=False
        )
        
        # Create data loaders
        train_loader = self._create_train_loader(train_dataset, labels[train_idx])
        test_loader = self._create_test_loader(test_dataset)
        
        # Initialize model
        model = GNNSTANHybrid(self.model_config).to(self.device)
        
        # Setup training components
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)
        criterion = self._setup_criterion()
        early_stopping = EarlyStopping(
            patience=self.training_config.get('patience', 15),
            min_delta=self.training_config.get('min_delta', 1e-4)
        )
        
        # Training loop
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        epochs = self.training_config.get('epochs', 100)
        
        self.logger.info(f"Training for fold {fold_idx + 1}...")
        
        with tqdm(range(epochs), desc=f"Fold {fold_idx + 1} Training", leave=False) as pbar:
            for epoch in pbar:
                # Training phase
                train_metrics = self._train_epoch(model, train_loader, criterion, optimizer)
                
                # Validation phase (using test set as validation for early stopping)
                val_metrics = self._validate_epoch(model, test_loader, criterion)
                
                # Update scheduler
                if scheduler:
                    scheduler.step(val_metrics['loss'])
                
                # Record history
                training_history['train_loss'].append(train_metrics['loss'])
                training_history['train_acc'].append(train_metrics['accuracy'])
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_acc'].append(val_metrics['accuracy'])
                
                # Update progress bar
                pbar.set_postfix({
                    'Train_Acc': f"{train_metrics['accuracy']:.3f}",
                    'Val_Acc': f"{val_metrics['accuracy']:.3f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Early stopping
                if early_stopping(val_metrics['accuracy'], model):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Final evaluation
        test_metrics = self.evaluate_model(model, test_loader)
        
        training_time = time.time() - fold_start_time
        
        return {
            'fold_id': fold_idx,
            'test_site': test_site,
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            'train_sites': np.unique(sites[train_idx]).tolist(),
            'training_time': training_time,
            'final_epoch': epoch + 1,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'model_state': model.state_dict() if self.training_config.get('save_models', False) else None
        }
    
    def _create_train_loader(self, dataset: ADHDDataset, labels: np.ndarray) -> DataLoader:
        """Create training data loader with balanced sampling"""
        
        # Create balanced sampler
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
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True
        )
    
    def _create_test_loader(self, dataset: ADHDDataset) -> DataLoader:
        """Create test data loader"""
        
        return DataLoader(
            dataset,
            batch_size=self.training_config.get('batch_size', 16),
            shuffle=False,
            collate_fn=self._collate_batch,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True
        )
    
    def _collate_batch(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for graph batching"""
        
        # Extract components
        fc_matrices = torch.stack([item['fc_matrix'] for item in batch_list])
        roi_timeseries = torch.stack([item['roi_timeseries'] for item in batch_list])
        labels = torch.stack([item['label'] for item in batch_list])
        sites = [item['site'] for item in batch_list]
        
        # Create batch indices for graph pooling
        batch_indices = []
        edge_indices = []
        edge_weights = []
        
        for i, item in enumerate(batch_list):
            edge_index = item['edge_index']
            edge_weight = item['edge_weights']
            
            # Add batch offset to edge indices
            edge_index_offset = edge_index + i * fc_matrices.shape[1]  # n_rois offset
            edge_indices.append(edge_index_offset)
            edge_weights.append(edge_weight)
            
            # Create batch indices (which graph each node belongs to)
            batch_indices.extend([i] * fc_matrices.shape[1])  # n_rois per graph
        
        # Concatenate all edges
        if edge_indices:
            all_edge_indices = torch.cat(edge_indices, dim=1)
            all_edge_weights = torch.cat(edge_weights)
        else:
            all_edge_indices = torch.empty((2, 0), dtype=torch.long)
            all_edge_weights = torch.empty(0)
        
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)
        
        return {
            'fc_matrix': fc_matrices,
            'roi_timeseries': roi_timeseries,
            'edge_index': all_edge_indices,
            'edge_weights': all_edge_weights,
            'batch': batch_tensor,
            'label': labels,
            'sites': sites
        }
    
    def _setup_optimizer(self, model: nn.Module):
        """Setup optimizer"""
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
        """Setup learning rate scheduler"""
        if self.training_config.get('use_scheduler', True):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=False
            )
        return None
    
    def _setup_criterion(self):
        """Setup loss criterion"""
        if self.training_config.get('use_focal_loss', True):
            return FocalLoss(
                alpha=self.training_config.get('focal_alpha', 0.8),
                gamma=self.training_config.get('focal_gamma', 2.0)
            )
        else:
            return nn.CrossEntropyLoss()
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    criterion: nn.Module, optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                batch['fc_matrix'],
                batch['roi_timeseries'],
                batch['edge_index'],
                batch['batch']
            )
            
            # Compute loss
            loss = criterion(outputs['logits'], batch['label'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    batch['fc_matrix'],
                    batch['roi_timeseries'],
                    batch['edge_index'],
                    batch['batch']
                )
                
                # Compute loss
                loss = criterion(outputs['logits'], batch['label'])
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _log_site_statistics(self, labels: np.ndarray, sites: np.ndarray):
        """Log statistics for each site"""
        self.logger.info("\nSite Statistics:")
        self.logger.info("-" * 40)
        
        for site in np.unique(sites):
            site_mask = sites == site
            site_labels = labels[site_mask]
            n_total = len(site_labels)
            n_adhd = np.sum(site_labels == 1)
            n_control = np.sum(site_labels == 0)
            adhd_ratio = n_adhd / n_total if n_total > 0 else 0
            
            self.logger.info(f"{site}: {n_total} subjects (ADHD: {n_adhd}, Control: {n_control}, Ratio: {adhd_ratio:.2f})")
    
    def _log_fold_results(self, fold_result: Dict[str, Any]):
        """Log results for a single fold"""
        metrics = fold_result['test_metrics']
        
        self.logger.info(f"Fold {fold_result['fold_id'] + 1} Results (Site: {fold_result['test_site']}):")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        self.logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        self.logger.info(f"  AUC: {metrics['auc']:.4f}")
        self.logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        self.logger.info(f"  Training time: {fold_result['training_time']:.1f}s")
        self.logger.info(f"  Final epoch: {fold_result['final_epoch']}")
    
    def _get_site_information(self, labels: np.ndarray, sites: np.ndarray) -> Dict[str, Any]:
        """Get detailed site information"""
        site_info = {}
        
        for site in np.unique(sites):
            site_mask = sites == site
            site_labels = labels[site_mask]
            
            site_info[site] = {
                'total_subjects': int(len(site_labels)),
                'adhd_subjects': int(np.sum(site_labels == 1)),
                'control_subjects': int(np.sum(site_labels == 0)),
                'adhd_ratio': float(np.mean(site_labels == 1))
            }
        
        return site_info
    
    def get_site_specific_results(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract site-specific performance results"""
        site_results = {}
        
        for fold_result in results['fold_results']:
            site = fold_result['test_site']
            metrics = fold_result['test_metrics']
            
            site_results[site] = {
                'accuracy': metrics['accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'auc': metrics['auc'],
                'f1': metrics['f1'],
                'n_subjects': metrics['n_samples'],
                'training_time': fold_result['training_time']
            }
        
        return site_results
    
    def analyze_site_effects(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze site-specific effects and variability"""
        
        # Extract site-specific results
        site_results = self.get_site_specific_results(results)
        
        # Compute statistics across sites
        metrics = ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1']
        site_analysis = {}
        
        for metric in metrics:
            values = [site_results[site][metric] for site in site_results]
            
            site_analysis[f'{metric}_mean'] = float(np.mean(values))
            site_analysis[f'{metric}_std'] = float(np.std(values))
            site_analysis[f'{metric}_min'] = float(np.min(values))
            site_analysis[f'{metric}_max'] = float(np.max(values))
            site_analysis[f'{metric}_range'] = float(np.max(values) - np.min(values))
            
            # Coefficient of variation
            if np.mean(values) > 0:
                site_analysis[f'{metric}_cv'] = float(np.std(values) / np.mean(values))
        
        # Identify best and worst performing sites
        best_site = max(site_results.keys(), key=lambda s: site_results[s]['accuracy'])
        worst_site = min(site_results.keys(), key=lambda s: site_results[s]['accuracy'])
        
        site_analysis['best_performing_site'] = {
            'site': best_site,
            'accuracy': site_results[best_site]['accuracy']
        }
        
        site_analysis['worst_performing_site'] = {
            'site': worst_site,
            'accuracy': site_results[worst_site]['accuracy']
        }
        
        # Site effect magnitude
        site_analysis['site_effect_magnitude'] = site_analysis['accuracy_range']
        
        return site_analysis
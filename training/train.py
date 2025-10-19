import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from .dataset import ADHDDataset
from .data_splitter import DataSplitter
from models import GNNSTANHybrid
from optimization import EarlyStopping, FocalLoss
from .memory_optimization import MemoryEfficientTrainer


class TrainingOptimizationModule:
    """Complete training pipeline with CV and LOSO validation"""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: torch.device = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (ROI count inferred automatically)
        self.model = GNNSTANHybrid(
            hidden_dim=model_config.get('hidden_dim', 128),
            num_classes=model_config.get('num_classes', 2),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.3),
            gnn_config=model_config.get('gnn', {}),
            stan_config=model_config.get('stan', {}),
            fusion_config=model_config.get('fusion', {}),
            classifier_dropout=model_config.get('classifier_dropout', 0.5)
        ).to(self.device)
        
        # Loss function
        if training_config.get('use_focal_loss', True):
            self.criterion = FocalLoss(
                alpha=training_config.get('focal_alpha', 0.25),
                gamma=training_config.get('focal_gamma', 2.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.get('early_stopping_patience', 10),
            min_delta=training_config.get('early_stopping_min_delta', 0.001),
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_model_path = None
        
        # Initialize memory-efficient trainer
        self.memory_optimizer = MemoryEfficientTrainer(
            use_amp=training_config.get('use_amp', True),
            use_gradient_checkpointing=training_config.get('use_gradient_checkpointing', True),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1)
        )
        
        # Enable gradient checkpointing
        self.memory_optimizer.enable_gradient_checkpointing(self.model)
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with optional weight decay"""
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        lr = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.training_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with memory optimization"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            # Move data to device
            batch = {
                'fc_matrix': batch['fc_matrix'].to(self.device),
                'timeseries': batch['timeseries'].to(self.device),
                'label': batch['label'].to(self.device)
            }
            
            # Memory-efficient training step
            loss, predicted = self.memory_optimizer.train_step(
                model=self.model,
                batch=batch,
                criterion=self.criterion,
                optimizer=self.optimizer,
                step=step
            )
            
            # Track metrics
            total_loss += loss
            total += batch['label'].size(0)
            correct += predicted.eq(batch['label']).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
            # Print memory stats periodically
            if step % 50 == 0:
                self.memory_optimizer.print_memory_stats(prefix=f"Step {step}: ")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                fc_matrices = batch['fc_matrix'].to(self.device)
                timeseries = batch['timeseries'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(fc_matrices, timeseries)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return avg_loss, accuracy, metrics
    
    def train_fold(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        fold_num: int,
        save_dir: Path
    ) -> Dict:
        """Train on a single fold"""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_num}")
        print(f"{'='*50}")
        
        # Create datasets and loaders
        train_dataset = ADHDDataset(train_data)
        val_dataset = ADHDDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Reset early stopping
        self.early_stopping.reset()
        
        # Training loop
        best_val_acc = 0
        epochs = self.training_config.get('epochs', 100)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Val Metrics - Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = save_dir / f'fold_{fold_num}_best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_metrics': val_metrics
                }, model_path)
                print(f"✓ Saved best model to {model_path}")
                self.best_model_path = model_path
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        return {
            'fold': fold_num,
            'best_val_acc': best_val_acc,
            'final_val_metrics': val_metrics,
            'training_history': self.history.copy()
        }
    
    def run_cv_training(
        self,
        feature_data: pd.DataFrame,
        cv_splits: List[Dict],
        save_dir: Path
    ) -> Dict:
        """Run cross-validation training"""
        print("\n" + "="*70)
        print("STARTING CROSS-VALIDATION TRAINING")
        print("="*70)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        cv_results = []
        
        for fold_info in cv_splits:
            fold_num = fold_info['fold']
            train_idx = fold_info['train_idx']
            val_idx = fold_info['val_idx']
            
            train_data = feature_data.iloc[train_idx]
            val_data = feature_data.iloc[val_idx]
            
            fold_result = self.train_fold(train_data, val_data, fold_num, save_dir)
            cv_results.append(fold_result)
        
        # Aggregate results
        avg_metrics = self._aggregate_cv_results(cv_results)
        
        # Save CV results
        results_path = save_dir / 'cv_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'cv_results': cv_results,
                'average_metrics': avg_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION COMPLETE")
        print(f"Average Validation Accuracy: {avg_metrics['avg_val_acc']:.2f}%")
        print("="*70)
        
        return {'cv_results': cv_results, 'average_metrics': avg_metrics}
    
    def run_loso_training(
        self,
        feature_data: pd.DataFrame,
        loso_splits: List[Dict],
        save_dir: Path
    ) -> Dict:
        """Run Leave-One-Site-Out validation"""
        print("\n" + "="*70)
        print("STARTING LEAVE-ONE-SITE-OUT VALIDATION")
        print("="*70)
        
        loso_dir = save_dir / 'loso'
        loso_dir.mkdir(parents=True, exist_ok=True)
        loso_results = []
        
        for loso_fold in loso_splits:
            fold_num = loso_fold['fold']
            held_out_site = loso_fold['held_out_site']
            train_idx = loso_fold['train_idx']
            test_idx = loso_fold['test_idx']
            
            print(f"\n{'='*50}")
            print(f"LOSO Fold {fold_num}: Holding out site {held_out_site}")
            print(f"{'='*50}")
            
            train_data = feature_data.iloc[train_idx]
            test_data = feature_data.iloc[test_idx]
            
            # Train on all sites except one
            fold_result = self.train_fold(train_data, test_data, fold_num, loso_dir)
            fold_result['held_out_site'] = held_out_site
            loso_results.append(fold_result)
        
        # Aggregate LOSO results
        avg_metrics = self._aggregate_loso_results(loso_results)
        
        # Save LOSO results
        results_path = loso_dir / 'loso_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'loso_results': loso_results,
                'average_metrics': avg_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print("\n" + "="*70)
        print("LOSO VALIDATION COMPLETE")
        print(f"Average Test Accuracy: {avg_metrics['avg_test_acc']:.2f}%")
        print("="*70)
        
        return {'loso_results': loso_results, 'average_metrics': avg_metrics}
    
    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        val_accs = [r['best_val_acc'] for r in cv_results]
        
        # Get final metrics from each fold
        precisions = [r['final_val_metrics']['precision'] for r in cv_results]
        recalls = [r['final_val_metrics']['recall'] for r in cv_results]
        f1s = [r['final_val_metrics']['f1'] for r in cv_results]
        aucs = [r['final_val_metrics']['auc'] for r in cv_results]
        
        return {
            'avg_val_acc': np.mean(val_accs),
            'std_val_acc': np.std(val_accs),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_f1': np.mean(f1s),
            'avg_auc': np.mean(aucs),
            'all_fold_accs': val_accs
        }
    
    def _aggregate_loso_results(self, loso_results: List[Dict]) -> Dict:
        """Aggregate LOSO results"""
        test_accs = [r['best_val_acc'] for r in loso_results]  # In LOSO, val is actually test
        
        precisions = [r['final_val_metrics']['precision'] for r in loso_results]
        recalls = [r['final_val_metrics']['recall'] for r in loso_results]
        f1s = [r['final_val_metrics']['f1'] for r in loso_results]
        aucs = [r['final_val_metrics']['auc'] for r in loso_results]
        
        return {
            'avg_test_acc': np.mean(test_accs),
            'std_test_acc': np.std(test_accs),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_f1': np.mean(f1s),
            'avg_auc': np.mean(aucs),
            'site_results': [{
                'site': r['held_out_site'],
                'accuracy': r['best_val_acc']
            } for r in loso_results]
        }
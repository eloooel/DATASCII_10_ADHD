"""
Simplified Ablation Study Runner

Runs ablation on a single fold to get empirical results quickly.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import GNNSTANHybrid
from configs.baseline_accurate_v6_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE
from utils.data_loader import load_features_and_labels
from training.dataset import ADHDDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


def simple_train_and_eval(model, train_loader, test_loader, device, epochs=10):
    """Simple training and evaluation"""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            fc = batch['fc_matrix'].to(device)
            roi = batch['roi_timeseries'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(fc, roi)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            fc = batch['fc_matrix'].to(device)
            roi = batch['roi_timeseries'].to(device)
            label = batch['label'].to(device)
            
            logits = model(fc, roi)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'confusion_matrix': cm.tolist()
    }


class SimpleGNNOnly(nn.Module):
    """Simplified GNN-only for quick ablation"""
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        from models.gnn import EnhancedGNNBranch
        self.gnn = EnhancedGNNBranch(
            input_dim=4,
            hidden_dims=[128, 64, 32],
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )
        self.n_rois = 200
    
    def forward(self, fc_matrix, roi_timeseries):
        batch_size = fc_matrix.shape[0]
        
        # Create edge index from FC
        fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
        threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
        mask = torch.abs(fc_matrix) > threshold
        
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            sources = sources + b * self.n_rois
            targets = targets + b * self.n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        edge_index = torch.cat(edge_index_list, dim=1)
        
        # Extract node features
        node_features = torch.zeros(batch_size * self.n_rois, 4, device=fc_matrix.device)
        for i in range(batch_size):
            fc = fc_matrix[i]
            degree = torch.sum(torch.abs(fc), dim=1)
            clustering = torch.diagonal(torch.matmul(torch.matmul(fc, fc), fc)) / (degree + 1e-8)
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(fc)
                eigen_centrality = torch.abs(eigenvecs[:, -1])
            except:
                eigen_centrality = degree / (torch.sum(degree) + 1e-8)
            local_eff = torch.mean(torch.abs(fc), dim=1)
            
            start_idx = i * self.n_rois
            end_idx = (i + 1) * self.n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        batch = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        embedding, _ = self.gnn(node_features, edge_index, batch)
        return self.classifier(embedding)


class SimpleSTANOnly(nn.Module):
    """Simplified STAN-only for quick ablation"""
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        from models.stan import EnhancedSTANBranch
        self.stan = EnhancedSTANBranch(
            input_dim=200,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, fc_matrix, roi_timeseries):
        embedding, _ = self.stan(roi_timeseries)
        return self.classifier(embedding)


def main():
    print("="*80)
    print("ABLATION STUDY - QUICK EVALUATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    feature_manifest_path = Path('data/features/feature_manifest.csv')
    manifest_df = pd.read_csv(feature_manifest_path)
    manifest_df = manifest_df[manifest_df['site'].isin(TRAINING_CONFIG_BASELINE['sites'])]
    
    fc_matrices, roi_timeseries, labels, sites = load_features_and_labels(manifest_df)
    print(f"Loaded {len(labels)} subjects")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Single fold: exclude NYU for test
    test_mask = sites == 'NYU'
    train_mask = ~test_mask
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print(f"\nFold: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Create datasets
    train_dataset = ADHDDataset(
        fc_matrices=fc_matrices[train_idx],
        roi_timeseries=roi_timeseries[train_idx],
        labels=labels[train_idx]
    )
    
    test_dataset = ADHDDataset(
        fc_matrices=fc_matrices[test_idx],
        roi_timeseries=roi_timeseries[test_idx],
        labels=labels[test_idx]
    )
    
    # Create data loaders with balanced sampling
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # Test variants
    variants = {
        'GNN Only': SimpleGNNOnly(),
        'STAN Only': SimpleSTANOnly(),
        'Full Hybrid (V6)': GNNSTANHybrid(**MODEL_CONFIG_BASELINE)
    }
    
    results = {}
    
    for variant_name, model in variants.items():
        print(f"\n{'='*80}")
        print(f"Testing: {variant_name}")
        print(f"{'='*80}")
        
        model = model.to(device)
        metrics = simple_train_and_eval(model, train_loader, test_loader, device, epochs=10)
        
        results[variant_name] = metrics
        
        print(f"\nResults:")
        print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"  Specificity: {metrics['specificity']*100:.2f}%")
    
    # Save results
    output_dir = Path('data/ablation_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir / 'ablation_results.json'}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Configuration':<25} {'Accuracy':<12} {'Sensitivity':<15} {'Specificity':<15}")
    print("-"*80)
    for variant_name, metrics in results.items():
        print(f"{variant_name:<25} {metrics['accuracy']*100:>6.2f}%      "
              f"{metrics['sensitivity']*100:>6.2f}%        {metrics['specificity']*100:>6.2f}%")
    print("="*80)
    
    return results


if __name__ == '__main__':
    main()

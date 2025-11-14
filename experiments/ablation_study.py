"""
Ablation Study Framework for GNN-STAN Hybrid Model

Tests model variants to understand contribution of each component:
1. GNN-only baseline
2. STAN-only baseline
3. Hybrid without attention
4. Hybrid without fusion
5. Full hybrid model (baseline)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import GNNSTANHybrid
from models.gnn import EnhancedGNNBranch
from models.stan import EnhancedSTANBranch
from validation import LeaveOneSiteOutValidator, KFoldValidator
from training.dataset import ADHDDataset


class GNNOnlyModel(nn.Module):
    """GNN-only baseline (no temporal processing)"""
    
    def __init__(self, gnn_config: dict, num_classes: int = 2, classifier_dropout: float = 0.5):
        super().__init__()
        self.n_rois = 200
        self.gnn_encoder = EnhancedGNNBranch(input_dim=4, **gnn_config)
        
        gnn_out_dim = gnn_config.get('hidden_dims', [128, 64, 32])[-1]
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_dim, gnn_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.BatchNorm1d(gnn_out_dim // 2),
            nn.Linear(gnn_out_dim // 2, num_classes)
        )
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor) -> torch.Tensor:
        batch_size = fc_matrix.shape[0]
        edge_index = self._create_edge_index_from_fc(fc_matrix).to(fc_matrix.device)
        batch = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        node_features = self._extract_node_features(fc_matrix)
        
        gnn_embedding, _ = self.gnn_encoder(node_features, edge_index, batch)
        logits = self.classifier(gnn_embedding)
        return logits
    
    def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Create edge index from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
        threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
        mask = torch.abs(fc_matrix) > threshold
        
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            sources = sources + b * n_rois
            targets = targets + b * n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        
        return torch.cat(edge_index_list, dim=1)
    
    def _extract_node_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Extract node features from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        node_features = torch.zeros(batch_size * n_rois, 4, device=fc_matrix.device)
        
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
            
            start_idx = i * n_rois
            end_idx = (i + 1) * n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        return node_features


class STANOnlyModel(nn.Module):
    """STAN-only baseline (no graph processing)"""
    
    def __init__(self, stan_config: dict, num_classes: int = 2, classifier_dropout: float = 0.5):
        super().__init__()
        self.n_rois = 200
        self.stan_encoder = EnhancedSTANBranch(input_dim=self.n_rois, **stan_config)
        
        stan_out_dim = stan_config.get('hidden_dim', 128)
        self.classifier = nn.Sequential(
            nn.Linear(stan_out_dim, stan_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.BatchNorm1d(stan_out_dim // 2),
            nn.Linear(stan_out_dim // 2, num_classes)
        )
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor) -> torch.Tensor:
        stan_embedding, _ = self.stan_encoder(roi_timeseries)
        logits = self.classifier(stan_embedding)
        return logits


class HybridNoAttention(nn.Module):
    """Hybrid model without attention mechanisms"""
    
    def __init__(self, gnn_config: dict, stan_config: dict, num_classes: int = 2, 
                 fusion_dim: int = 128, classifier_dropout: float = 0.5):
        super().__init__()
        self.n_rois = 200
        
        # Use simpler GNN without attention
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.conv1 = GCNConv(4, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        
        # Use LSTM without attention
        self.lstm = nn.LSTM(
            input_size=self.n_rois,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Simple fusion (concatenation)
        self.fusion = nn.Linear(32 + 256, fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor) -> torch.Tensor:
        batch_size = fc_matrix.shape[0]
        edge_index = self._create_edge_index_from_fc(fc_matrix).to(fc_matrix.device)
        batch = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        node_features = self._extract_node_features(fc_matrix)
        
        # GNN without attention
        h = torch.relu(self.conv1(node_features, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        from torch_geometric.nn import global_mean_pool
        gnn_embedding = global_mean_pool(h, batch)
        
        # LSTM without attention
        lstm_out, (h_n, c_n) = self.lstm(roi_timeseries)
        stan_embedding = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Simple fusion
        fused = torch.cat([gnn_embedding, stan_embedding], dim=1)
        fused = torch.relu(self.fusion(fused))
        
        logits = self.classifier(fused)
        return logits
    
    def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Create edge index from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
        threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
        mask = torch.abs(fc_matrix) > threshold
        
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            sources = sources + b * n_rois
            targets = targets + b * n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        
        return torch.cat(edge_index_list, dim=1)
    
    def _extract_node_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Extract node features from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        node_features = torch.zeros(batch_size * n_rois, 4, device=fc_matrix.device)
        
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
            
            start_idx = i * n_rois
            end_idx = (i + 1) * n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        return node_features


class HybridNoFusion(nn.Module):
    """Hybrid model without fusion layer (simple concatenation)"""
    
    def __init__(self, gnn_config: dict, stan_config: dict, num_classes: int = 2, 
                 classifier_dropout: float = 0.5):
        super().__init__()
        self.n_rois = 200
        self.gnn_encoder = EnhancedGNNBranch(input_dim=4, **gnn_config)
        self.stan_encoder = EnhancedSTANBranch(input_dim=self.n_rois, **stan_config)
        
        gnn_out_dim = gnn_config.get('hidden_dims', [128, 64, 32])[-1]
        stan_out_dim = stan_config.get('hidden_dim', 128)
        
        # No fusion - direct concatenation
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_dim + stan_out_dim, (gnn_out_dim + stan_out_dim) // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.BatchNorm1d((gnn_out_dim + stan_out_dim) // 2),
            nn.Linear((gnn_out_dim + stan_out_dim) // 2, num_classes)
        )
    
    def forward(self, fc_matrix: torch.Tensor, roi_timeseries: torch.Tensor) -> torch.Tensor:
        batch_size = fc_matrix.shape[0]
        edge_index = self._create_edge_index_from_fc(fc_matrix).to(fc_matrix.device)
        batch = torch.arange(batch_size, device=fc_matrix.device).repeat_interleave(self.n_rois)
        node_features = self._extract_node_features(fc_matrix)
        
        gnn_embedding, _ = self.gnn_encoder(node_features, edge_index, batch)
        stan_embedding, _ = self.stan_encoder(roi_timeseries)
        
        # Simple concatenation (no fusion)
        fused = torch.cat([gnn_embedding, stan_embedding], dim=1)
        logits = self.classifier(fused)
        return logits
    
    def _create_edge_index_from_fc(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Create edge index from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        fc_flat = torch.abs(fc_matrix).view(batch_size, -1)
        threshold = torch.quantile(fc_flat, 0.8, dim=1, keepdim=True).unsqueeze(-1)
        mask = torch.abs(fc_matrix) > threshold
        
        edge_index_list = []
        for b in range(batch_size):
            sources, targets = torch.where(mask[b])
            sources = sources + b * n_rois
            targets = targets + b * n_rois
            edge_index_list.append(torch.stack([sources, targets]))
        
        return torch.cat(edge_index_list, dim=1)
    
    def _extract_node_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """Extract node features from FC matrix"""
        batch_size, n_rois, _ = fc_matrix.shape
        node_features = torch.zeros(batch_size * n_rois, 4, device=fc_matrix.device)
        
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
            
            start_idx = i * n_rois
            end_idx = (i + 1) * n_rois
            node_features[start_idx:end_idx] = torch.stack([
                degree, clustering, eigen_centrality, local_eff
            ], dim=1)
        
        return node_features


class AblationStudy:
    """Run comprehensive ablation study on model variants"""
    
    def __init__(self, model_config: dict, training_config: dict, device: str = 'cuda'):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.variants = {
            'gnn_only': {
                'model_class': GNNOnlyModel,
                'description': 'GNN-only baseline (no temporal processing)',
                'params': {'gnn_config': model_config['gnn'], 'num_classes': 2}
            },
            'stan_only': {
                'model_class': STANOnlyModel,
                'description': 'STAN-only baseline (no graph processing)',
                'params': {'stan_config': model_config['stan'], 'num_classes': 2}
            },
            'hybrid_no_attention': {
                'model_class': HybridNoAttention,
                'description': 'Hybrid without attention mechanisms',
                'params': {'gnn_config': model_config['gnn'], 'stan_config': model_config['stan'], 'num_classes': 2}
            },
            'hybrid_no_fusion': {
                'model_class': HybridNoFusion,
                'description': 'Hybrid without fusion layer',
                'params': {'gnn_config': model_config['gnn'], 'stan_config': model_config['stan'], 'num_classes': 2}
            },
            'full_hybrid': {
                'model_class': GNNSTANHybrid,
                'description': 'Full hybrid model (baseline)',
                'params': {
                    'hidden_dim': model_config['hidden_dim'],
                    'num_classes': 2,
                    'num_heads': model_config['num_heads'],
                    'dropout': model_config['dropout'],
                    'gnn_config': model_config['gnn'],
                    'stan_config': model_config['stan'],
                    'fusion_config': model_config.get('fusion', {}),
                    'classifier_dropout': model_config.get('classifier_dropout', 0.5)
                }
            }
        }
    
    def run_ablation(
        self,
        fc_matrices: np.ndarray,
        roi_timeseries: np.ndarray,
        labels: np.ndarray,
        sites: np.ndarray,
        validation_strategy: str = 'loso',
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Run ablation study on all model variants
        
        Args:
            fc_matrices: Functional connectivity matrices
            roi_timeseries: ROI time series
            labels: Subject labels
            sites: Site information
            validation_strategy: 'loso' or 'kfold'
            output_dir: Directory to save results
        """
        if output_dir is None:
            output_dir = Path('./experiments/ablation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("🔬 ABLATION STUDY: MODEL VARIANT COMPARISON")
        print("="*70)
        
        results = {}
        
        for variant_name, variant_info in self.variants.items():
            print(f"\n{'='*70}")
            print(f"Testing: {variant_name.upper()}")
            print(f"Description: {variant_info['description']}")
            print(f"{'='*70}")
            
            # Create validator
            if validation_strategy == 'loso':
                validator = LeaveOneSiteOutValidator(
                    model_config={'model_class': variant_info['model_class'], **variant_info['params']},
                    training_config=self.training_config
                )
            else:
                validator = KFoldValidator(
                    model_config={'model_class': variant_info['model_class'], **variant_info['params']},
                    training_config=self.training_config,
                    n_splits=5
                )
            
            # Run validation
            variant_results = validator.validate(
                fc_matrices=fc_matrices,
                roi_timeseries=roi_timeseries,
                labels=labels,
                sites=sites
            )
            
            results[variant_name] = {
                'description': variant_info['description'],
                'validation_results': variant_results,
                'summary_metrics': {
                    'accuracy': variant_results['summary']['accuracy_mean'],
                    'accuracy_std': variant_results['summary']['accuracy_std'],
                    'sensitivity': variant_results['summary']['sensitivity_mean'],
                    'specificity': variant_results['summary']['specificity_mean'],
                    'auc': variant_results['summary']['auc_mean'],
                    'f1': variant_results['summary']['f1_mean']
                }
            }
            
            # Save individual variant results
            variant_dir = output_dir / variant_name
            validator.save_results(variant_dir, variant_results)
        
        # Generate comparison report
        comparison = self._generate_comparison(results)
        
        # Save comparison results
        comparison_path = output_dir / 'ablation_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save summary table
        summary_df = self._create_summary_table(results)
        summary_df.to_csv(output_dir / 'ablation_summary.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ ABLATION STUDY COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        print(f"\nSummary:")
        print(summary_df.to_string(index=False))
        
        return {
            'variant_results': results,
            'comparison': comparison,
            'summary_table': summary_df,
            'output_dir': str(output_dir)
        }
    
    def _generate_comparison(self, results: Dict) -> Dict:
        """Generate statistical comparison between variants"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'best_variant': None,
            'best_accuracy': 0.0,
            'improvements': {},
            'statistical_differences': {}
        }
        
        # Find best variant
        for variant_name, variant_data in results.items():
            acc = variant_data['summary_metrics']['accuracy']
            if acc > comparison['best_accuracy']:
                comparison['best_accuracy'] = acc
                comparison['best_variant'] = variant_name
        
        # Calculate improvements relative to baselines
        baselines = ['gnn_only', 'stan_only']
        full_hybrid_acc = results['full_hybrid']['summary_metrics']['accuracy']
        
        for baseline in baselines:
            if baseline in results:
                baseline_acc = results[baseline]['summary_metrics']['accuracy']
                improvement = ((full_hybrid_acc - baseline_acc) / baseline_acc) * 100
                comparison['improvements'][f'full_hybrid_vs_{baseline}'] = {
                    'absolute_diff': full_hybrid_acc - baseline_acc,
                    'relative_improvement': improvement,
                    'baseline_acc': baseline_acc,
                    'full_hybrid_acc': full_hybrid_acc
                }
        
        return comparison
    
    def _create_summary_table(self, results: Dict) -> pd.DataFrame:
        """Create summary table of all variants"""
        rows = []
        
        for variant_name, variant_data in results.items():
            metrics = variant_data['summary_metrics']
            rows.append({
                'Variant': variant_name,
                'Description': variant_data['description'],
                'Accuracy': f"{metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}"
            })
        
        return pd.DataFrame(rows)


def main():
    """Run ablation study from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study on GNN-STAN variants')
    parser.add_argument('--manifest', type=str, required=True, help='Path to feature manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./experiments/ablation_results',
                       help='Output directory for results')
    parser.add_argument('--validation', type=str, choices=['loso', 'kfold'], default='loso',
                       help='Validation strategy')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration (import from main.py)
    from main import MODEL_CONFIG, TRAINING_CONFIG
    
    # Load data
    print("Loading data...")
    manifest = pd.read_csv(args.manifest)
    
    # Load features (simplified - adjust based on your data loading logic)
    from training.dataset import load_features_from_manifest
    fc_matrices, roi_timeseries, labels, sites = load_features_from_manifest(manifest)
    
    # Run ablation study
    study = AblationStudy(MODEL_CONFIG, TRAINING_CONFIG, device=args.device)
    results = study.run_ablation(
        fc_matrices=fc_matrices,
        roi_timeseries=roi_timeseries,
        labels=labels,
        sites=sites,
        validation_strategy=args.validation,
        output_dir=Path(args.output_dir)
    )
    
    print("\n✅ Ablation study complete!")


if __name__ == '__main__':
    main()

"""
Evaluation Module

Performs evaluation of a trained GNN-STAN hybrid model using
preprocessed FC matrices and ROI time series.
Computes predictions, probabilities, embeddings, and comprehensive metrics.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging

from models.gnn_stan_hybrid import GNNSTANHybrid
from training.dataset import ADHDDataset

class ADHDModelEvaluator:
    """Evaluator for GNN-STAN ADHD model"""
    
    def __init__(self, model_path: str, model_config: Dict[str, Any], device: Optional[str] = None):
        """
        Args:
            model_path: Path to saved model checkpoint
            model_config: Dictionary with model hyperparameters
            device: 'cuda' or 'cpu'. Default uses GPU if available
        """
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load model
        self.logger.info("Loading trained GNN-STAN model...")
        self.model = GNNSTANHybrid(model_config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.logger.info("Model loaded and set to evaluation mode.")
    
    def evaluate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray,
                 labels: np.ndarray, sites: np.ndarray = None,
                 batch_size: int = 16) -> Dict[str, Any]:
        """
        Evaluate model on given test data
        
        Args:
            fc_matrices: Functional connectivity matrices (n_subjects, n_rois, n_rois)
            roi_timeseries: ROI time series (n_subjects, n_rois, n_timepoints)
            labels: Ground truth labels (n_subjects,)
            sites: Optional site info (for analysis, not used in model)
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with metrics, predictions, probabilities, embeddings, and attention maps
        """
        self.logger.info(f"Evaluating on {len(labels)} subjects...")
        
        # Create dataset and loader
        test_dataset = ADHDDataset(fc_matrices, roi_timeseries, labels, sites, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_embeddings = {'gnn': [], 'stan': [], 'fused': []}
        all_attention = {'gnn': [], 'stan': []}
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(batch['fc_matrix'], batch['roi_timeseries'], batch['edge_index'], batch['batch'])
                
                # Predictions
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                # Collect
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                
                all_embeddings['gnn'].extend(outputs['gnn_embedding'].cpu().numpy())
                all_embeddings['stan'].extend(outputs['stan_embedding'].cpu().numpy())
                all_embeddings['fused'].extend(outputs['fused_embedding'].cpu().numpy())
                
                all_attention['gnn'].append(outputs['gnn_attention'])
                all_attention['stan'].append(outputs['stan_attention'])
        
        # Convert to arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        true_labels = np.array(all_labels)
        embeddings = {k: np.array(v) for k, v in all_embeddings.items()}
        
        # Compute metrics (reuse BaseValidator function if desired)
        from validation.base_validator import BaseValidator
        metrics = BaseValidator._compute_metrics(self, true_labels, predictions, probabilities)
        
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'embeddings': embeddings,
            'attention_maps': all_attention,
            'true_labels': true_labels
        }
        
        self.logger.info("Evaluation complete.")
        return results

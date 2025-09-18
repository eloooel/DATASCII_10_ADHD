"""
Evaluation Module

Evaluates a trained GNN-STAN hybrid model on a test dataset.
Computes metrics and returns results for analysis and logging.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any
from .training.training_optimization import ADHDDataset
from .models import GNNSTANHybrid
from .validation.base_validator import BaseValidator


class Evaluator:
    """Evaluation wrapper for GNN-STAN model"""

    def __init__(self, model: GNNSTANHybrid, device: str = None):
        self.model = model
        self.device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, fc_matrices: np.ndarray, roi_timeseries: np.ndarray, 
                 labels: np.ndarray, sites: np.ndarray, batch_size: int = 16) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary containing metrics and predictions
        """
        # Create dataset and loader
        dataset = ADHDDataset(fc_matrices, roi_timeseries, labels, sites, augment=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in loader:
                batch_data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = self.model(
                    batch_data['fc_matrix'], batch_data['roi_timeseries'], batch_data['edge_index'], batch_data['batch']
                )
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch_data['label'].cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute predicted labels
        probs = torch.softmax(all_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        true = all_labels.numpy()

        # Use BaseValidator metrics computation
        validator = BaseValidator.__new__(BaseValidator)
        metrics = validator._compute_metrics(true, preds, probs)

        return metrics

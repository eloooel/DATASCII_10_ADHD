"""
Main script to run ADHD GNN-STAN pipeline
- Loads data
- Initializes model
- Trains or loads pretrained weights
- Evaluates model
"""

import torch
from pathlib import Path
from data_loader import ADHDDataLoader
from models import GNNSTANHybrid
from evaluation import Evaluator

# --- Configuration ---
FEATURE_MANIFEST = "data/features/feature_manifest.csv"
DEMOGRAPHICS = "data/demographics.csv"
MODEL_CONFIG = {
    'n_rois': 200,
    'n_classes': 2,
    'gnn': {'hidden_dims': [128, 64, 32], 'dropout': 0.3, 'pool_ratios': [0.8, 0.6]},
    'stan': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3},
    'fusion': {'fusion_dim': 128, 'dropout': 0.3},
    'classifier_dropout': 0.5
}
PRETRAINED_WEIGHTS = None  # Optional path to .pt weights

# --- Load data ---
data_loader = ADHDDataLoader(FEATURE_MANIFEST, DEMOGRAPHICS)
data = data_loader.load_data_for_training()

fc_matrices = data['fc_matrices']
roi_timeseries = data['roi_timeseries']
labels = data['labels']
sites = data['sites']

# --- Initialize model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GNNSTANHybrid(MODEL_CONFIG).to(device)

# Load pretrained weights if available
if PRETRAINED_WEIGHTS:
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))

# --- Evaluate ---
evaluator = Evaluator(model, device=device)
metrics = evaluator.evaluate(fc_matrices, roi_timeseries, labels, sites)

# --- Display results ---
print("\nEvaluation Metrics:")
for k, v in metrics.items():
    if isinstance(v, (float, int)):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

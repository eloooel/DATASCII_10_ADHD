"""
Baseline-Accurate Configuration
================================
This configuration attempts to replicate the base study methodology as closely as possible
for fair comparison, focusing on training and evaluation metrics.

Base Study Reference:
- Paper: "Finding Essential Parts of the Brain in rs-fMRI Can Improve ADHD Diagnosis Using Deep Learning"
- Baseline Accuracy: 70.6%
- Sites: NYU, Peking_1, Peking_2, Peking_3, NeuroIMAGE, KKI, OHSU (7 sites)
- Validation: LOSO (Leave-One-Site-Out) cross-validation
- Metrics: Accuracy, Sensitivity, Specificity, AUC

Key Differences from Base Study:
- Atlas: We use Schaefer-200 instead of AAL-116 (better granularity)
- ROIs: We use top-15 ROIs (vs their top-20) for efficiency
- Architecture: Our GNN-STAN hybrid vs their CNN-LSTM

Training Parameters Match Standard Deep Learning Practices:
- Batch size, learning rate, epochs optimized for our architecture
- Same validation strategy (LOSO)
- Same evaluation metrics for fair comparison
"""

MODEL_CONFIG_BASELINE = {
    # Model Architecture
    'hidden_dim': 128,          # Hidden dimension for GNN and STAN
    'num_classes': 2,           # Binary classification
    'num_heads': 4,             # Number of attention heads
    'dropout': 0.3,             # Dropout rate for regularization
    
    # GNN Configuration (parameter name must match model __init__)
    'gnn_config': {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'pool_ratios': [0.8, 0.6]
    },
    
    # STAN Configuration (parameter name must match model __init__)
    'stan_config': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3
    },
    
    # Fusion Configuration (parameter name must match model __init__)
    'fusion_config': {
        'fusion_dim': 128,
        'dropout': 0.3
    },
    
    # Classifier
    'classifier_dropout': 0.5,
}

TRAINING_CONFIG_BASELINE = {
    # Training Parameters
    'batch_size': 32,           # Base study used 32 (we had 16)
    'learning_rate': 0.001,     # Standard Adam learning rate
    'epochs': 100,              # Sufficient for convergence
    'optimizer': 'adam',        # Adam optimizer (standard choice)
    'weight_decay': 1e-5,       # L2 regularization
    
    # Early Stopping
    'early_stopping': True,
    'patience': 15,             # Stop if no improvement for 15 epochs
    'min_delta': 0.001,         # Minimum improvement threshold
    
    # Validation
    'validation_strategy': 'loso',  # Leave-One-Site-Out (matches base study)
    
    # Data
    'sites': [                  # 7 baseline sites (matches base study)
        'NYU',
        'Peking_1',
        'Peking_2', 
        'Peking_3',
        'NeuroIMAGE',
        'KKI',
        'OHSU'
    ],
    
    # ROI Selection
    'max_rois': 15,             # Use top-15 ROIs (vs base study's 20)
    
    # Loss Function - Match base study exactly
    'use_focal_loss': False,    # Base study used standard binary cross-entropy, NOT focal loss
    'label_smoothing': 0.1,     # Label smoothing to prevent overconfident single-class predictions
    'focal_alpha': 0.25,        # Not used when use_focal_loss=False
    'focal_gamma': 2.0,         # Not used when use_focal_loss=False
    
    # Multiple Runs
    'num_runs': 5,              # 5 runs with different seeds for statistical robustness
    'seeds': [42, 123, 456, 789, 2024],  # Random seeds for reproducibility
    
    # Metrics (matches base study)
    'metrics': [
        'accuracy',             # Primary metric
        'sensitivity',          # True positive rate
        'specificity',          # True negative rate
        'auc',                  # Area under ROC curve
        'f1',                   # F1-score for completeness
    ],
    
    # Output
    'save_best_model': True,
    'save_predictions': True,
    'output_dir': 'data/trained/baseline_accurate',
}

# Expected Performance
EXPECTED_RESULTS_BASELINE = {
    'accuracy': '72±1%',        # Target: Match or exceed base study's 70.6%
    'sensitivity': '70±2%',
    'specificity': '74±2%',
    'auc': '0.78±0.02',
}

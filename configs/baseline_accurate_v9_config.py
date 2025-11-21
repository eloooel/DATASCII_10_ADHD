"""
Baseline-Accurate Configuration V9
===================================
This configuration uses ALL available sites (8 sites total) for maximum data utilization.

Key Features:
- All 8 sites: Brown, KKI, NYU, NeuroIMAGE, OHSU, Peking, Pittsburgh, WashU
- Same class weighting as v6 (weights=[1.0, 4.0])
- 8-fold LOSO cross-validation (one fold per site)
- No oversampling (multiplier=1)
- Label smoothing=0.05

Rationale:
- Maximize training data by using all available sites
- Test if additional sites (Brown, Pittsburgh, WashU) improve performance
- Maintain same hyperparameters as v6 for fair comparison
"""

MODEL_CONFIG_BASELINE = {
    # Model Architecture
    'hidden_dim': 128,
    'num_classes': 2,
    'num_heads': 4,
    'dropout': 0.3,
    
    # GNN Configuration
    'gnn_config': {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'pool_ratios': [0.8, 0.6]
    },
    
    # STAN Configuration
    'stan_config': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3
    },
    
    # Fusion Configuration
    'fusion_config': {
        'fusion_dim': 128,
        'dropout': 0.3
    },
    
    # Classifier
    'classifier_dropout': 0.5,
}

TRAINING_CONFIG_BASELINE = {
    # Training Parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    
    # Early Stopping
    'early_stopping': True,
    'patience': 15,
    'min_delta': 0.001,
    
    # Validation
    'validation_strategy': 'loso',
    
    # Data - ALL 8 sites
    'sites': [
        'Brown',
        'KKI',
        'NYU',
        'NeuroIMAGE',
        'OHSU',
        'Peking',
        'Pittsburgh',
        'WashU'
    ],
    
    # ROI Selection
    'max_rois': 15,
    
    # Loss Function
    'use_focal_loss': False,
    'label_smoothing': 0.05,
    'class_weights': [1.0, 4.0],  # 4x weight for ADHD minority class
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Sampling
    'num_samples_multiplier': 1,  # No oversampling
    
    # Multiple Runs
    'num_runs': 5,
    'seeds': [42, 123, 456, 789, 2024],
    
    # Metrics
    'metrics': [
        'accuracy',
        'sensitivity',
        'specificity',
        'auc',
        'f1',
    ],
    
    # Output
    'save_best_model': True,
    'save_predictions': True,
    'output_dir': 'data/trained/baseline_accurate_v9',
}

EXPECTED_RESULTS_BASELINE = {
    'accuracy': '72±2%',      # Expected improvement with more data
    'sensitivity': '50±5%',
    'specificity': '78±5%',
    'auc': '0.72±0.05',
}

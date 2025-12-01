

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
    
    # Data - 5 sites with Peking merged
    'sites': [
        'NYU',
        'Peking',          # Merged from Peking_1/2/3
        'NeuroIMAGE',
        'KKI',
        'OHSU'
    ],
    
    # ROI Selection
    'max_rois': 15,
    
    # Loss Function - MORE AGGRESSIVE than v6
    'use_focal_loss': False,
    'label_smoothing': 0.05,
    'class_weights': [1.0, 5.0],     # INCREASED from 4.0 to 5.0 (more aggressive)
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Sampling - standard (no oversampling)
    'num_samples_multiplier': 1,
    
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
    'output_dir': 'data/trained/baseline_accurate_v8',
}

EXPECTED_RESULTS_BASELINE = {
    'accuracy': '55±5%',        # May decrease slightly from v6
    'sensitivity': '55±5%',     # Target improvement over v6's 45%
    'specificity': '55±5%',     # May decrease from v6's 58%
    'note': 'More aggressive weighting may improve sensitivity but risk specificity drop'
}

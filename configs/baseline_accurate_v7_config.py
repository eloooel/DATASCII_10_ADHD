

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
    'batch_size': 32,           # Matches base study
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
    
    # Loss Function - EXACT BASE STUDY APPROACH
    'use_focal_loss': False,
    'label_smoothing': 0.0,          # NO label smoothing (base study didn't use)
    'class_weights': [1.0, 1.0],     # NO class weights (base study didn't use)
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Sampling - standard (no oversampling)
    'num_samples_multiplier': 1,     # Balanced mini-batches via WeightedRandomSampler
    
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
    'output_dir': 'data/trained/baseline_accurate_v7',
}

EXPECTED_RESULTS_BASELINE = {
    'accuracy': '75±2%',        # High accuracy from predicting majority class
    'sensitivity': '10±5%',     # Very poor ADHD detection (imbalance problem)
    'specificity': '95±3%',     # High specificity (good at HC detection)
    'note': 'Expected to show severe majority class bias due to 75/25 imbalance'
}

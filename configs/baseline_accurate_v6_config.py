"""
Baseline-Accurate Configuration V6
===================================
This configuration merges Peking_1/2/3 into a single "Peking" site for proper LOSO validation.

Key Changes from V5:
- Peking_1/2/3 merged into "Peking" (5 sites instead of 7)
- Removed 2x oversampling (back to 1x for faster training)
- Batch size reverted to 32 (matches base study)
- Keep class_weights=[1.0, 4.0] and label_smoothing=0.05

Rationale:
- Peking_1 has 0 ADHD subjects, breaking test metrics when used as test fold
- Merging gives: Peking = 191 HC + 54 ADHD (sufficient for evaluation)
- 5 proper LOSO folds: NYU, Peking, NeuroIMAGE, KKI, OHSU
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
    'batch_size': 32,           # Back to base study's batch size
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
    
    # Loss Function
    'use_focal_loss': False,
    'label_smoothing': 0.05,
    'class_weights': [1.0, 4.0],  # 4x weight for ADHD minority class
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Sampling - removed 2x oversampling for faster training
    'num_samples_multiplier': 1,  # Standard sampling (no oversampling)
    
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
    'output_dir': 'data/trained/baseline_accurate_v6',
}

EXPECTED_RESULTS_BASELINE = {
    'accuracy': '70±2%',
    'sensitivity': '50±5%',  # More realistic with 75/25 imbalance
    'specificity': '75±5%',
    'auc': '0.70±0.05',
}

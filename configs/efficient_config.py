"""
Efficient Configuration (COMMENTED OUT FOR NOW)
===============================================
This configuration is designed to achieve the best possible performance
using all available data and optimized hyperparameters.

When ready to use, uncomment the configurations below.
"""

MODEL_CONFIG_EFFICIENT = {
    # Model Architecture - Enhanced
    'hidden_dim': 256,          # Larger hidden dimension for more capacity
    'num_heads': 8,             # More attention heads for richer representations
    'dropout': 0.4,             # Higher dropout for better regularization
    'num_gnn_layers': 3,        # Deeper GNN for better spatial learning
    'num_stan_layers': 3,       # Deeper STAN for better temporal learning
    
    # ROI Selection
    'max_rois': 15,             # Use optimal top-15 ROIs
    
    # Loss Function
    'focal_alpha': 0.3,         # Adjusted for class imbalance
    'focal_gamma': 2.5,         # Stronger focus on hard examples
}

TRAINING_CONFIG_EFFICIENT = {
    # Training Parameters - Optimized
    'batch_size': 8,            # Smaller batch for better gradient estimation
    'learning_rate': 5e-4,      # Lower learning rate for fine-tuning
    'epochs': 150,              # More epochs for thorough training
    'optimizer': 'adamw',       # AdamW with decoupled weight decay
    'weight_decay': 1e-4,       # Stronger L2 regularization
    
    # Learning Rate Scheduler
    'lr_scheduler': 'cosine',   # Cosine annealing for smooth convergence
    'warmup_epochs': 10,        # Warmup period for stability
    
    # Early Stopping
    'early_stopping': True,
    'patience': 20,             # More patience for complex training
    'min_delta': 0.0005,        # Tighter improvement threshold
    
    # Validation
    'validation_strategy': 'loso',  # Leave-One-Site-Out
    
    # Data - Use All Sites
    'sites': [                  # All 10 sites for maximum data
        'NYU',
        'Peking_1',
        'Peking_2',
        'Peking_3',
        'NeuroIMAGE',
        'KKI',
        'OHSU',
        'Brown',
        'Pittsburgh',
        'WashU'
    ],
    
    # Data Augmentation
    'augmentation': True,       # Enable data augmentation
    'augmentation_factor': 1.5, # Augment by 50%
    
    # Multiple Runs
    'num_runs': 10,             # More runs for robust statistics
    'seeds': [42, 123, 456, 789, 2024, 1337, 9876, 5555, 7777, 3141],
    
    # Metrics
    'metrics': [
        'accuracy',
        'sensitivity',
        'specificity',
        'auc',
        'f1',
        'precision',
        'recall',
    ],
    
    # Output
    'save_best_model': True,
    'save_predictions': True,
    'save_attention_weights': True,  # Save for interpretability
    'output_dir': 'data/trained/efficient',
}

EXPECTED_RESULTS_EFFICIENT = {
    'accuracy': '82±2%',        # Target: Significant improvement
    'sensitivity': '80±2%',
    'specificity': '84±2%',
    'auc': '0.87±0.02',
}

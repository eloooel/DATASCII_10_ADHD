# Training Configurations

This directory contains configuration files for different training approaches.

## Available Configurations

### 1. Baseline-Accurate Configuration (`baseline_accurate_config.py`)
**Purpose**: Replicate the base study methodology for fair comparison.

**Key Features**:
- 7 baseline sites (NYU, Peking_1/2/3, NeuroIMAGE, KKI, OHSU)
- Top-15 ROIs for efficiency
- Standard hyperparameters (batch_size=16, lr=0.001, epochs=100)
- LOSO cross-validation
- 5 runs with different seeds
- Same metrics as base study (Accuracy, Sensitivity, Specificity, AUC)

**Expected Performance**: 72±1% accuracy (vs base study's 70.6%)

**Use Case**: 
- Reproducing baseline results
- Fair comparison with published work
- Thesis baseline experiments

### 2. Efficient Configuration (`efficient_config.py`)
**Status**: COMMENTED OUT - Ready for future use

**Purpose**: Achieve best possible performance with optimizations.

**Key Features** (when enabled):
- All 10 sites for maximum data
- Enhanced architecture (hidden_dim=256, num_heads=8)
- Optimized hyperparameters (batch_size=8, lr=5e-4, epochs=150)
- Data augmentation
- 10 runs for robust statistics
- Learning rate scheduling with warmup

**Expected Performance**: 82±2% accuracy

**Use Case**:
- Maximum performance experiments
- Advanced optimization research
- Full dataset utilization

## Usage

### Import Configuration
```python
from configs.baseline_accurate_config import (
    MODEL_CONFIG_BASELINE,
    TRAINING_CONFIG_BASELINE,
    EXPECTED_RESULTS_BASELINE
)
```

### Use in Training Script
```python
# Example training script
from configs.baseline_accurate_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE

# Initialize model with config
model = GNNSTANHybrid(**MODEL_CONFIG_BASELINE)

# Train with config
trainer = Trainer(model, **TRAINING_CONFIG_BASELINE)
results = trainer.train()
```

## Configuration Structure

Each configuration file contains three main dictionaries:

1. **MODEL_CONFIG**: Architecture hyperparameters
   - `hidden_dim`: Hidden dimension size
   - `num_heads`: Number of attention heads
   - `dropout`: Dropout rate
   - `num_gnn_layers`: GNN depth
   - `num_stan_layers`: STAN depth
   - `max_rois`: Number of ROIs to use
   - `focal_alpha`, `focal_gamma`: Focal loss parameters

2. **TRAINING_CONFIG**: Training hyperparameters
   - `batch_size`: Batch size
   - `learning_rate`: Learning rate
   - `epochs`: Maximum epochs
   - `optimizer`: Optimizer choice
   - `weight_decay`: L2 regularization
   - `early_stopping`: Enable early stopping
   - `patience`: Early stopping patience
   - `validation_strategy`: Validation approach
   - `sites`: List of sites to use
   - `num_runs`: Number of independent runs
   - `seeds`: Random seeds for reproducibility
   - `metrics`: Evaluation metrics

3. **EXPECTED_RESULTS**: Target performance metrics

## Reproducing Baseline Study

To reproduce the baseline study results:

1. Ensure all 7 baseline sites are preprocessed and feature-extracted
2. Run ROI-ranking with baseline sites:
   ```bash
   python main.py --stage roi-ranking --site-config baseline --max-rois 15
   ```
3. Create training script using `baseline_accurate_config.py`
4. Run 5 independent training runs with different seeds
5. Report mean±std for all metrics

## Enabling Efficient Configuration

When ready to use the efficient configuration:

1. Ensure all 10 sites are preprocessed
2. Update training scripts to use `MODEL_CONFIG_EFFICIENT` and `TRAINING_CONFIG_EFFICIENT`
3. Run 10 independent training runs

## Notes

- Both configurations use **LOSO cross-validation** for proper evaluation
- Both report the **same core metrics** (accuracy, sensitivity, specificity, AUC) for comparison
- The baseline config prioritizes **reproducibility** and **fair comparison**
- The efficient config prioritizes **maximum performance**
- All configurations include random seeds for **reproducibility**

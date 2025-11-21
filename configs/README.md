# Training Configurations

This directory contains configuration files for different training approaches.

## Available Configurations

### 1. Baseline-Accurate V6 Configuration (`baseline_accurate_v6_config.py`) ⭐ RECOMMENDED
**Purpose**: Adapted methodology for severe class imbalance (3:1 HC/ADHD).

**Key Features**:
- 5 sites with ADHD: NYU, Peking (merged), NeuroIMAGE, KKI, OHSU
- Class weights = [1.0, 4.0] (4× penalty for ADHD misclassification)
- Label smoothing = 0.05
- Balanced mini-batch sampling via WeightedRandomSampler
- 5-fold LOSO cross-validation
- Batch size = 32, no oversampling (multiplier=1)

**Actual Performance**: 
- Overall Accuracy: 54.73%
- Sensitivity: **44.90%** (81% improvement over baseline)
- Specificity: 58.00%

**Use Case**: 
- **FINAL MODEL** for thesis
- Handling severe class imbalance
- Balanced sensitivity/specificity trade-off

### 2. Baseline-Accurate V7 Configuration (`baseline_accurate_v7_config.py`)
**Purpose**: TRUE BASELINE - demonstrates imbalance problem without adaptations.

**Key Features**:
- Same 5 sites as v6
- **NO class weights** [1.0, 1.0]
- **NO label smoothing**
- Proves that base study methodology fails with severe imbalance

**Actual Performance**:
- Overall Accuracy: 64.49%
- Sensitivity: **24.79%** (severe majority class bias)
- Specificity: 77.65%

**Use Case**:
- Baseline comparison to show problem
- Demonstrates need for class weighting
- Thesis problem statement

### 3. Baseline-Accurate V8 Configuration (`baseline_accurate_v8_config.py`)
**Purpose**: Tests more aggressive class weighting.

**Key Features**:
- Same 5 sites as v6
- Class weights = [1.0, 5.0] (5× penalty)
- Label smoothing = 0.05
- Tests if higher weighting improves sensitivity

**Actual Performance**:
- Overall Accuracy: 55.77%
- Sensitivity: 41.98% (worse than v6)
- Specificity: 60.35%

**Use Case**:
- Validation that v6 is near-optimal
- Demonstrates diminishing returns
- Thesis optimization narrative

### 4. Original Baseline Configuration (`baseline_accurate_config.py`) [DEPRECATED]
**Status**: Superseded by v6/v7/v8

**Issues**:
- Used 7 sites including Peking_1 (0 ADHD subjects)
- Caused NaN sensitivity when Peking_1 was test fold
- See v6 for corrected version

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
# V6 (RECOMMENDED - Final model)
from configs.baseline_accurate_v6_config import (
    MODEL_CONFIG_BASELINE,
    TRAINING_CONFIG_BASELINE
)

# V7 (Baseline comparison)
from configs.baseline_accurate_v7_config import (
    MODEL_CONFIG_BASELINE,
    TRAINING_CONFIG_BASELINE
)

# V8 (Optimization validation)
from configs.baseline_accurate_v8_config import (
    MODEL_CONFIG_BASELINE,
    TRAINING_CONFIG_BASELINE
)
```

### Run Training
```bash
# V6 - Final adapted model (RECOMMENDED)
python scripts/train_baseline_accurate_v6.py --num-runs 5

# V7 - True baseline (no adaptations)
python scripts/train_baseline_accurate_v7.py --num-runs 5

# V8 - Aggressive weighting test
python scripts/train_baseline_accurate_v8.py --num-runs 5
```

### Compare Results
```bash
# View individual run results
for run in 1 2 3 4 5; do
    echo "V6 Run $run:"
    cat data/trained/baseline_accurate_v6/run_$run/summary.json | grep -E '(accuracy|sensitivity|specificity)'
done

# Or use comparison script
python scripts/compare_experiments.py \
    --experiments \
        data/trained/baseline_accurate_v6/run_1/results.json \
        data/trained/baseline_accurate_v7/run_1/results.json \
        data/trained/baseline_accurate_v8/run_1/results.json \
    --names "V6 (Adapted)" "V7 (Baseline)" "V8 (Aggressive)"
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

## Key Insights from V6/V7/V8 Experiments

### Class Imbalance Handling
1. **Dataset imbalance severity matters**: 3:1 (ours) vs 1.26:1 (base study) requires different strategies
2. **V7 proves the problem**: Without class weights → 64% accuracy but only 24.79% sensitivity (majority class bias)
3. **V6 shows the solution**: Class weights [1.0, 4.0] → 44.90% sensitivity (+81% improvement)
4. **V8 validates optimization**: Higher weights [1.0, 5.0] → 41.98% sensitivity (diminishing returns)

### Balanced Mini-Batch Sampling
- All configurations use `WeightedRandomSampler` (validation/loso.py line 239)
- Creates ~50/50 HC/ADHD batches from 75/25 dataset
- ADHD samples oversampled 3× during training
- See `verify_balanced_batches.py` for demonstration

### Methodology Contributions
- Complete experimental narrative: problem (v7) → solution (v6) → validation (v8)
- Demonstrates that simple base study replication fails with severe imbalance
- Shows optimal class weight is 4× (not higher)
- 44.9% sensitivity with 58.0% specificity = reasonable trade-off

## Notes

- All configurations use **5-fold LOSO cross-validation** (KKI, NYU, NeuroIMAGE, OHSU, Peking)
- All report **same core metrics** (accuracy, sensitivity, specificity) for fair comparison
- V6 is **recommended final model** for thesis
- V7/V8 provide **experimental validation** of design choices
- All include **random seeds** [42, 123, 456, 789, 2024] for reproducibility
- Peking sites merged to fix 0-ADHD problem (see `scripts/merge_peking_sites.py`)

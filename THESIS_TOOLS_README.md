# Thesis Enhancements - Complete ✅

This directory contains tools added to support thesis requirements beyond the core implementation.

## 📁 Directory Structure

```
experiments/
├── ablation_study.py          # Automated ablation study framework
└── ablation_results/          # Results from ablation experiments (generated)

scripts/
└── compare_experiments.py     # Experiment comparison tool for thesis tables

utils/
└── visualization.py           # Comprehensive visualization utilities
```

## 🔬 Ablation Study Framework

**File:** `experiments/ablation_study.py`

Tests contribution of each model component by systematically removing parts:

### Model Variants Tested:
1. **GNN-only** - Only graph processing (no temporal)
2. **STAN-only** - Only temporal processing (no graph)
3. **Hybrid without attention** - Full hybrid but no attention mechanisms
4. **Hybrid without fusion** - Full hybrid but simple concatenation instead of fusion
5. **Full hybrid** - Complete model (baseline)

### Usage:

```bash
# Run ablation study with LOSO validation
python experiments/ablation_study.py \
    --manifest data/features/feature_manifest.csv \
    --validation loso \
    --output-dir ./experiments/ablation_results

# Run with K-fold validation
python experiments/ablation_study.py \
    --manifest data/features/feature_manifest.csv \
    --validation kfold \
    --output-dir ./experiments/ablation_results
```

### Outputs:
- `ablation_summary.csv` - Summary metrics for all variants
- `ablation_comparison.json` - Statistical comparison results
- Individual variant results in subdirectories
- Automatically identifies best variant and improvements

### Integration with Existing Code:

```python
from experiments.ablation_study import AblationStudy

study = AblationStudy(MODEL_CONFIG, TRAINING_CONFIG)
results = study.run_ablation(
    fc_matrices=fc_matrices,
    roi_timeseries=roi_timeseries,
    labels=labels,
    sites=sites,
    validation_strategy='loso',
    output_dir=Path('./experiments/ablation_results')
)
```

---

## 📊 Visualization Utilities

**File:** `utils/visualization.py`

Publication-quality visualization functions for all aspects of model performance.

### Available Functions:

#### Attention Visualization
- `plot_attention_maps()` - Heatmaps of attention weights (multi-head support)
- `plot_spatial_attention()` - GNN graph attention visualization
- `plot_temporal_attention()` - STAN temporal attention patterns

#### Performance Metrics
- `plot_confusion_matrix()` - With counts and percentages
- `plot_roc_curve()` - ROC curve with AUC
- `plot_precision_recall_curve()` - PR curve for imbalanced data
- `plot_training_curves()` - Loss and accuracy over epochs
- `plot_feature_importance()` - Top ROI rankings

#### Multi-Experiment Analysis
- `plot_site_comparison()` - Performance across sites (LOSO)
- `plot_cross_validation_results()` - Fold-wise performance
- `create_visualization_report()` - Comprehensive report generation

### Usage Examples:

```python
from utils.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_attention_maps,
    create_visualization_report
)

# Confusion matrix
plot_confusion_matrix(
    y_true=labels,
    y_pred=predictions,
    save_path=Path('outputs/confusion_matrix.png')
)

# ROC curve
plot_roc_curve(
    y_true=labels,
    y_proba=probabilities,
    save_path=Path('outputs/roc_curve.png')
)

# Attention maps (from model output)
plot_attention_maps(
    attention_weights=model_outputs['attention_weights'],
    roi_names=roi_names,
    save_path=Path('outputs/attention_heatmap.png')
)

# Generate full visualization report from validation results
create_visualization_report(
    results=validation_results,
    output_dir=Path('outputs/visualizations'),
    experiment_name='full_hybrid_loso'
)
```

### Auto-Generated Visualizations:
The `create_visualization_report()` function automatically creates:
- ✅ Confusion matrix
- ✅ ROC curve
- ✅ Precision-Recall curve
- ✅ Cross-validation bar charts
- ✅ Site comparison (for LOSO)

All saved as high-resolution PNG (300 DPI) for thesis inclusion.

---

## 📈 Experiment Comparison Tool

**File:** `scripts/compare_experiments.py`

Compare multiple experiments and generate thesis-ready tables with statistical tests.

### Features:
- **Comparison tables** in LaTeX, Markdown, and CSV formats
- **Statistical tests** (paired/independent t-tests, Cohen's d effect sizes)
- **Ablation summaries** with automatic variant detection
- **Site-wise comparison** for LOSO experiments
- **Comprehensive reports** with all metrics and significance tests

### Usage:

```bash
# Compare multiple experiments
python scripts/compare_experiments.py \
    --experiments \
        data/trained/experiment1/final_results.json \
        data/trained/experiment2/final_results.json \
        data/trained/experiment3/final_results.json \
    --names "Full Hybrid" "GNN Only" "STAN Only" \
    --baseline "Full Hybrid" \
    --output-dir ./experiments/comparisons
```

### Programmatic Usage:

```python
from scripts.compare_experiments import ExperimentComparer

comparer = ExperimentComparer(output_dir=Path('./experiments/comparisons'))

# Add experiments
comparer.add_experiment(
    name='full_hybrid',
    results_path=Path('data/trained/full_hybrid/loso_results.json'),
    description='Full GNN-STAN hybrid model'
)
comparer.add_experiment(
    name='gnn_only',
    results_path=Path('experiments/ablation_results/gnn_only/validation_summary.json'),
    description='GNN-only baseline'
)

# Generate comprehensive report
comparer.generate_full_report(baseline_experiment='full_hybrid')
```

### Generated Outputs:

1. **comparison_table.csv/tex/md**
   - All experiments side-by-side
   - Metrics with confidence intervals
   - Publication-ready formatting

2. **statistical_tests.csv**
   - p-values for all comparisons
   - Cohen's d effect sizes
   - Significance indicators (✓/✗)
   - Effect size interpretation

3. **ablation_summary.csv/tex**
   - Ordered by model complexity
   - Automatic variant detection
   - LaTeX-ready for thesis

4. **site_comparison.csv**
   - Site-wise accuracy for LOSO
   - Cross-site generalization analysis

5. **summary_report.txt**
   - Human-readable summary
   - All tables in text format
   - Experiment descriptions

---

## 🎯 Integration with Main Pipeline

These tools are designed to work seamlessly with the existing pipeline:

### After Training:
```python
# In main.py after training completes
from utils.visualization import create_visualization_report

# Generate visualizations
create_visualization_report(
    results=training_results,
    output_dir=TRAINED_OUT / 'visualizations',
    experiment_name='experiment_name'
)
```

### For Thesis Comparisons:
```python
from scripts.compare_experiments import ExperimentComparer

comparer = ExperimentComparer()

# Add all experiments
comparer.add_experiment('all_sites_8', Path('data/trained/all_sites/results.json'))
comparer.add_experiment('baseline_sites_5', Path('data/trained/baseline/results.json'))

# Generate comparison
comparer.generate_full_report(baseline_experiment='baseline_sites_5')
```

---

## 📋 Quick Reference

### ROI-Ranking Status ✅
- **Completed:** 37 optimal ROIs identified (80.91% accuracy)
- **Output files:**
  - `data/roi_ranking/roi_rankings.csv` - All 200 ROIs ranked
  - `data/roi_ranking/optimal_rois_top_37.csv` - Optimal subset
  - `data/roi_ranking/incremental_roi_performance.png` - Performance curve
  - `data/roi_ranking/feature_manifest_optimal_rois.csv` - Filtered manifest

### Next Steps for Thesis

1. **Run ablation study:**
   ```bash
   python experiments/ablation_study.py \
       --manifest data/roi_ranking/feature_manifest_optimal_rois.csv \
       --validation loso
   ```

2. **Train both configurations:**
   ```bash
   python main.py --stage training --site-config both
   ```

3. **Generate visualizations:**
   - Automatically done during validation
   - Or run manually with `create_visualization_report()`

4. **Compare experiments:**
   ```bash
   python scripts/compare_experiments.py \
       --experiments <paths_to_results> \
       --names <experiment_names> \
       --baseline <baseline_name>
   ```

---

## 📝 Example Workflow

```bash
# 1. Run ablation study (2-3 hours)
python experiments/ablation_study.py \
    --manifest data/features/feature_manifest.csv \
    --validation loso \
    --output-dir ./experiments/ablation_results

# 2. Train final models (4-6 hours)
python main.py --stage training --site-config both

# 3. Compare all experiments
python scripts/compare_experiments.py \
    --experiments \
        experiments/ablation_results/full_hybrid/validation_summary.json \
        experiments/ablation_results/gnn_only/validation_summary.json \
        experiments/ablation_results/stan_only/validation_summary.json \
        data/trained/all_sites/final_results.json \
        data/trained/baseline_sites/final_results.json \
    --names "Full Hybrid (LOSO)" "GNN Only" "STAN Only" "All 8 Sites" "Baseline 5 Sites" \
    --baseline "Full Hybrid (LOSO)"

# 4. Generate visualizations for key experiments
python -c "
from utils.visualization import create_visualization_report
import json
from pathlib import Path

with open('data/trained/all_sites/final_results.json') as f:
    results = json.load(f)

create_visualization_report(
    results=results,
    output_dir=Path('thesis_figures'),
    experiment_name='all_sites_final'
)
"
```

---

## 🔧 Requirements

All tools use existing dependencies:
- ✅ matplotlib (already in requirements.txt)
- ✅ seaborn (already in requirements.txt)
- ✅ scipy (for statistical tests)
- ✅ pandas, numpy
- ✅ PyTorch (for models)

No additional installations needed!

---

## 📖 Citation

If you use these tools in your thesis, they are part of the GNN-STAN ADHD classification pipeline:

```
Tools for ablation studies, visualization, and experiment comparison
developed as part of the GNN-STAN Hybrid Model for ADHD Classification thesis.
```

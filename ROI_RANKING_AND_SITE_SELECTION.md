# Pipeline Updates - ROI-Ranking and Site Selection

## Overview

The pipeline has been updated to implement the ROI-ranking feature selection methodology from your baseline study and support multiple experimental configurations for fair comparison.

## Key Additions

### 1. ROI-Ranking Feature Selection (`feature_extraction/roi_ranking.py`)

**Purpose**: Identify the most discriminative brain regions for ADHD classification

**Methodology** (from baseline study):
- **Phase 1**: Individual ROI Evaluation
  - Train 200 separate models, each using only ONE ROI's features
  - Use LOSO (Leave-One-Site-Out) cross-validation for each
  - Rank all 200 ROIs by their individual classification accuracy
  
- **Phase 2**: Incremental Combination
  - Start with top-1 ranked ROI
  - Incrementally add next best ROI (top-2, top-3, ..., top-50)
  - Plot performance curve to identify optimal ROI subset
  - Baseline study found top-20 ROIs achieved 70.6% accuracy

**Output**:
- `roi_rankings.csv`: All 200 ROIs ranked by discriminative power
- `incremental_roi_results.csv`: Performance for each ROI subset size
- `optimal_rois_top_N.csv`: Final selected ROI subset
- `feature_manifest_optimal_rois.csv`: Filtered manifest for training

**Usage**:
```bash
# Run ROI-ranking separately
python main.py --stage roi-ranking --max-rois 50

# Skip ROI-ranking (use all 200 ROIs)
python main.py --stage training --skip-roi-ranking
```

### 2. Site Selection (`utils/site_selector.py`)

**Purpose**: Enable two experimental configurations for comprehensive evaluation

**Configurations**:

1. **All 8 Sites** (759 subjects)
   - Brown, NYU, NeuroIMAGE, Peking_1, Peking_2, Peking_3, Pittsburgh, WashU
   - Maximum data utilization
   - Tests model's ability to generalize across diverse sites

2. **Baseline-Comparable** (5 sites)
   - NYU, Peking_1, Peking_2, Peking_3, NeuroIMAGE
   - Matches baseline study's site selection philosophy
   - Fair comparison with published results

**Site Mapping** (your data → baseline study):
- NYU → NYU ✅
- Peking_1/2/3 → Peking (combined) ✅
- NeuroIMAGE → NI (likely) ✅
- Brown, Pittsburgh, WashU → Not in baseline (excluded for comparability)

**Output**:
- `feature_manifest_all_sites.csv`: All 8 sites
- `feature_manifest_baseline_sites.csv`: 5 baseline-comparable sites
- `experiment_configurations.json`: Configuration metadata

**Usage**:
```bash
# Run both experiments
python main.py --stage training --site-config both

# Run only all-sites
python main.py --stage training --site-config all

# Run only baseline-comparable
python main.py --stage training --site-config baseline
```

### 3. Updated Main Pipeline (`main.py`)

**New Command-Line Options**:
```bash
--stage roi-ranking        # Run only ROI-ranking feature selection
--site-config [all|baseline|both]  # Select site configuration
--skip-roi-ranking         # Skip ROI-ranking, use all 200 ROIs
--max-rois N               # Maximum ROIs to test in incremental phase (default: 50)
```

**Complete Pipeline Flow**:
1. **Preprocessing** → 759 subjects across 8 sites ✅ (DONE)
2. **Feature Extraction** → Extract from all 200 Schaefer ROIs
3. **Site Configuration** → Create filtered datasets (all vs baseline)
4. **ROI-Ranking** → Identify optimal ROI subset (optional)
5. **Data Splitting** → Create CV/LOSO splits for each configuration
6. **Training** → Train models for each configuration separately
7. **Evaluation** → Compare results across experiments

## Recommended Experimental Workflow

### Experiment 1: Baseline-Comparable (for fair comparison)
```bash
# 1. Run full pipeline with baseline sites and ROI-ranking
python main.py --stage full --site-config baseline

# This will:
# - Use 5 baseline-comparable sites
# - Perform ROI-ranking to find optimal subset
# - Train model on optimal ROIs
# - Generate LOSO validation results comparable to baseline study
```

### Experiment 2: Maximum Data (all 8 sites)
```bash
# 2. Run with all sites to leverage maximum data
python main.py --stage training --site-config all

# This will:
# - Use all 759 subjects from 8 sites
# - Test if additional sites improve generalization
# - Show benefits of larger, more diverse dataset
```

### Experiment 3: ROI Subset Analysis
```bash
# 3. Compare different ROI subsets
python main.py --stage roi-ranking --max-rois 100

# Then manually select different ROI counts to test:
# - Top 10, Top 20, Top 50, Top 100, All 200
```

### Experiment 4: Run Both Configurations
```bash
# 4. Comprehensive comparison
python main.py --stage full --site-config both

# This will:
# - Run ROI-ranking on full dataset
# - Create splits for both configurations
# - Train separate models for:
#   a) All 8 sites with optimal ROIs
#   b) Baseline 5 sites with optimal ROIs
# - Compare generalization across configurations
```

## Expected Outputs

```
data/
├── roi_ranking/
│   ├── roi_rankings.csv                    # All 200 ROIs ranked
│   ├── top_50_rois.csv                     # Top 50 for reference
│   ├── incremental_roi_results.csv         # Performance curve data
│   ├── incremental_roi_performance.png     # Visualization
│   ├── optimal_rois_top_N.csv              # Selected ROIs
│   ├── feature_manifest_optimal_rois.csv   # Filtered manifest
│   └── roi_ranking_summary.json            # Summary statistics
│
├── site_configs/
│   ├── feature_manifest_all_sites.csv      # 759 subjects (8 sites)
│   ├── feature_manifest_baseline_sites.csv # ~500+ subjects (5 sites)
│   └── experiment_configurations.json      # Config metadata
│
├── splits/
│   ├── all_sites/
│   │   └── splits.json                     # Splits for 8-site experiment
│   └── baseline/
│       └── splits.json                     # Splits for 5-site experiment
│
└── trained/
    ├── all_sites/
    │   ├── cv/                             # Cross-validation results
    │   ├── loso/                           # LOSO validation results
    │   └── final_results.json
    └── baseline/
        ├── cv/
        ├── loso/
        └── final_results.json
```

## Key Benefits

### 1. Methodological Rigor
- ✅ ROI-ranking follows established methodology from baseline study
- ✅ LOSO validation handles site effects properly
- ✅ Incremental evaluation prevents cherry-picking optimal ROI count

### 2. Fair Comparison
- ✅ Baseline-comparable configuration enables direct comparison with published results
- ✅ Same validation strategy (LOSO) as baseline study
- ✅ Same feature selection approach (ROI-ranking)

### 3. Comprehensive Evaluation
- ✅ All-sites configuration tests generalizability with maximum data
- ✅ Multiple experiments reveal impact of site selection on performance
- ✅ ROI subset analysis provides interpretable biomarkers

### 4. Reproducibility
- ✅ All configurations saved and documented
- ✅ Site filtering is explicit and traceable
- ✅ ROI selection is data-driven and automated

## Next Steps

1. **Run Feature Extraction** (if not done):
   ```bash
   python main.py --stage features
   ```

2. **Run ROI-Ranking**:
   ```bash
   python main.py --stage roi-ranking --max-rois 50
   ```
   - This will take ~2-4 hours depending on hardware
   - Creates optimal ROI subset for training

3. **Run Both Experimental Configurations**:
   ```bash
   python main.py --stage training --site-config both
   ```
   - Trains on both all-sites and baseline configurations
   - Enables comprehensive comparison

4. **Analyze Results**:
   - Compare LOSO accuracies across configurations
   - Identify which brain regions (ROIs) are most discriminative
   - Evaluate if additional sites improve or hurt generalization
   - Compare your results with baseline study benchmarks

## Troubleshooting

**If ROI-ranking takes too long**:
- Reduce `--max-rois` to 30 or 20 for faster initial testing
- Use `--skip-roi-ranking` to bypass and use all 200 ROIs

**If memory issues occur**:
- ROI-ranking uses simple Random Forest classifiers (lightweight)
- Site filtering reduces data size for baseline configuration
- Incremental evaluation processes one ROI subset at a time

**If sites don't match baseline exactly**:
- This is expected - OHSU and KKI not in your dataset
- Baseline configuration uses best available proxy sites
- All-sites experiment leverages your unique advantage (more data)

## Citation

This implementation follows the methodology from:
- Original ADHD-200 preprocessing protocols
- ROI-ranking feature selection from baseline studies
- LOSO cross-validation for multi-site neuroimaging

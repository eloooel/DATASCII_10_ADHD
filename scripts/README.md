# Scripts Directory

This directory contains utility scripts for specific tasks in the ADHD classification pipeline.

## Training Scripts

### `train_baseline_accurate_v6.py` ⭐ RECOMMENDED
**Purpose:** Final adapted model with class weighting for severe imbalance.

**Configuration:**
- 5 sites: NYU, Peking, NeuroIMAGE, KKI, OHSU
- Class weights = [1.0, 4.0]
- Label smoothing = 0.05
- Batch size = 32

**Results:** 54.73% acc, 44.90% sens, 58.00% spec

**Usage:**
```bash
python scripts/train_baseline_accurate_v6.py --num-runs 5
```

### `train_baseline_accurate_v7.py`
**Purpose:** True baseline (no adaptations) to demonstrate imbalance problem.

**Configuration:**
- Same 5 sites as v6
- NO class weights [1.0, 1.0]
- NO label smoothing

**Results:** 64.49% acc, 24.79% sens (severe bias), 77.65% spec

**Usage:**
```bash
python scripts/train_baseline_accurate_v7.py --num-runs 5
```

### `train_baseline_accurate_v8.py`
**Purpose:** Test more aggressive class weighting.

**Configuration:**
- Same 5 sites as v6
- Class weights = [1.0, 5.0]
- Label smoothing = 0.05

**Results:** 55.77% acc, 41.98% sens, 60.35% spec (diminishing returns)

**Usage:**
```bash
python scripts/train_baseline_accurate_v8.py --num-runs 5
```

## Site Preparation Scripts

### `merge_peking_sites.py`
**Purpose:** Merge Peking_1/2/3 into single "Peking" site for proper LOSO.

**Why:** Peking_1 had 0 ADHD subjects causing NaN sensitivity when used as test fold.

**Usage:**
```bash
python scripts/merge_peking_sites.py
```

**Output:** Updates `data/features/feature_manifest.csv`, creates backup.

## Preprocessing Scripts

### `run_washu_preprocessing.py`
Preprocess only WashU subjects (60 subjects using first run per subject).

**Usage:**
```bash
# Run with parallel processing (default)
python scripts/run_washu_preprocessing.py

# Run sequentially (for debugging)
python scripts/run_washu_preprocessing.py --no-parallel
```

**Purpose:** One-time preprocessing of WashU dataset that was added after initial pipeline setup.

## Analysis Scripts

### `compare_experiments.py`
**Purpose:** Generate thesis-ready comparison tables with statistical tests.

**Usage:**
```bash
python scripts/compare_experiments.py \
    --experiments \
        data/trained/baseline_accurate_v6/run_1/results.json \
        data/trained/baseline_accurate_v7/run_1/results.json \
        data/trained/baseline_accurate_v8/run_1/results.json \
    --names "V6 (Adapted)" "V7 (Baseline)" "V8 (Aggressive)" \
    --output-dir ./experiments/comparisons
```

### Verification Scripts

**`verify_balanced_batches.py`** - Demonstrates WeightedRandomSampler creating ~50/50 batches from 75/25 dataset
**`check_v9_distribution.py`** - Analyzes distribution when using all 8 sites (shows 3 sites have 0 ADHD)
**`check_distribution.py`** - General dataset distribution analysis

## Build Scripts

- `build.sh` - Docker build script
- `run-*.sh` - Various pipeline execution scripts

## Notes

- For main pipeline execution, use `python main.py`
- For training experiments, use `scripts/train_baseline_accurate_v{6,7,8}.py`
- For WashU-specific preprocessing, use `scripts/run_washu_preprocessing.py`
- For diagnostics, see `debug/washu_diagnostics.py`
- For balanced mini-batch demonstration, see `verify_balanced_batches.py`

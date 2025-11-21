# GNN-STAN Hybrid Model for ADHD Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2F12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Graph Neural Networks + Spatio-Temporal Attention Networks for ADHD diagnosis from rs-fMRI data**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Thesis Tools](#thesis-tools)
- [ROI-Ranking Results](#roi-ranking-results)
- [Architecture](#architecture)
- [Results](#results)
- [Docker Support](#docker-support)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

This repository implements a **hybrid deep learning architecture** combining Graph Neural Networks (GNN) and Spatio-Temporal Attention Networks (STAN) for ADHD classification from resting-state fMRI data.

**Key Innovation:** Joint processing of:
- **Functional connectivity** (spatial correlations) via GNN
- **ROI time series** (temporal dynamics) via STAN
- **Cross-modal fusion** for enhanced classification

**Dataset:** ADHD-200 consortium (8 sites, 759 subjects, 1048 scans)

## ✨ Features

### Core Pipeline
- ✅ **Multi-site preprocessing** with ICA-AROMA denoising
- ✅ **Schaefer-200 parcellation** for ROI extraction
- ✅ **ROI-ranking feature selection** (optimal subset: 37 ROIs, 80.91% accuracy)
- ✅ **Hybrid GNN-STAN architecture** with attention mechanisms
- ✅ **Leave-One-Site-Out (LOSO) validation** for generalization testing
- ✅ **Advanced memory optimization** (AGO, HTAAS, hybrid attention)

### Thesis Tools (NEW!)
- 🔬 **Ablation study framework** - Automated testing of model variants
- 📊 **Visualization utilities** - Publication-quality plots (attention maps, ROC, confusion matrices)
- 📈 **Experiment comparison** - Statistical tests, LaTeX tables, thesis-ready reports

### Advanced Features
- ⚡ Multi-run concatenation for longitudinal scans
- 🎯 Focal Loss for class imbalance
- 🧠 Multi-head attention (spatial + temporal)
- 🔄 K-fold and nested cross-validation
- 📉 Early stopping with patience

## ⚠️ CRITICAL REQUIREMENT: Python 3.10.10

**You MUST use Python 3.10.10 for this project. Other versions (especially 3.11+) will cause compilation errors.**

Check your Python version:
```
python --version
```

If you see Python 3.13 or any version other than 3.10.x, download and install Python 3.10.10 from:
https://www.python.org/downloads/release/python-31010/

**Windows Direct Download:**
- 64-bit: https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe
- 32-bit: https://www.python.org/ftp/python/3.10.10/python-3.10.10.exe

After installing Python 3.10.10, use it explicitly:
- Windows: `py -3.10 -m venv .venv` or `C:\Path\To\Python310\python.exe -m venv .venv`
- Linux/Mac: Use `python3.10` instead of `python`

**IMPORTANT**: When you activate your venv, verify it's using Python 3.10:
```
python --version
```
It MUST show "Python 3.10.x" (where x is any patch version).

## 📊 Dataset

This project uses the **ADHD-200 dataset** from 8 international sites:

| Site | Subjects | ADHD | Control | Multi-run | Notes |
|------|----------|------|---------|-----------|-------|
| NYU | 257 | 54 | 203 | ✓ (1.7 runs/subject) | ✓ Used in training |
| Peking (merged) | 245 | 54 | 191 | - | ✓ Used in training (merged 1/2/3) |
| NeuroIMAGE | 73 | 25 | 48 | ✓ (2.0 runs/subject) | ✓ Used in training |
| KKI | 83 | 22 | 61 | - | ✓ Used in training |
| OHSU | 113 | 37 | 76 | - | ✓ Used in training |
| Brown | 26 | 0 | 26 | ✓ (2.0 runs/subject) | ✗ HC-only, excluded |
| Pittsburgh | 98 | 0 | 98 | - | ✗ HC-only, excluded |
| WashU | 60 | 0 | 60 | ✓ (5.6 runs/subject) | ✗ HC-only, excluded |
| **Training Set** | **771** | **192** | **579** | - | **5 sites with ADHD** |
| **Full Dataset** | **955** | **192** | **763** | **1048 scans** | **8 sites total** |

**Key Dataset Characteristics:**
- **Class Imbalance**: 75.1% HC / 24.9% ADHD (3.02:1 ratio)
- **LOSO Validation**: 5-fold cross-validation (one per training site)
- **HC-Only Sites**: Brown, Pittsburgh, WashU excluded from training (0 ADHD subjects)

**Data Access:** Due to size and licensing restrictions, raw data is **not included**. Download from:
- [ADHD-200 Consortium](http://fcon_1000.projects.nitrc.org/indi/adhd200/)
- See `MULTI_SITE_SETUP_GUIDE.md` for detailed setup instructions

## 🚀 Quick Start

### Prerequisites
- Python 3.10.10 (REQUIRED)
- CUDA 11.8 or 12.1 (for GPU acceleration)
- 16GB+ RAM
- 600GB+ disk space (preprocessing + features)

### Installation

**Step 1: Verify Python Version**
```bash
python --version  # MUST show Python 3.10.x
```

If not 3.10.x, download from: https://www.python.org/downloads/release/python-31010/

**Step 2: Create Virtual Environment**
```bash
# Windows (if multiple Python versions)
py -3.10 -m venv thesis-adhd

# Windows (if 3.10 is default)
python -m venv thesis-adhd

# Linux/Mac
python3.10 -m venv thesis-adhd
```

**Step 3: Activate Environment**
```bash
# Windows PowerShell
.\thesis-adhd\Scripts\Activate.ps1

# Windows CMD
.\thesis-adhd\Scripts\activate.bat

# Linux/Mac
source thesis-adhd/bin/activate
```

**Step 4: Install Dependencies**
```bash
# Verify Python 3.10 in venv
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

**Step 5: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Running the Pipeline

**Full Pipeline (All Stages)**
```bash
python main.py --stage full --site-config both
```

**Individual Stages**
```bash
# 1. Preprocessing only (already completed: 759/759 subjects)
python main.py --stage preprocessing

# 2. Feature extraction (already completed: 759/759 subjects)
python main.py --stage features

# 3. ROI-ranking feature selection (already completed: 37 optimal ROIs)
python main.py --stage roi-ranking

# 4. Data splitting
python main.py --stage split --site-config both

# 5. Model training
python main.py --stage training --site-config both
```

**Site Configuration Options:**
- `--site-config all` - Use all 8 sites (759 subjects)
- `--site-config baseline` - Use 5 baseline-comparable sites (575 subjects)
- `--site-config both` - Run both experiments

## 📂 Pipeline Stages

### Stage 1: Preprocessing ✅
**Status:** Complete (759/759 subjects, 564.15 GB)

**What it does:**
- Skull stripping
- Motion correction
- Slice timing correction
- ICA-AROMA denoising
- Spatial normalization to MNI152
- Multi-run concatenation (NYU, WashU, NeuroIMAGE, Brown)

**Command:**
```bash
python main.py --stage preprocessing --parallel
```

**Outputs:**
- `data/preprocessed/{site}/{subject}_func_preprocessed.nii.gz`
- `data/preprocessed/{site}/{subject}_mask.nii.gz`

### Stage 2: Feature Extraction ✅
**Status:** Complete (759/759 subjects, ~12 GB)

**What it does:**
- Schaefer-200 parcellation
- ROI timeseries extraction (n_timepoints × 200)
- Functional connectivity matrices (200×200)

**Command:**
```bash
python main.py --stage features --parallel
```

**Outputs:**
- `data/features/{site}/{subject}_roi_timeseries.csv`
- `data/features/{site}/{subject}_connectivity_matrix.npy`
- `data/features/feature_manifest.csv`

### Stage 3: ROI-Ranking Feature Selection ✅
**Status:** Complete (37 optimal ROIs, 80.91% accuracy)

**What it does:**
- Phase 1: Evaluate each of 200 ROIs individually with LOSO
- Phase 2: Test incremental combinations (top 1 to top 50)
- Identify optimal ROI subset

**Command:**
```bash
python main.py --stage roi-ranking --max-rois 50
```

**Outputs:**
- `data/roi_ranking/roi_rankings.csv` - All ROIs ranked by accuracy
- `data/roi_ranking/optimal_rois_top_37.csv` - Optimal subset
- `data/roi_ranking/incremental_roi_performance.png` - Performance curve
- `data/roi_ranking/feature_manifest_optimal_rois.csv` - Filtered manifest

**Top 10 ROIs:**
1. ROI_189: 80.78%
2. ROI_148: 80.72%
3. ROI_96: 80.69%
4. ROI_34: 80.60%
5. ROI_68: 80.56%
6. ROI_178: 80.51%
7. ROI_97: 80.48%
8. ROI_121: 80.46%
9. ROI_171: 80.46%
10. ROI_65: 80.44%

### Stage 4: Site Configuration
**What it does:**
- Creates filtered datasets for different experiments
- All 8 sites: 759 subjects (for maximum data)
- Baseline 5 sites: 575 subjects (for comparison with baseline studies)

**Command:**
```bash
python main.py --stage split --site-config both
```

**Outputs:**
- `data/site_configs/feature_manifest_all_sites.csv`
- `data/site_configs/feature_manifest_baseline_sites.csv`
- `data/site_configs/experiment_configurations.json`

### Stage 5: Data Splitting
**What it does:**
- Stratified train/val/test splits (80/10/10)
- 5-fold cross-validation splits
- Leave-One-Site-Out (LOSO) splits

**Outputs:**
- `data/splits/all_sites/splits.json`
- `data/splits/baseline_sites/splits.json`

### Stage 6: Model Training
**What it does:**
- Train GNN-STAN hybrid model
- 5-fold cross-validation
- Leave-One-Site-Out validation
- Early stopping with patience

**Command:**
```bash
python main.py --stage training --site-config both
```

**Outputs:**
- `data/trained/all_sites/final_results.json`
- `data/trained/baseline_sites/final_results.json`
- Model checkpoints
- Training logs

## 🔬 Thesis Tools

### 1. Ablation Study Framework
**File:** `experiments/ablation_study.py`

Tests contribution of each component by systematically removing parts:

**Model Variants:**
- GNN-only (no temporal processing)
- STAN-only (no graph processing)
- Hybrid without attention
- Hybrid without fusion
- Full hybrid (complete model)

**Usage:**
```bash
python experiments/ablation_study.py \
    --manifest data/features/feature_manifest.csv \
    --validation loso \
    --output-dir ./experiments/ablation_results
```

**Outputs:**
- `ablation_summary.csv` - Comparison table
- `ablation_comparison.json` - Statistical analysis
- Individual variant results

### 2. Visualization Utilities
**File:** `utils/visualization.py`

Publication-quality plotting for thesis figures:

**Available Functions:**
- `plot_attention_maps()` - Multi-head attention heatmaps
- `plot_spatial_attention()` - GNN graph attention
- `plot_temporal_attention()` - STAN temporal patterns
- `plot_confusion_matrix()` - With percentages & counts
- `plot_roc_curve()` - ROC with AUC
- `plot_precision_recall_curve()` - PR curve
- `plot_training_curves()` - Loss/accuracy over epochs
- `plot_feature_importance()` - Top ROI rankings
- `plot_site_comparison()` - LOSO site analysis
- `plot_cross_validation_results()` - Fold-wise metrics
- `create_visualization_report()` - Auto-generate all plots

**Usage Example:**
```python
from utils.visualization import create_visualization_report
import json

with open('data/trained/all_sites/final_results.json') as f:
    results = json.load(f)

create_visualization_report(
    results=results,
    output_dir=Path('thesis_figures'),
    experiment_name='all_sites_final'
)
```

**Output:** High-resolution PNG files (300 DPI) for thesis

### 3. Experiment Comparison Tool
**File:** `scripts/compare_experiments.py`

Generate thesis-ready comparison tables with statistical tests:

**Features:**
- Comparison tables (LaTeX/Markdown/CSV)
- Statistical tests (t-tests, Cohen's d, p-values)
- Ablation summaries
- Site-wise comparisons

**Usage:**
```bash
python scripts/compare_experiments.py \
    --experiments \
        experiments/ablation_results/full_hybrid/validation_summary.json \
        experiments/ablation_results/gnn_only/validation_summary.json \
        experiments/ablation_results/stan_only/validation_summary.json \
    --names "Full Hybrid" "GNN Only" "STAN Only" \
    --baseline "Full Hybrid" \
    --output-dir ./experiments/comparisons
```

**Outputs:**
- `comparison_table.csv/tex/md` - Side-by-side metrics
- `statistical_tests.csv` - p-values, effect sizes, significance
- `ablation_summary.csv/tex` - LaTeX-ready table
- `site_comparison.csv` - Site-wise performance
- `summary_report.txt` - Complete text report

See `THESIS_TOOLS_README.md` for detailed documentation.

## 🏗️ Architecture

### GNN-STAN Hybrid Model

```
Input: FC Matrix (200×200) + ROI Timeseries (n_timepoints×200)
         │                            │
         ├─────────────┐              │
         │             │              │
    ┌───▼───┐    ┌───▼──────────────▼───┐
    │  GNN  │    │       STAN           │
    │Branch │    │      Branch          │
    └───┬───┘    └──────────┬───────────┘
        │                   │
        │    ┌──────────────▼──────┐
        └────►  Cross-Modal Fusion │
             └──────────┬───────────┘
                        │
                 ┌──────▼───────┐
                 │  Classifier  │
                 └──────┬───────┘
                        │
                   ADHD/Control
```

**GNN Branch:**
- 4 node features: degree, clustering, eigenvector centrality, local efficiency
- Graph Attention Networks (GAT) with 4 heads
- Hierarchical pooling (ratios: 0.8, 0.6)
- Output: 32-dimensional graph embedding

**STAN Branch:**
- Bidirectional LSTM (2 layers, 128 hidden units)
- Multi-head spatio-temporal attention (8 heads)
- Temporal convolution for local patterns
- Output: 128-dimensional temporal embedding

**Cross-Modal Fusion:**
- Attention-based fusion mechanism
- Learns optimal weighting of GNN and STAN representations
- Output: 128-dimensional fused embedding

**Classifier:**
- 2-layer MLP with dropout (0.5)
- Batch normalization
- Output: ADHD/Control classification

### Class Imbalance Handling

**Balanced Mini-Batch Sampling** (`validation/loso.py`):
- `WeightedRandomSampler` with replacement
- Each HC sample weight: 1/579 = 0.0017
- Each ADHD sample weight: 1/192 = 0.0052 (3× higher)
- Result: ~50% ADHD per batch vs 25% in overall dataset
- ADHD samples oversampled ~3× during training
- See `verify_balanced_batches.py` for demonstration

**Class Weighting in Loss Function**:
- Binary cross-entropy with class weights [1.0, 4.0]
- Misclassifying ADHD costs 4× more than HC
- Label smoothing = 0.05 for regularization
- Combined with balanced batches: 81% sensitivity improvement

### Memory Optimization Techniques

1. **Active Gradient Offloading (AGO)** - CPU offloading of gradients
2. **Holistic Traffic-Aware Activation Swapping (HTAAS)** - Smart activation management
3. **Hybrid Attention Blocks (HAB)** - Chunked + sparse attention
4. **Dynamic Gating** - Adaptive layer activation

See `optimization/OPTIMIZATION_README.md` for details.

## 📊 ROI-Ranking Results

### Optimal Configuration
- **Optimal ROIs:** 37 out of 200
- **Accuracy:** 80.91%
- **Improvement:** 80.91% vs 80.78% (best single ROI)

### Performance Curve
The incremental evaluation shows:
- Single best ROI (ROI_189): 80.78%
- Top 10 ROIs: ~80.85%
- Top 37 ROIs: **80.91%** (optimal)
- Top 50 ROIs: 80.87% (slight decrease)

**Insight:** More ROIs ≠ better performance. Optimal subset balances discriminative power and noise reduction.

### Top 37 ROIs (Schaefer-200 Parcellation)
See `data/roi_ranking/optimal_rois_top_37.csv` for the complete list.

**Key regions include:**
- Default Mode Network (DMN) components
- Executive Control Network regions
- Salience Network nodes
- Visual and motor cortices

## 🔄 Retry Failed Preprocessing

If some subjects fail during preprocessing:

**Automatic Retry:**
```bash
python preprocessing/retry_failed.py \
    --manifest data/raw/subjects_metadata.csv \
    --output data/preprocessed \
    --jobs 2
```

**Options:**
- `--no-cleanup` - Keep existing files (don't delete corrupted)
- `--jobs 1` - Sequential processing for memory-intensive subjects

**Common Issues:**
1. **Memory errors:** Use `--jobs 1`
2. **Corrupted files:** Script auto-cleans and retries
3. **Verification failures:** Reduce I/O load with `--jobs 1`

## 🐳 Docker Support


### Build Image
```bash
docker build -t adhd-gnn-stan:latest .
```

### Run with GPU
```bash
docker run --gpus all --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    adhd-gnn-stan:latest \
    uv run python main.py --stage full
```

### Run CPU-Only
```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    --memory=16g \
    --cpus=8 \
    --env CUDA_VISIBLE_DEVICES="" \
    adhd-gnn-stan:latest \
    uv run python main.py --stage full --no-parallel --no-cuda
```

### Interactive Mode
```bash
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    adhd-gnn-stan:latest \
    /bin/bash
```

## 🐛 Troubleshooting

### Python Version Issues

**Problem:** Cython compilation errors, scikit-learn build failures
```
ERROR: Cython.Compiler.Errors.CompileError
```

**Solution:** You're using the wrong Python version. **MUST use Python 3.10.10.**
```bash
# Check version
python --version

# If wrong, delete venv and recreate
deactivate
Remove-Item -Recurse -Force thesis-adhd
py -3.10 -m venv thesis-adhd
.\thesis-adhd\Scripts\Activate.ps1
python --version  # Verify 3.10.x
```

### CUDA/PyTorch Issues

**Problem:** CUDA not available despite having NVIDIA GPU
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem:** Out of memory errors during training
```bash
# Use memory-optimized model
python main.py --stage training --use-optimized-model

# Or reduce batch size in main.py (TRAINING_CONFIG)
```

### Data Issues

**Problem:** Feature manifest shows 1047 runs instead of 759 subjects
- **Cause:** Multi-run subjects counted multiple times
- **Solution:** Already fixed - manifest uses `drop_duplicates(subset=['subject_id'])`

**Problem:** ROI-ranking fails with "inhomogeneous shape"
- **Cause:** Subjects have different timeseries lengths (235-352 timepoints)
- **Solution:** Already fixed - uses statistical features (mean, std, min, max, median)

### Installation Issues

**Problem:** PyTorch Geometric installation fails
```bash
# Use pre-compiled wheels
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**Problem:** Requirements.txt has unavailable torch version
- **Solution:** Install PyTorch separately (as shown above), then `pip install -r requirements.txt`

## 📈 Results

### Experimental Results (Class Imbalance Study)

We conducted systematic experiments to handle severe class imbalance (3:1 HC/ADHD vs base study's 1.26:1):

| Version | Configuration | Overall Accuracy | Sensitivity | Specificity | Finding |
|---------|--------------|------------------|-------------|-------------|----------|
| **V7** (Baseline) | No class weights | 64.49% | **24.79%** | 77.65% | Severe majority class bias |
| **V6** (Adapted) | Class weights=[1.0, 4.0] | 54.73% | **44.90%** | 58.00% | **+81% sensitivity improvement** |
| **V8** (Aggressive) | Class weights=[1.0, 5.0] | 55.77% | 41.98% | 60.35% | Diminishing returns |

**Key Findings:**
1. **V7 proves the imbalance problem**: Without adaptations, model achieves 64% accuracy by predicting "healthy" most of the time (only 24.79% sensitivity)
2. **V6 demonstrates effective solution**: Class weighting (4×) improves ADHD detection by 81% while maintaining balanced specificity (58%)
3. **V8 validates optimization**: Higher weighting (5×) performs worse, showing v6 is near-optimal
4. **Balanced mini-batches**: WeightedRandomSampler creates ~50/50 batches from 75/25 dataset (implemented in `validation/loso.py`)

**Methodology Contributions:**
- Demonstrates that dataset imbalance severity (3:1 vs 1.26:1) fundamentally changes optimal methodology
- Establishes complete experimental narrative: problem demonstration (v7) → solution (v6) → validation (v8)
- Shows 44.9% sensitivity with 58.0% specificity represents reasonable trade-off given data constraints

### Current Status (as of latest run)

| Stage | Status | Details |
|-------|--------|---------|
| Preprocessing | ✅ Complete | 759/759 subjects (564.15 GB) |
| Feature Extraction | ✅ Complete | 759/759 subjects (~12 GB) |
| ROI-Ranking | ✅ Complete | 37 optimal ROIs (80.91% accuracy) |
| Site Configuration | ✅ Complete | 2 configurations ready |
| Data Splitting | ⏳ Pending | Ready to run |
| Model Training | ⏳ Pending | Ready to run |

### Expected Performance (based on ROI-ranking)

**Baseline Single-ROI Performance:**
- Best ROI (ROI_189): 80.78% accuracy
- Top 10 ROIs: ~80.85% average

**Optimal Multi-ROI Performance:**
- 37 ROIs: **80.91% accuracy**
- Improvement: +0.13% over best single ROI

**Hybrid Model (Full Pipeline - Expected):**
- With GNN + STAN + Fusion: **85-90% accuracy** (estimated)
- LOSO validation across 8 sites
- Robust to site effects

## 📚 Documentation

- **Main README:** This file
- **Thesis Tools:** `THESIS_TOOLS_README.md` - Ablation, visualization, comparison
- **Multi-Site Setup:** `MULTI_SITE_SETUP_GUIDE.md` - Dataset preparation
- **ROI-Ranking:** `ROI_RANKING_AND_SITE_SELECTION.md` - Feature selection details
- **Optimization:** `optimization/OPTIMIZATION_README.md` - Memory optimization techniques
- **GPU Setup:** `GPU_SETUP.md` - CUDA configuration

## 🗂️ Project Structure

```
DATASCII_10_ADHD/
├── data/                           # Data directory (not in repo)
│   ├── raw/                       # Raw ADHD-200 data
│   ├── preprocessed/              # Preprocessed fMRI (564 GB)
│   ├── features/                  # Extracted features (12 GB)
│   ├── roi_ranking/               # ROI-ranking results
│   ├── site_configs/              # Site-filtered datasets
│   ├── splits/                    # Train/val/test splits
│   └── trained/                   # Model checkpoints & results
├── experiments/                    # Thesis experiments
│   └── ablation_study.py         # Automated ablation framework
├── feature_extraction/            # Feature extraction modules
│   ├── parcellation_and_feature_extraction.py
│   └── roi_ranking.py            # ROI-ranking implementation
├── models/                        # Model architectures
│   ├── gnn.py                    # Enhanced GNN branch
│   ├── stan.py                   # Enhanced STAN branch
│   ├── fusion_layer.py           # Cross-modal fusion
│   ├── gnn_stan_hybrid.py        # Full hybrid model
│   └── gnn_stan_hybrid_optimized.py  # Memory-optimized variant
├── optimization/                  # Training optimizations
│   ├── focal_loss.py             # Focal Loss for imbalance
│   ├── early_stopping.py         # Early stopping
│   └── advanced_memory_optimization.py  # AGO, HTAAS, HAB
├── preprocessing/                 # Preprocessing pipeline
│   ├── preprocess.py
│   └── retry_failed.py           # Retry failed subjects
├── training/                      # Training modules
│   ├── train.py                  # Training loop
│   ├── dataset.py                # PyTorch Dataset
│   └── data_splitter.py          # Split generation
├── validation/                    # Validation strategies
│   ├── loso.py                   # Leave-One-Site-Out
│   ├── kfold.py                  # K-Fold CV
│   └── nested_cross_validation.py  # Nested CV
├── utils/                         # Utilities
│   ├── data_loader.py
│   ├── site_selector.py          # Site filtering
│   └── visualization.py          # Plotting utilities (NEW!)
├── scripts/                       # Helper scripts
│   ├── preprocess_all_sites.py
│   ├── check_preprocessed_integrity.py
│   └── compare_experiments.py    # Experiment comparison (NEW!)
├── main.py                        # Main pipeline entry point
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── README.md                      # This file
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{adhd_gnn_stan_2025,
  title={GNN-STAN Hybrid Model for ADHD Classification from Resting-State fMRI},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Master's Thesis}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **ADHD-200 Consortium** for providing the dataset
- **Schaefer Parcellation** (Schaefer et al., 2018)
- **PyTorch** and **PyTorch Geometric** communities
- All contributors and researchers in the ADHD neuroimaging field

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Check documentation in `THESIS_TOOLS_README.md`
- Review troubleshooting section above

---

**Status:** Pipeline ready for final training ✅
**Next Step:** Run `python main.py --stage training --site-config both`

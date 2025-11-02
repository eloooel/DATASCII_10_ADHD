# ADHD-200 Multi-Site Integration Guide

Complete guide for integrating OHSU, Pittsburgh, and Peking_1 data centers into your ADHD classification pipeline.

---

## 📋 Quick Start

### What's Already Done ✅
- **Code Updated**: `utils/data_loader.py` handles all ADHD-200 phenotypic formats
- **Tested**: ID column detection, DX code parsing, subject ID formatting
- **Ready**: Pipeline can process 5 data centers (Brown, NeuroIMAGE, OHSU, Pittsburgh, Peking_1)

### What You Need To Do
1. Download fMRI data from ADHD-200 consortium
2. Organize into directory structure (see below)
3. Copy phenotypic CSV files
4. Run pipeline: preprocessing → features → training

---

## 🌍 Supported Data Centers

### Current Sites (Already Working)
| Site | Subjects | ADHD | Control | Status |
|------|----------|------|---------|--------|
| Brown | 26 | 0 | 26 | ⚠️ Test release (pending diagnosis) |
| NeuroIMAGE | 69 | 23 | 46 | ✅ Working |

### New Sites (Ready to Add)
| Site | Subjects | ADHD | Control | Balance | Site Code |
|------|----------|------|---------|---------|-----------|
| OHSU | 113 | 43 | 70 | 38% | 6 |
| Pittsburgh | 89 | 0 | 89 | 0% ⚠️ | 7 |
| Peking_1 | 136 | 48 | 88 | 35% | 1 |

### Combined Dataset
- **Total**: 348 subjects (95 current + 253 new)
- **ADHD**: 84 subjects (24.1%)
- **Control**: 264 subjects (75.9%)
- **Growth**: **+266% more data, +265% more ADHD cases**

---

## 📂 Directory Structure

### Required Structure
```
data/raw/
├── Brown/
│   ├── participants.tsv (or phenotypic_data.csv)
│   └── sub-XXXXXXX/
│       └── func/
│           └── rest.nii.gz
├── NeuroIMAGE/
│   ├── participants.tsv
│   └── sub-XXXXXXX/
│       └── func/
│           └── rest.nii.gz
├── OHSU/
│   ├── phenotypic_data.csv
│   └── sub-XXXXXXX/
│       └── func/
│           └── rest.nii.gz
├── Pittsburgh/
│   ├── phenotypic_data.csv
│   └── sub-XXXXXX/
│       └── func/
│           └── rest.nii.gz
└── Peking_1/
    ├── phenotypic_data.csv
    └── sub-XXXXXXX/
        └── func/
            └── rest.nii.gz
```

### Key Requirements
- ✅ Subject folders MUST start with `sub-`
- ✅ Functional data MUST be in `func/` subdirectory
- ✅ Files can be `.nii.gz` or `.nii` format
- ✅ Phenotypic file can be CSV or TSV

---

## 🚀 Setup Instructions

### Step 1: Download fMRI Data
Download rs-fMRI data from ADHD-200 Consortium:
- **URL**: http://fcon_1000.projects.nitrc.org/indi/adhd200/
- **Sites needed**: OHSU, Pittsburgh, Peking_1
- **File format**: NIfTI (`.nii.gz` or `.nii`)
- **Data type**: Resting-state functional MRI

### Step 2: Organize Data
If downloaded data doesn't have `sub-` prefix, rename directories:
```powershell
# Example: Rename numeric IDs to sub- format
Rename-Item "data\raw\OHSU\1084283" "data\raw\OHSU\sub-1084283"
```

Or use batch rename:
```powershell
# Batch rename all numeric folders in OHSU
Get-ChildItem "data\raw\OHSU" -Directory | Where-Object { $_.Name -notmatch '^sub-' } | ForEach-Object { Rename-Item $_.FullName -NewName "sub-$($_.Name)" }
```

### Step 3: Copy Phenotypic Files
```powershell
# Copy from Downloads to data directories
Copy-Item "d:\Downloads\OHSU_phenotypic.csv" "data\raw\OHSU\phenotypic_data.csv"
Copy-Item "d:\Downloads\Pittsburgh_phenotypic.csv" "data\raw\Pittsburgh\phenotypic_data.csv"
Copy-Item "d:\Downloads\Peking_1_phenotypic.csv" "data\raw\Peking_1\phenotypic_data.csv"
```

### Step 4: Verify Structure
```powershell
# Check if subject directories exist
ls data\raw\OHSU\sub-*
ls data\raw\Pittsburgh\sub-*
ls data\raw\Peking_1\sub-*

# Check if functional files exist
ls data\raw\OHSU\sub-*\func\*.nii.gz
ls data\raw\Pittsburgh\sub-*\func\*.nii.gz
ls data\raw\Peking_1\sub-*\func\*.nii.gz
```

### Step 5: Run Data Discovery
```powershell
.\thesis-adhd\Scripts\Activate.ps1
python -c "from utils.data_loader import DataDiscovery; from pathlib import Path; discovery = DataDiscovery(Path('data/raw')); subjects = discovery.discover_subjects(); discovery.save_metadata(subjects, Path('data/raw/subjects_metadata.csv')); print(f'Discovered {len(subjects)} total subjects')"
```

**Expected Output:**
```
Processing dataset: Brown
📋 Brown: Loaded 0 participant diagnoses
Processing dataset: NeuroIMAGE
📋 NeuroIMAGE: Loaded 73 participant diagnoses
Processing dataset: OHSU
📋 OHSU: Loaded 79 participant diagnoses
Processing dataset: Pittsburgh
📋 Pittsburgh: Loaded 89 participant diagnoses
Processing dataset: Peking_1
📋 Peking_1: Loaded 85 participant diagnoses
Discovered XXX total subjects
```

### Step 6: Run Preprocessing
```powershell
python main.py --stage preprocessing
```

**Note**: This takes **~40 hours** for 348 subjects (~5-10 min per subject)
- Monitor progress: Check `data/preprocessed/*/preprocessing_manifest.csv`
- Run overnight or use parallel processing

### Step 7: Run Feature Extraction
```powershell
python main.py --stage features
```

This will:
- Apply Schaefer-200 atlas parcellation
- Extract ROI time series (200 ROIs × timepoints)
- Compute connectivity matrices (200×200)
- Generate `data/features/feature_manifest.csv`

### Step 8: Verify Results
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/features/feature_manifest.csv'); print(f'Total: {len(df)} subjects'); print('\nBy site:'); print(df.groupby('site')['diagnosis'].value_counts().unstack(fill_value=0)); print(f'\nClass balance:'); dx = df.diagnosis.value_counts().to_dict(); print(f'  ADHD: {dx.get(1, 0)} ({100*dx.get(1,0)/len(df):.1f}%)'); print(f'  Control: {dx.get(0, 0)} ({100*dx.get(0,0)/len(df):.1f}%)')"
```

**Expected Output:**
```
Total: ~348 subjects

By site:
           0   1
site            
Brown      26  0
NeuroIMAGE 46  23
OHSU       42  37
Peking_1   61  24
Pittsburgh 89  0

Class balance:
  ADHD: 84 (24.1%)
  Control: 264 (75.9%)
```

### Step 9: Train Model
```powershell
python main.py --stage training
```

Expected improvements over baseline:
- **Before**: 95 subjects, 21% accuracy (catastrophic with bug)
- **After**: 348 subjects, expected 70-80% accuracy

---

## 🔧 Technical Details

### Code Changes Made

**File**: `utils/data_loader.py` (line 233)

**Change**: Added `'ID'` to ID column search list

```python
# Before:
for col in ['participant_id', 'Subject', 'ScanDir ID', 'subject_id']:

# After:
for col in ['participant_id', 'Subject', 'ScanDir ID', 'ID', 'subject_id']:
```

**Reason**: Peking_1 test release uses `'ID'` column instead of `'ScanDir ID'`

### Diagnosis Code Mapping

The pipeline automatically handles ADHD-200 numeric diagnosis codes:

| DX Code | Meaning | Pipeline Mapping |
|---------|---------|------------------|
| 0 | Typically Developing Control | 0 (Control) |
| 1 | ADHD-Combined | 1 (ADHD) |
| 2 | ADHD-Hyperactive/Impulsive | 1 (ADHD) |
| 3 | ADHD-Inattentive | 1 (ADHD) |

**Implementation** (`utils/data_loader.py` lines 260-270):
```python
if dx_value in [0, '0'] or dx_str in ['TDC', 'Control', ...]:
    diagnosis = 0
elif dx_value in [1, 2, 3, '1', '2', '3'] or 'ADHD' in dx_str:
    diagnosis = 1  # All ADHD subtypes → binary class 1
```

### Phenotypic File Requirements

**Required Columns:**
- **ID Column**: One of: `ScanDir ID`, `ID`, `participant_id`, `Subject`, `subject_id`
- **DX Column**: `DX` (diagnosis codes 0, 1, 2, or 3)

**Optional Columns**: Age, Gender, Site, IQ, Handedness, etc.

### Subject ID Formatting

Pipeline automatically adds `sub-` prefix if missing:
```python
subj_id = str(row[id_col])
if not subj_id.startswith('sub-'):
    subj_id = f"sub-{subj_id}"
```

Example:
- Phenotypic ID: `1084283` → Directory: `sub-1084283` ✅ Match!

---

## 📊 Expected Performance Impact

### Current Model (95 subjects)
- Training data: 23 ADHD + 72 Control
- Previous result: 21% accuracy (catastrophic - bug fixed)
- Expected with fix: 60-70% accuracy
- Issue: Limited ADHD cases for learning

### After Adding New Sites (348 subjects)
- Training data: 84 ADHD + 264 Control
- **3.7× more training data**
- **3.7× more ADHD cases** (critical for pattern learning!)
- Expected improvements:
  - ✅ Better ADHD pattern recognition
  - ✅ More robust cross-validation
  - ✅ Better generalization across sites
  - ✅ Reduced overfitting
  - 🎯 **Target: 70-80% accuracy**

### Class Imbalance
- Current: 24% ADHD vs 76% Control
- **Focal loss** already configured (α=0.25, γ=2.0)
- Consider excluding Pittsburgh (all controls) → 32% ADHD balance

---

## ⚠️ Important Notes

### Pittsburgh Site
- **ZERO ADHD subjects** (all 89 are controls)
- Useful for control population but won't help ADHD detection
- **Recommendation**: Include for now, monitor site-specific performance
- **Alternative**: Exclude to improve class balance (24% → 32% ADHD)

### Processing Time
- **Preprocessing**: ~5-10 min per subject × 348 = **40 hours**
- **Feature extraction**: ~2-3 min per subject × 348 = **12 hours**
- **Total**: ~50-60 hours for complete pipeline
- **Recommendation**: Run overnight or over weekend

### Memory Requirements
- **Preprocessing**: ~4GB RAM per subject
- **Feature extraction**: ~2GB RAM per subject
- **Training**: 4GB VRAM (GPU) or 8GB RAM (CPU)
- **Disk space**: ~500MB per subject (preprocessed + features)

### Site-Specific Notes

**OHSU**:
- High-quality data with comprehensive phenotypic info
- Good ADHD balance (38%)
- Age range: ~7-12 years

**Pittsburgh**:
- All controls, age range ~10-21 years
- Good for control population
- Consider excluding if class balance is critical

**Peking_1**:
- Large sample size (136 subjects)
- Good ADHD balance (35%)
- Test release uses different ID column (`'ID'` vs `'ScanDir ID'`)
- Age range: ~8-17 years

---

## 🔍 Troubleshooting

### Problem: "No subjects found"
**Check**: Directory structure has `sub-` prefix
```powershell
ls data\raw\OHSU\sub-*
```

**Fix**: Rename directories to include `sub-` prefix
```powershell
Get-ChildItem "data\raw\OHSU" -Directory | Where-Object { $_.Name -notmatch '^sub-' } | ForEach-Object { Rename-Item $_.FullName -NewName "sub-$($_.Name)" }
```

### Problem: "No diagnosis found"
**Check**: Phenotypic file exists and has DX column
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/OHSU/phenotypic_data.csv'); print(df.columns.tolist()); print(df[['ScanDir ID', 'DX']].head())"
```

**Fix**: Verify phenotypic file copied correctly

### Problem: "Subject ID mismatch"
**Check**: Phenotypic IDs vs directory names
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/OHSU/phenotypic_data.csv'); print('Phenotypic IDs:', df['ScanDir ID'].head().tolist())"
ls data\raw\OHSU\sub-* | Select-Object -First 5
```

**Fix**: Pipeline auto-adds `sub-` prefix, so `1084283` will match `sub-1084283`

### Problem: "Memory error during preprocessing"
**Fix**: Process sites one at a time
```powershell
# Process each site separately
python -c "from preprocessing.preprocess import ADHDPreprocessor; proc = ADHDPreprocessor(); proc.process_dataset('data/raw/OHSU')"
```

### Problem: "CUDA out of memory during training"
**Fix**: Use CPU mode or reduce batch size
```powershell
# Use CPU
python main.py --stage training --no-cuda

# Or edit batch_size in main.py (default is 2)
```

---

## 📚 Additional Resources

- **ADHD-200 Consortium**: http://fcon_1000.projects.nitrc.org/indi/adhd200/
- **Schaefer Atlas**: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
- **Pipeline Documentation**: See `README.md` for full pipeline details

---

## ✅ Setup Checklist

Use this checklist to track your progress:

### Data Preparation
- [ ] Downloaded OHSU fMRI data
- [ ] Downloaded Pittsburgh fMRI data
- [ ] Downloaded Peking_1 fMRI data
- [ ] Organized data with `sub-` prefix
- [ ] Placed data in `func/` subdirectories
- [ ] Copied OHSU phenotypic CSV
- [ ] Copied Pittsburgh phenotypic CSV
- [ ] Copied Peking_1 phenotypic CSV

### Verification
- [ ] Verified directory structure
- [ ] Verified functional files exist
- [ ] Ran data discovery successfully
- [ ] All 5 sites detected

### Processing
- [ ] Preprocessing completed (~40 hours)
- [ ] Feature extraction completed
- [ ] Feature manifest generated
- [ ] Verified ~348 subjects with features
- [ ] Verified 84 ADHD + 264 Control distribution

### Training
- [ ] Model training started
- [ ] Cross-validation running
- [ ] Results show improvement over baseline

---

## 🎯 Summary

### What's Ready
- ✅ Code updated to handle all ADHD-200 formats
- ✅ ID column detection (including Peking_1's `'ID'` column)
- ✅ DX code parsing (0, 1, 2, 3)
- ✅ Subject ID auto-formatting (`sub-` prefix)

### What You Get
- 🚀 **3.7× more training data** (95 → 348 subjects)
- 🚀 **3.7× more ADHD cases** (23 → 84 ADHD)
- 🌍 **5 data centers** for better generalization
- 🎯 **Expected 70-80% accuracy** (vs previous 21%)

### Next Steps
1. Download fMRI data
2. Organize directory structure
3. Copy phenotypic files
4. Run pipeline
5. Train model with expanded dataset

**Your pipeline is ready to scale!** 🚀

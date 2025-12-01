# Complete ROI Ranking and Feature Extraction Documentation

## Table of Contents
1. [Overview](#overview)
2. [Feature Extraction Pipeline](#feature-extraction-pipeline)
3. [Schaefer-200 Parcellation](#schaefer-200-parcellation)
4. [ROI Timeseries Extraction](#roi-timeseries-extraction)
5. [Functional Connectivity Computation](#functional-connectivity-computation)
6. [ROI Ranking Phase 1: Individual Evaluation](#roi-ranking-phase-1-individual-evaluation)
7. [ROI Ranking Phase 2: Incremental Combination](#roi-ranking-phase-2-incremental-combination)
8. [Statistical Features](#statistical-features)
9. [Validation Strategies](#validation-strategies)
10. [File Format Specifications](#file-format-specifications)
11. [Parallel Processing Architecture](#parallel-processing-architecture)

---

## Overview

The feature extraction and ROI ranking pipeline transforms preprocessed fMRI data into analysis-ready features optimized for ADHD classification. This involves three major stages:

**Stage 1: Feature Extraction**
- Load preprocessed 4D fMRI data
- Apply Schaefer-200 parcellation
- Extract ROI timeseries (200 ROIs)
- Compute functional connectivity matrices (200×200)

**Stage 2: ROI Ranking Phase 1**
- Evaluate each of 200 ROIs independently
- Train 200 separate classifiers (one per ROI)
- Use Leave-One-Site-Out (LOSO) cross-validation
- Rank ROIs by individual classification accuracy

**Stage 3: ROI Ranking Phase 2**
- Test incremental combinations (top 1, top 2, ..., top N)
- Find optimal ROI subset
- Create filtered feature manifest for final training

**Pipeline Flow:**
```
Preprocessed fMRI → Schaefer-200 Parcellation → ROI Timeseries → 
Functional Connectivity → Individual ROI Ranking → Incremental Combination → 
Optimal ROI Selection → Training Features
```

**Key Outputs:**
- ROI timeseries CSV files: `(n_timepoints, 200)` per subject
- Functional connectivity matrices: `(200, 200)` per subject
- ROI rankings: Sorted by classification performance
- Optimal ROI subset: Best performing combination

---

## Feature Extraction Pipeline

### Input Requirements

**Preprocessed Data Structure:**
```
data/preprocessed/
├── site_1/
│   ├── subject_001/
│   │   ├── func_preproc.nii.gz    # 4D fMRI (x, y, z, timepoints)
│   │   └── mask.nii.gz            # 3D brain mask (x, y, z)
│   ├── subject_002/
│   │   ├── func_preproc.nii.gz
│   │   └── mask.nii.gz
├── site_2/
│   ├── subject_003/
...
```

**Metadata CSV:**
```csv
subject_id,site,diagnosis,input_path,out_dir
subject_001,site_1,0,/path/to/func.nii,/output/
subject_002,site_1,1,/path/to/func.nii,/output/
```

**Variables:**
- `subject_id`: Unique identifier (e.g., `sub-1019436`)
- `site`: Acquisition site (e.g., `KKI`, `OHSU`, `Peking_1`)
- `diagnosis`: 0 = Control, 1 = ADHD
- `input_path`: Path to preprocessed functional file
- `out_dir`: Output directory for features

### Parcellation Atlas

**Schaefer-200 7-Network Parcellation:**
- **File:** `Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii`
- **Space:** MNI152 2mm isotropic
- **Shape:** `(91, 109, 91)` voxels
- **ROIs:** 200 cortical parcels
- **Networks:** 7 intrinsic connectivity networks
  - Visual (Vis)
  - Somatomotor (SomMot)
  - Dorsal Attention (DorsAttn)
  - Ventral Attention (SalVentAttn)
  - Limbic
  - Frontoparietal Control (Cont)
  - Default Mode Network (Default)

**ROI Labeling:**
```python
ROI_1, ROI_2, ..., ROI_200
```

**Atlas Values:**
- 0: Background (non-brain)
- 1-200: ROI indices

### Worker Function Architecture

```python
def extract_features_worker(row, preproc_dir, feature_out_dir, atlas_labels, parcellation_path):
    """
    Process single subject: functional data → ROI timeseries → connectivity
    
    Args:
        row: Subject metadata (dict)
        preproc_dir: Path to preprocessed data
        feature_out_dir: Output directory
        atlas_labels: List of 200 ROI labels
        parcellation_path: Path to Schaefer atlas
    
    Returns:
        result: Dictionary with status and outputs
    """
```

---

## Schaefer-200 Parcellation

### Loading Parcellation

#### Class Initialization
```python
class SchaeferParcellation:
    def __init__(self, parcellation_path: Path = None):
        self.parcellation_path = parcellation_path
        self.atlas_data = None          # 3D array (91, 109, 91)
        self.roi_labels = None          # List of 200 strings
        self.n_rois = 200
```

#### Load Atlas
```python
def load_parcellation(self) -> bool:
    if self.parcellation_path and self.parcellation_path.exists():
        atlas_img = nib.load(str(self.parcellation_path))
        self.atlas_data = atlas_img.get_fdata().astype(int)
        self.roi_labels = self._generate_roi_labels()
        return True
    else:
        return False
```

**Atlas Data:**
- **Type:** `numpy.ndarray`, dtype `int`
- **Shape:** `(91, 109, 91)` for MNI152 2mm
- **Values:** 0 (background), 1-200 (ROI indices)

**Example Atlas Slice (z=45):**
```
[[ 0  0  0  0  0  0  0  0  0 ...]
 [ 0  0  0 78 78 78 79 79  0 ...]
 [ 0  0 78 78 78 79 79 79  0 ...]
 [ 0 82 82 83 83 84 84 85  0 ...]
 ...]
```

#### ROI Label Generation
```python
def _generate_roi_labels(self):
    return [f"ROI_{i+1}" for i in range(200)]
```

**Labels List:**
```python
['ROI_1', 'ROI_2', 'ROI_3', ..., 'ROI_200']
```

### Spatial Resampling

If functional data has different resolution than atlas:

```python
if self.atlas_data.shape != fmri_data.shape[:3]:
    zoom_factors = [
        fmri_data.shape[i] / self.atlas_data.shape[i] 
        for i in range(3)
    ]
    self.atlas_data = zoom(
        self.atlas_data.astype(float), 
        zoom_factors, 
        order=0  # Nearest-neighbor interpolation
    ).astype(int)
```

**Zoom Factors Calculation:**
```
Example:
fMRI shape: (64, 64, 33)
Atlas shape: (91, 109, 91)

zoom_factors = [
    64/91 = 0.703,  # x-axis
    64/109 = 0.587, # y-axis
    33/91 = 0.363   # z-axis
]
```

**Nearest-Neighbor Interpolation (order=0):**
- Preserves integer ROI labels
- No averaging (critical for discrete labels)
- Each output voxel takes value of nearest input voxel

**After Resampling:**
- Atlas shape: `(64, 64, 33)` (matches fMRI)
- Values: Still 0-200 (labels preserved)

---

## ROI Timeseries Extraction

### Purpose
Extract average BOLD signal from each ROI across all timepoints.

### Input
- **Functional Data:** 4D array `(x, y, z, n_timepoints)`
  - Example: `(91, 109, 91, 176)`
- **Brain Mask:** 3D array `(x, y, z)`
  - Binary: 0 (background), 1 (brain)
- **Atlas Data:** 3D array `(x, y, z)`
  - Values: 0-200 (ROI labels)

### Process

#### Initialize Output Array
```python
n_timepoints = fmri_data.shape[-1]  # 176
roi_timeseries = np.zeros((n_timepoints, self.n_rois))  # (176, 200)
```

#### Create Combined Mask
```python
mask_3d = brain_mask > 0 if brain_mask is not None else self.atlas_data > 0
```

**Mask Purpose:**
- Exclude background voxels (intensity = 0)
- Only average signal from brain tissue

#### Extract ROI Timeseries
```python
for roi_id in range(1, self.n_rois + 1):  # 1 to 200
    # Create binary mask for this ROI
    roi_mask = (self.atlas_data == roi_id) & mask_3d
    
    # Count voxels in this ROI
    n_voxels = np.sum(roi_mask)
    
    if n_voxels > 0:
        # Extract all voxel timeseries in this ROI
        roi_voxel_timeseries = fmri_data[roi_mask]  # Shape: (n_voxels, n_timepoints)
        
        # Average across voxels
        roi_timeseries[:, roi_id-1] = np.mean(roi_voxel_timeseries, axis=0)
    else:
        # Empty ROI (rare, but possible)
        roi_timeseries[:, roi_id-1] = np.zeros(n_timepoints)
```

**ROI Mask Example (ROI 84):**
```
atlas_data == 84:
[[False False False  True  True ...]
 [False False  True  True  True ...]
 ...]

mask_3d:
[[True True True True True ...]
 [True True True True True ...]
 ...]

roi_mask = (atlas_data == 84) & mask_3d:
[[False False False  True  True ...]
 [False False  True  True  True ...]
 ...]
```

**Extraction Details:**
- `fmri_data[roi_mask]`: Boolean indexing
  - Selects all voxels where `roi_mask == True`
  - Result shape: `(n_voxels_in_roi, n_timepoints)`
- `np.mean(..., axis=0)`: Average across voxels
  - Result shape: `(n_timepoints,)` - one value per TR

**Example for ROI 84:**
```
n_voxels_in_roi = 156 voxels
fmri_data[roi_mask].shape = (156, 176)

For each timepoint t:
roi_timeseries[t, 83] = mean([
    voxel_1[t],
    voxel_2[t],
    ...,
    voxel_156[t]
])
```

### Output

**ROI Timeseries Array:**
- **Shape:** `(n_timepoints, n_rois)` = `(176, 200)`
- **Type:** `numpy.ndarray`, dtype `float64`
- **Values:** BOLD signal intensity (arbitrary units)

**Typical Value Ranges:**
- Mean: 100-500 (depends on scanner, preprocessing)
- Std: 10-50
- After standardization: mean ≈ 0, std ≈ 1

**CSV Format:**
```csv
ROI_1,ROI_2,ROI_3,...,ROI_200
234.56,198.23,312.45,...,267.89
235.12,199.01,311.78,...,268.34
234.89,197.56,313.12,...,267.45
...
```

### Quality Checks

#### Check for NaN Values
```python
if np.any(np.isnan(roi_timeseries)):
    print(f"Warning: NaN values in ROI timeseries")
```

**Causes of NaN:**
- Division by zero (empty ROI)
- Invalid voxel intensities
- Preprocessing artifacts

#### Check for Infinite Values
```python
if np.any(np.isinf(roi_timeseries)):
    print(f"Warning: Infinite values in ROI timeseries")
```

**Causes of Inf:**
- Extreme outlier voxels
- Normalization errors
- Data corruption

---

## Functional Connectivity Computation

### Purpose
Compute pairwise Pearson correlation between all ROI timeseries to create functional connectivity matrix.

### Input
- **ROI Timeseries:** `(n_timepoints, n_rois)` = `(176, 200)`

### Process

#### Class Initialization
```python
class FunctionalConnectivityExtractor:
    def __init__(self, method='pearson', standardize=True):
        self.method = method          # 'pearson' correlation
        self.standardize = standardize  # Z-score normalization
```

#### Standardization (Z-Score)
```python
if self.standardize:
    scaler = StandardScaler()
    timeseries = scaler.fit_transform(timeseries)
```

**StandardScaler Formula:**
```
X_scaled = (X - mean(X)) / std(X)
```

**Per-ROI Standardization:**
```python
For each ROI column j:
    mean_j = np.mean(timeseries[:, j])
    std_j = np.std(timeseries[:, j])
    
    For each timepoint i:
        timeseries[i, j] = (timeseries[i, j] - mean_j) / std_j
```

**After Standardization:**
- Each ROI timeseries: mean = 0, std = 1
- Removes scale differences between ROIs
- Makes correlation interpretation consistent

#### Pearson Correlation
```python
conn = np.corrcoef(timeseries.T)  # Transpose to correlate ROIs
```

**Pearson Correlation Formula:**
```
r_xy = Σ((x_i - x̄)(y_i - ȳ)) / (n * σ_x * σ_y)

Where:
- x, y: Two ROI timeseries
- x̄, ȳ: Means
- σ_x, σ_y: Standard deviations
- n: Number of timepoints
```

**np.corrcoef Behavior:**
- Input: `(n_timepoints, n_rois)` after transpose
- Output: `(n_rois, n_rois)` = `(200, 200)`
- Computes all pairwise correlations

**Correlation Matrix Structure:**
```
       ROI_1  ROI_2  ROI_3  ...  ROI_200
ROI_1   1.00   0.45   0.23  ...   0.12
ROI_2   0.45   1.00   0.67  ...   0.34
ROI_3   0.23   0.67   1.00  ...   0.56
...
ROI_200 0.12   0.34   0.56  ...   1.00
```

**Properties:**
- **Symmetric:** `conn[i, j] == conn[j, i]`
- **Diagonal = 1:** Self-correlation
- **Range:** -1 to +1
  - +1: Perfect positive correlation
  - 0: No correlation
  - -1: Perfect negative correlation

#### Remove Diagonal
```python
np.fill_diagonal(conn, 0.0)
```

**Reason:**
- Self-correlation (diagonal) is always 1.0
- Not informative for connectivity
- Set to 0 to exclude from analysis

#### Handle Invalid Values
```python
conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)
```

**Replace:**
- `NaN` → 0.0 (undefined correlations)
- `+Inf` → 0.0 (invalid computations)
- `-Inf` → 0.0 (invalid computations)

### Output

**Connectivity Matrix:**
- **Shape:** `(200, 200)` - symmetric
- **Type:** `numpy.ndarray`, dtype `float64`
- **Values:** Pearson correlation coefficients
- **Diagonal:** 0.0 (self-connections removed)
- **Range:** -1 to +1 (mostly 0.0 to 0.8)

**File Formats:**

**NumPy Binary (.npy):**
```python
np.save('subject_001_connectivity_matrix.npy', conn)
```
- Fast loading/saving
- Preserves exact precision
- Used for model training

**CSV Format (.csv):**
```csv
,ROI_1,ROI_2,ROI_3,...,ROI_200
ROI_1,0.00,0.45,0.23,...,0.12
ROI_2,0.45,0.00,0.67,...,0.34
ROI_3,0.23,0.67,0.00,...,0.56
...
ROI_200,0.12,0.34,0.56,...,0.00
```
- Human-readable
- Easy inspection
- Used for visualization

**Typical Statistics:**
```
Mean correlation: 0.15 ± 0.08
Min: -0.45
Max: 0.89
Non-zero connections: 19,900 (200×200 - 200 diagonal)
Strong connections (|r| > 0.5): ~2,000 (10%)
```

---

## ROI Ranking Phase 1: Individual Evaluation

### Purpose
Evaluate each of 200 ROIs independently to identify which brain regions are most discriminative for ADHD classification.

### Methodology

Based on original SCCNN-RNN baseline approach:
1. Extract features from single ROI
2. Train classifier on single-ROI features
3. Evaluate with Leave-One-Site-Out (LOSO) cross-validation
4. Repeat for all 200 ROIs
5. Rank ROIs by classification accuracy

### Data Loading

#### Load All Subject Data
```python
roi_data_dict = {}  # subject_id -> timeseries array
labels = []         # diagnosis (0=Control, 1=ADHD)
sites = []          # acquisition site
subject_ids = []    # subject identifiers

for idx, row in feature_manifest.iterrows():
    # Load ROI timeseries CSV
    ts_df = pd.read_csv(row['ts_path'])
    roi_data_dict[row['subject_id']] = ts_df.values  # (n_timepoints, 200)
    
    labels.append(row['diagnosis'])
    sites.append(row['site'])
    subject_ids.append(row['subject_id'])

labels = np.array(labels)  # Shape: (n_subjects,)
sites = np.array(sites)    # Shape: (n_subjects,)
```

**Variables:**
- `roi_data_dict`: Dictionary mapping `subject_id` → timeseries array
- `labels`: Array of diagnoses `[0, 1, 0, 1, ...]`
- `sites`: Array of site names `['KKI', 'OHSU', 'Peking_1', ...]`
- `subject_ids`: List of subject IDs

**Data Sizes:**
```
Total subjects: 759
Total ROIs: 200
Timeseries length: Variable (140-220 timepoints)
Memory per subject: ~1.4 MB (176 TRs × 200 ROIs × 8 bytes)
Total memory: ~1.1 GB
```

### ROI-by-ROI Evaluation Loop

```python
roi_results = []

for roi_idx in range(self.n_rois):  # 0 to 199
    # Extract this ROI's timeseries for all subjects
    roi_timeseries_list = [
        roi_data_dict[subj][:, roi_idx] for subj in subject_ids
    ]
    
    # Compute statistical features
    roi_features = self._compute_roi_features_from_list(roi_timeseries_list)
    
    # Evaluate with LOSO cross-validation
    accuracy = self._evaluate_with_loso(roi_features, labels, sites)
    
    # Store result
    roi_results.append({
        'roi_id': roi_idx + 1,        # 1-based indexing
        'roi_name': f'ROI_{roi_idx + 1}',
        'accuracy': accuracy
    })
```

**Loop Execution:**
- **Iterations:** 200 (one per ROI)
- **Time per iteration:** 5-15 seconds
- **Total time:** ~20-50 minutes

### Statistical Feature Extraction

#### Single-ROI Features
```python
def _compute_roi_features_from_list(self, timeseries_list: list) -> np.ndarray:
    """
    Compute 7 statistical features per subject
    
    Args:
        timeseries_list: List of arrays, each (n_timepoints,)
    
    Returns:
        features: Array (n_subjects, 7)
    """
    features = []
    
    for ts in timeseries_list:  # One timeseries per subject
        feat = [
            np.mean(ts),           # Mean activation
            np.std(ts),            # Standard deviation
            np.min(ts),            # Minimum value
            np.max(ts),            # Maximum value
            np.median(ts),         # Median value
            np.percentile(ts, 25), # 25th percentile (Q1)
            np.percentile(ts, 75)  # 75th percentile (Q3)
        ]
        features.append(feat)
    
    return np.array(features)  # Shape: (n_subjects, 7)
```

**Feature Interpretation:**

| Feature | Formula | Description |
|---------|---------|-------------|
| Mean | `μ = Σx_i / n` | Average BOLD signal |
| Std Dev | `σ = sqrt(Σ(x_i - μ)² / n)` | Signal variability |
| Min | `min(x)` | Lowest activation |
| Max | `max(x)` | Highest activation |
| Median | `x[n/2]` | Middle value (robust to outliers) |
| Q1 | `x[n/4]` | 25th percentile |
| Q3 | `x[3n/4]` | 75th percentile |

**Example Features (ROI 84, one subject):**
```python
Timeseries: [234.5, 235.1, 234.8, 236.2, ..., 235.7]  # 176 values

Features:
[235.3,    # Mean
 1.8,      # Std
 231.2,    # Min
 239.5,    # Max
 235.2,    # Median
 234.1,    # Q1
 236.5]    # Q3
```

**Feature Matrix:**
```
Subject      Mean   Std   Min    Max    Median  Q1     Q3
sub-001      235.3  1.8   231.2  239.5  235.2   234.1  236.5
sub-002      198.7  2.1   194.3  203.1  198.5   197.2  200.1
...
sub-759      267.4  1.5   264.8  271.2  267.3   266.2  268.6

Shape: (759 subjects, 7 features)
```

### Leave-One-Site-Out (LOSO) Cross-Validation

#### Purpose
Evaluate generalization across different acquisition sites to ensure model robustness to site effects.

#### Implementation
```python
def _evaluate_with_loso(self, features, labels, sites):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    logo = LeaveOneGroupOut()
    accuracies = []
    
    # Split by site
    for train_idx, test_idx in logo.split(features, labels, sites):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        clf.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    return np.mean(accuracies)
```

**LOSO Cross-Validation Example:**

Assume 4 sites: KKI, OHSU, Peking_1, Peking_2

**Fold 1:**
- **Train:** OHSU, Peking_1, Peking_2 (600 subjects)
- **Test:** KKI (159 subjects)
- **Accuracy:** 68.5%

**Fold 2:**
- **Train:** KKI, Peking_1, Peking_2 (650 subjects)
- **Test:** OHSU (109 subjects)
- **Accuracy:** 71.2%

**Fold 3:**
- **Train:** KKI, OHSU, Peking_2 (568 subjects)
- **Test:** Peking_1 (191 subjects)
- **Accuracy:** 69.8%

**Fold 4:**
- **Train:** KKI, OHSU, Peking_1 (659 subjects)
- **Test:** Peking_2 (100 subjects)
- **Accuracy:** 72.1%

**Average Accuracy:** `(68.5 + 71.2 + 69.8 + 72.1) / 4 = 70.4%`

**Random Forest Parameters:**
- `n_estimators=100`: 100 decision trees
- `max_depth=10`: Maximum tree depth (prevents overfitting)
- `random_state=42`: Reproducible results

### Ranking Results

#### Create Rankings DataFrame
```python
rankings_df = pd.DataFrame(roi_results)
rankings_df = rankings_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
rankings_df['rank'] = range(1, len(rankings_df) + 1)
```

**Rankings Table:**
```csv
rank,roi_id,roi_name,accuracy
1,84,ROI_84,0.7029
2,123,ROI_123,0.6847
3,56,ROI_56,0.6792
4,145,ROI_145,0.6738
5,91,ROI_91,0.6701
...
196,34,ROI_34,0.5123
197,178,ROI_178,0.5098
198,12,ROI_12,0.5076
199,189,ROI_189,0.5042
200,67,ROI_67,0.5011
```

**Interpretation:**
- **Rank 1 (ROI_84):** 70.29% accuracy - most discriminative
- **Rank 2 (ROI_123):** 68.47% accuracy
- **Rank 200 (ROI_67):** 50.11% accuracy - least discriminative (≈ chance)

**Top 15 ROIs Visualization:**
```
Rank  ROI Name  Accuracy
----  --------  --------
1     ROI_84    70.29%   ████████████████████████████
2     ROI_123   68.47%   ██████████████████████████
3     ROI_56    67.92%   █████████████████████████
4     ROI_145   67.38%   ████████████████████████
5     ROI_91    67.01%   ████████████████████████
6     ROI_178   66.54%   ███████████████████████
7     ROI_102   66.12%   ███████████████████████
8     ROI_34    65.89%   ██████████████████████
9     ROI_156   65.43%   ██████████████████████
10    ROI_201   65.01%   ██████████████████████
11    ROI_78    64.67%   █████████████████████
12    ROI_143   64.23%   █████████████████████
13    ROI_89    63.98%   █████████████████████
14    ROI_167   63.54%   █████████████████████
15    ROI_45    63.12%   ████████████████████
```

### Output Files

**1. Complete Rankings:**
```
Path: data/roi_ranking/roi_rankings.csv
Rows: 200 ROIs
Columns: rank, roi_id, roi_name, accuracy
```

**2. Top 50 ROIs:**
```
Path: data/roi_ranking/top_50_rois.csv
Rows: 50 ROIs
Columns: rank, roi_id, roi_name, accuracy
```

---

## ROI Ranking Phase 2: Incremental Combination

### Purpose
Determine optimal number of ROIs by incrementally combining top-ranked ROIs and evaluating performance.

### Methodology

Test combinations:
- Top 1 ROI only
- Top 2 ROIs (rank 1 + rank 2)
- Top 3 ROIs (rank 1 + rank 2 + rank 3)
- ...
- Top N ROIs

Find combination with highest accuracy.

### Data Preparation

```python
# Get ordered ROI indices by rank
ranked_roi_indices = (roi_rankings.sort_values('rank')['roi_id'].values - 1).astype(int)

# Example:
# ranked_roi_indices = [83, 122, 55, 144, 90, ...]  # 0-based
#                       ROI_84, ROI_123, ROI_56, ROI_145, ROI_91
```

### Incremental Testing Loop

```python
roi_counts = list(range(1, max_rois + 1, step_size))  # [1, 2, 3, ..., 50]
incremental_results = []

for n_rois in roi_counts:
    # Select top N ROIs
    selected_roi_indices = ranked_roi_indices[:n_rois]
    
    # Extract multi-ROI timeseries for all subjects
    combined_timeseries_list = [
        roi_data_dict[subj][:, selected_roi_indices] 
        for subj in subject_ids
    ]
    
    # Compute features from multiple ROIs
    combined_features = self._compute_multi_roi_features_from_list(
        combined_timeseries_list
    )
    
    # Evaluate with LOSO
    accuracy = self._evaluate_with_loso(combined_features, labels, sites)
    
    # Store result
    incremental_results.append({
        'n_rois': n_rois,
        'selected_rois': selected_roi_indices.tolist(),
        'accuracy': accuracy
    })
```

**Loop Execution:**
- **Test counts:** 1, 2, 3, ..., 50 ROIs (50 iterations)
- **Time per iteration:** 10-30 seconds
- **Total time:** ~10-25 minutes

### Multi-ROI Feature Extraction

```python
def _compute_multi_roi_features_from_list(self, timeseries_list):
    """
    Compute features from multiple ROI timeseries
    
    Args:
        timeseries_list: List of arrays (n_timepoints, n_rois)
    
    Returns:
        features: Array (n_subjects, n_features)
    """
    features = []
    
    for subj_ts in timeseries_list:
        # subj_ts shape: (n_timepoints, n_rois)
        n_timepoints, n_rois = subj_ts.shape
        subj_features = []
        
        # Statistical features per ROI
        for roi_idx in range(n_rois):
            roi_ts = subj_ts[:, roi_idx]
            subj_features.extend([
                np.mean(roi_ts),              # Mean
                np.std(roi_ts),               # Std
                np.max(roi_ts) - np.min(roi_ts)  # Range
            ])
        
        # Inter-ROI correlations
        if n_rois > 1:
            corr_matrix = np.corrcoef(subj_ts.T)  # (n_rois, n_rois)
            upper_tri = corr_matrix[np.triu_indices(n_rois, k=1)]
            subj_features.extend([
                np.mean(upper_tri),  # Mean correlation
                np.std(upper_tri)    # Std correlation
            ])
        
        features.append(subj_features)
    
    return np.array(features)
```

**Feature Count Calculation:**

For N ROIs:
- **Per-ROI features:** 3 features × N ROIs = 3N features
  - Mean, Std, Range for each ROI
- **Inter-ROI features:** 2 features (if N > 1)
  - Mean correlation, Std correlation
- **Total features:** 3N + 2

**Examples:**

**1 ROI:**
- Features: [mean, std, range]
- Total: 3 features

**2 ROIs:**
- Per-ROI: [mean_1, std_1, range_1, mean_2, std_2, range_2]
- Inter-ROI: [mean_corr, std_corr]
- Total: 3×2 + 2 = 8 features

**5 ROIs:**
- Per-ROI: 3 × 5 = 15 features
- Inter-ROI: 2 features
- Total: 17 features

**50 ROIs:**
- Per-ROI: 3 × 50 = 150 features
- Inter-ROI: 2 features
- Total: 152 features

**Inter-ROI Correlation Calculation:**

```python
corr_matrix = np.corrcoef(subj_ts.T)

# Example for 3 ROIs:
corr_matrix = [
    [1.00, 0.45, 0.23],
    [0.45, 1.00, 0.67],
    [0.23, 0.67, 1.00]
]

# Upper triangle (exclude diagonal)
upper_tri = [0.45, 0.23, 0.67]  # Correlations: (1,2), (1,3), (2,3)

mean_corr = np.mean(upper_tri) = 0.45
std_corr = np.std(upper_tri) = 0.19
```

**Upper Triangle Indices:**
```python
n_rois = 3
np.triu_indices(n_rois, k=1)  # k=1 excludes diagonal

Returns:
(array([0, 0, 1]), array([1, 2, 2]))

Indices: (0,1), (0,2), (1,2)
```

### Finding Optimal ROI Count

```python
results_df = pd.DataFrame(incremental_results)

# Find best performing combination
optimal_idx = results_df['accuracy'].idxmax()
optimal_n_rois = results_df.loc[optimal_idx, 'n_rois']
optimal_accuracy = results_df.loc[optimal_idx, 'accuracy']
```

**Example Results:**

```csv
n_rois,accuracy
1,0.7029
2,0.6923
3,0.6845
4,0.6798
5,0.6756
6,0.6723
7,0.6701
8,0.6689
9,0.6678
10,0.6654
...
25,0.6301
...
50,0.5987
```

**Performance Curve:**
```
Accuracy (%)
72 |  *
70 |     *
68 |        *  *
66 |              *  *  *
64 |                       *  *  *
62 |                                *  *  *
60 |                                         *  *  *
58 |________________________________________________
   1    5    10   15   20   25   30   35   40   45   50
                    Number of ROIs
```

**Interpretation:**
- **Peak at N=1:** Single best ROI (ROI_84) achieves 70.29%
- **Decreasing performance:** Adding more ROIs decreases accuracy
- **Optimal:** Use only top 1 ROI

**Why Fewer ROIs Can Be Better:**
1. **Overfitting:** More features with limited samples
2. **Noise accumulation:** Lower-ranked ROIs add noise
3. **Curse of dimensionality:** High-dimensional feature space
4. **Simplicity:** Single ROI is most interpretable

### Visualization

```python
plt.figure(figsize=(12, 6))

plt.plot(results_df['n_rois'], results_df['accuracy'] * 100, 
        marker='o', linewidth=2, markersize=4)
plt.axvline(optimal_n_rois, color='red', linestyle='--', linewidth=2,
           label=f'Optimal: {optimal_n_rois} ROIs')
plt.axhline(optimal_accuracy * 100, color='green', 
           linestyle=':', alpha=0.5)

plt.xlabel('Number of Top-Ranked ROIs', fontsize=12)
plt.ylabel('LOSO Accuracy (%)', fontsize=12)
plt.title('Incremental ROI Combination Performance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.savefig('incremental_roi_performance.png', dpi=300)
```

### Output Files

**1. Incremental Results:**
```
Path: data/roi_ranking/incremental_roi_results.csv
Columns: n_rois, selected_rois, accuracy
Rows: 50 (one per ROI count)
```

**2. Optimal ROI List:**
```
Path: data/roi_ranking/optimal_rois_top_1.csv
Columns: rank, roi_id, roi_name, accuracy
Rows: 1 (or N if optimal > 1)
```

**3. Performance Plot:**
```
Path: data/roi_ranking/incremental_roi_performance.png
Format: PNG, 300 DPI
Size: ~200 KB
```

---

## Statistical Features

### Single-ROI Features (7 features)

| Feature | Formula | Purpose | Typical Range |
|---------|---------|---------|---------------|
| Mean | `μ = Σx_i / n` | Average activation level | 100-500 |
| Std Dev | `σ = sqrt(Σ(x_i - μ)² / n)` | Signal variability | 10-50 |
| Min | `min(x)` | Lowest activation | 50-450 |
| Max | `max(x)` | Peak activation | 150-550 |
| Median | `x[n/2]` | Robust central tendency | 100-500 |
| Q1 (25%) | `x[n/4]` | Lower quartile | 90-490 |
| Q3 (75%) | `x[3n/4]` | Upper quartile | 110-510 |

### Multi-ROI Features (3N + 2 features)

**Per-ROI Features (3 × N):**
1. **Mean:** Average BOLD signal
2. **Std:** Signal variability
3. **Range:** Max - Min

**Inter-ROI Features (2):**
1. **Mean Correlation:** Average pairwise correlation
2. **Std Correlation:** Variability of correlations

### Feature Standardization

Applied before classification:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**StandardScaler Per-Feature:**
```
For each feature column j:
    mean_j = mean(X_train[:, j])
    std_j = std(X_train[:, j])
    
    X_train[:, j] = (X_train[:, j] - mean_j) / std_j
    X_test[:, j] = (X_test[:, j] - mean_j) / std_j  # Use training stats
```

**Result:**
- Each feature: mean = 0, std = 1 (on training set)
- All features on same scale
- Prevents feature dominance

---

## Validation Strategies

### Leave-One-Site-Out (LOSO) Cross-Validation

**Advantages:**
- Tests generalization across sites
- Robust to site effects (scanner, protocol differences)
- Realistic evaluation of multi-site deployment

**Disadvantages:**
- Fewer folds than K-fold
- Unbalanced fold sizes
- One poor site can skew results

**When to Use:**
- Multi-site datasets
- Evaluating site robustness
- Final model evaluation

### K-Fold Cross-Validation (Alternative)

```python
def _evaluate_with_cv(self, features, labels, n_folds=5):
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Standardize, train, evaluate
        ...
        
        accuracies.append(acc)
    
    return np.mean(accuracies)
```

**Advantages:**
- Balanced fold sizes
- More stable estimates
- Standard practice

**Disadvantages:**
- Doesn't account for site effects
- May overestimate performance
- Site confounding possible

**When to Use:**
- Single-site datasets
- Initial feature selection
- Exploratory analysis

---

## File Format Specifications

### ROI Timeseries CSV

**Filename:** `{subject_id}_roi_timeseries.csv`

**Format:**
```csv
ROI_1,ROI_2,ROI_3,...,ROI_200
234.56,198.23,312.45,...,267.89
235.12,199.01,311.78,...,268.34
...
```

**Properties:**
- **Shape:** (n_timepoints, 200)
- **Columns:** 200 ROIs
- **Rows:** Variable (140-220 timepoints)
- **Data Type:** float64
- **Size:** ~350 KB per subject

### Connectivity Matrix NumPy

**Filename:** `{subject_id}_connectivity_matrix.npy`

**Format:** Binary NumPy array

**Properties:**
- **Shape:** (200, 200)
- **Data Type:** float64
- **Symmetric:** `conn[i,j] == conn[j,i]`
- **Diagonal:** 0.0
- **Size:** ~320 KB per subject

**Loading:**
```python
conn = np.load('subject_001_connectivity_matrix.npy')
```

### Connectivity Matrix CSV

**Filename:** `{subject_id}_connectivity_matrix.csv`

**Format:**
```csv
,ROI_1,ROI_2,ROI_3,...,ROI_200
ROI_1,0.00,0.45,0.23,...,0.12
ROI_2,0.45,0.00,0.67,...,0.34
...
```

**Properties:**
- **Shape:** (200, 200)
- **Index:** ROI labels
- **Columns:** ROI labels
- **Size:** ~800 KB per subject

### Feature Manifest CSV

**Filename:** `feature_manifest.csv`

**Format:**
```csv
subject_id,site,fc_path,ts_path,diagnosis
sub-001,KKI,/path/to/sub-001_connectivity_matrix.npy,/path/to/sub-001_roi_timeseries.csv,0
sub-002,KKI,/path/to/sub-002_connectivity_matrix.npy,/path/to/sub-002_roi_timeseries.csv,1
...
```

**Columns:**
- `subject_id`: Unique identifier
- `site`: Acquisition site
- `fc_path`: Path to connectivity .npy file
- `ts_path`: Path to timeseries .csv file
- `diagnosis`: 0=Control, 1=ADHD

**Usage:** Input for model training

### ROI Rankings CSV

**Filename:** `roi_rankings.csv`

**Format:**
```csv
rank,roi_id,roi_name,accuracy
1,84,ROI_84,0.7029
2,123,ROI_123,0.6847
...
```

**Properties:**
- **Rows:** 200 (all ROIs)
- **Sorted by:** accuracy (descending)

### Incremental Results CSV

**Filename:** `incremental_roi_results.csv`

**Format:**
```csv
n_rois,selected_rois,accuracy
1,"[83]",0.7029
2,"[83, 122]",0.6923
3,"[83, 122, 55]",0.6845
...
```

**Properties:**
- `n_rois`: Number of ROIs in combination
- `selected_rois`: List of ROI indices (0-based)
- `accuracy`: LOSO accuracy

---

## Parallel Processing Architecture

### Feature Extraction Worker

**Execution Model:** Process pool parallelism

```python
def extract_features_worker(row, preproc_dir, feature_out_dir, atlas_labels, parcellation_path):
    # 1. Load preprocessed data
    # 2. Apply parcellation
    # 3. Extract ROI timeseries
    # 4. Compute connectivity
    # 5. Save outputs
    return result
```

**Process Isolation:**
- Each worker: Independent Python process
- Memory: Not shared between workers
- Error: One failure doesn't affect others

### Parallel Execution

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all subjects
    futures = {
        executor.submit(extract_features_worker, row, ...): row["subject_id"]
        for _, row in metadata.iterrows()
    }
    
    # Collect results as they complete
    results = []
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            subject_id = futures[future]
            results.append({
                "subject_id": subject_id,
                "status": "failed",
                "error": str(e)
            })
```

**Parameters:**
- `max_workers`: Number of parallel processes
  - Default: `os.cpu_count() - 1`
  - Typical: 4-8 workers
  - Max: CPU cores - 1

**Performance:**
- **Sequential:** ~2-3 minutes per subject × 759 subjects = ~38 hours
- **Parallel (4 workers):** ~38 hours / 4 = ~9.5 hours
- **Parallel (8 workers):** ~38 hours / 8 = ~4.75 hours

### Resource Requirements

**Per-Subject (Worker):**
- **CPU:** 1 core
- **Memory:** 2-4 GB RAM
- **Disk I/O:** 
  - Read: ~20 MB (preprocessed data)
  - Write: ~1.5 MB (features)
- **Time:** 2-3 minutes

**System-Wide (8 workers):**
- **CPU:** 8 cores active
- **Memory:** 16-32 GB RAM
- **Disk I/O:** ~160 MB read, ~12 MB write per batch

### Error Handling

**File Corruption Detection:**
```python
def validate_gzipped_nifti(file_path):
    # Test gzip integrity
    with gzip.open(file_path, 'rb') as gz_file:
        chunk = gz_file.read(1024 * 1024)
    
    # Test NIfTI loading
    img = nib.load(file_path)
    data = img.get_fdata()[:5, :5, :5, :1]
    
    return validation_result
```

**Error Types:**
- **gzip_block_corruption:** Invalid stored block lengths
- **truncated_file:** Incomplete download
- **nifti_format_error:** Header corruption
- **parcellation_unavailable:** Missing atlas file
- **roi_extraction_error:** Shape mismatch

**Recovery:**
- Log detailed error info
- Skip corrupted subject
- Continue processing remaining subjects
- Generate error report

---

## Summary Statistics

### Feature Extraction

**Typical Results (759 subjects):**
```
Successful: 743 (97.9%)
Failed: 16 (2.1%)

Failure breakdown:
  File corruption: 12 (75%)
  ROI extraction errors: 3 (19%)
  Disk space issues: 1 (6%)

Processing time:
  Mean: 2.3 ± 0.8 minutes per subject
  Total: ~4.5 hours (8 workers)

Output sizes:
  ROI timeseries: 350 KB per subject
  Connectivity matrix: 320 KB per subject
  Total: ~500 MB for all subjects
```

### ROI Ranking Phase 1

**Typical Results:**
```
Total ROIs evaluated: 200
Evaluation time: 35 minutes

Performance range:
  Best ROI (ROI_84): 70.29%
  Worst ROI (ROI_67): 50.11%
  Mean: 59.8% ± 5.2%

Top 10 ROIs: 66-70% accuracy
Bottom 10 ROIs: 50-52% accuracy (near chance)
```

### ROI Ranking Phase 2

**Typical Results:**
```
Combinations tested: 50 (1 to 50 ROIs)
Evaluation time: 18 minutes

Optimal configuration:
  Number of ROIs: 1
  ROI ID: ROI_84
  Accuracy: 70.29%

Performance trend:
  1 ROI: 70.29%
  5 ROIs: 67.56%
  10 ROIs: 66.54%
  25 ROIs: 63.01%
  50 ROIs: 59.87%
```

---

## Key Equations Reference

### Pearson Correlation
```
r_xy = Σ((x_i - x̄)(y_i - ȳ)) / (n * σ_x * σ_y)

Simplified after standardization:
r_xy = (1/n) * Σ(x_i * y_i)
```

### Z-Score Standardization
```
z = (x - μ) / σ

where:
μ = mean
σ = standard deviation
```

### Statistical Features
```
Mean: μ = Σx_i / n
Std: σ = sqrt(Σ(x_i - μ)² / n)
Range: R = max(x) - min(x)
Median: x[n/2]
Q1: x[n/4]
Q3: x[3n/4]
```

### Feature Counts
```
Single ROI: 7 features
N ROIs: 3N + 2 features
  - Per-ROI: 3 × N
  - Inter-ROI: 2
```

### Upper Triangle Indices
```
For n × n correlation matrix:
Number of correlations = n(n-1)/2

Example (n=5):
Correlations = 5×4/2 = 10
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Missing Parcellation File
**Symptom:** "Atlas data not loaded"
**Solution:** Verify Schaefer atlas path, re-download if needed

#### 2. Shape Mismatch
**Symptom:** "Atlas shape != fMRI shape"
**Solution:** Check preprocessing normalization, verify MNI152 space

#### 3. NaN in Timeseries
**Symptom:** "NaN values in ROI timeseries"
**Solution:** Check for empty ROIs, validate preprocessing

#### 4. File Corruption
**Symptom:** "Error -3 while decompressing"
**Solution:** Re-download preprocessed data, check disk integrity

#### 5. Memory Issues
**Symptom:** Process killed, memory > 90%
**Solution:** Reduce max_workers, process in batches

---

This documentation covers every computation, variable, and process in the ROI ranking and feature extraction pipeline from preprocessed fMRI to optimized training features.

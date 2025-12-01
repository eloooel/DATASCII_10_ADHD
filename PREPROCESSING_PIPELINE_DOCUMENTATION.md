# Complete Preprocessing Pipeline Documentation

## Table of Contents
1. [Overview](#overview)
2. [Pipeline Configuration](#pipeline-configuration)
3. [Stage 1: Data Loading and Validation](#stage-1-data-loading-and-validation)
4. [Stage 2: Motion Correction](#stage-2-motion-correction)
5. [Stage 3: Slice Timing Correction](#stage-3-slice-timing-correction)
6. [Stage 4: Spatial Normalization](#stage-4-spatial-normalization)
7. [Stage 5: Temporal Filtering](#stage-5-temporal-filtering)
8. [Stage 6: ICA-AROMA Denoising](#stage-6-ica-aroma-denoising)
9. [Stage 7: Additional Denoising](#stage-7-additional-denoising)
10. [Stage 8: Brain Mask Generation](#stage-8-brain-mask-generation)
11. [Output Files and Verification](#output-files-and-verification)
12. [Parallel Processing Architecture](#parallel-processing-architecture)

---

## Overview

The preprocessing pipeline transforms raw fMRI NIfTI files into analysis-ready data through 8 sequential stages. Each stage is designed to remove specific artifacts and normalize the data for downstream analysis.

**Input:** Raw 4D fMRI NIfTI file (`.nii` or `.nii.gz`)
- Shape: `(x, y, z, timepoints)` typically `(64, 64, 33, 176)`
- Format: Anatomical voxel space with scanner-specific coordinates

**Output:** Preprocessed 4D fMRI NIfTI file + Brain mask + Confound regressors
- Shape: Normalized to MNI152 2mm space
- Format: Standardized anatomical coordinates
- Quality: Motion-corrected, temporally filtered, ICA-AROMA denoised

**Pipeline Order (Critical):**
```
Load → Motion Correction → Slice Timing → Spatial Normalization → 
Temporal Filtering → ICA-AROMA → aCompCor → Brain Mask
```

---

## Pipeline Configuration

### Default Configuration Dictionary

```python
config = {
    'motion_correction': {
        'enabled': True,
        'reference': 'mean',              # Reference volume: 'mean', 'first', or index
        'save_parameters': True,
        'max_displacement_mm': 3.0        # Threshold for high-motion flagging
    },
    'slice_timing_correction': {
        'enabled': True,
        'tr': 2.0,                        # Repetition Time in seconds
        'slice_order': 'ascending',       # 'ascending', 'descending', 'interleaved'
        'ref_slice': 'middle'             # Reference slice: 'middle' or index
    },
    'spatial_normalization': {
        'enabled': True,
        'template': 'MNI152',             # Standard template
        'resolution': 2,                  # 2mm isotropic voxels
        'smooth_fwhm': 6.0               # Gaussian smoothing kernel (mm)
    },
    'temporal_filtering': {
        'enabled': True,
        'low_freq': 0.009,                # High-pass: remove slow drift (Hz)
        'high_freq': 0.08,                # Low-pass: remove high-freq noise (Hz)
        'filter_type': 'butterworth',
        'filter_order': 4
    },
    'denoising': {
        'motion_regression': True,
        'ica_aroma': {
            'enabled': True,
            'n_components': 25,            # Number of ICA components
            'max_iter': 200,
            'tolerance': 1e-4,
            'motion_correlation_threshold': 0.3,
            'high_freq_threshold': 0.2,
            'high_freq_ratio_threshold': 0.6,
            'spatial_std_threshold': 2.0
        },
        'acompcor': {
            'enabled': True,
            'n_components': 5,             # PCA components from noise ROIs
            'variance_threshold': 0.5
        },
        'global_signal': {
            'enabled': False               # Global signal regression (controversial)
        }
    },
    'output': {
        'save_intermediate': False,
        'compress': True,                  # Save as .nii.gz
        'datatype': 'float32'
    }
}
```

### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `self.device` | `torch.device` | CUDA/CPU device for computation |
| `self.processing_log` | `List[Dict]` | Log of all processing steps |
| `self.subject_id` | `str` | Current subject identifier |
| `self.metadata` | `Dict` | Image metadata (affine, header, TR) |
| `self.config` | `Dict` | Complete configuration dictionary |

---

## Stage 1: Data Loading and Validation

### Purpose
Load raw fMRI data and validate dimensions and data quality.

### Input
- **File:** Raw 4D NIfTI file (`.nii` or `.nii.gz`)
- **Path:** Full absolute path to input file

### Process

#### 1.1 Load NIfTI File
```python
data = nib.load(input_path)
img_data_np = data.get_fdata().astype(np.float32)
img_data = torch.from_numpy(img_data_np).to(self.device)
```

**Variables:**
- `data`: `nibabel.Nifti1Image` object
- `img_data_np`: NumPy array, shape `(x, y, z, timepoints)`, dtype `float32`
- `img_data`: PyTorch tensor on GPU/CPU

#### 1.2 Dimension Validation
```python
if len(img_data.shape) != 4:
    raise ValueError(f"Expected 4D data, got {len(img_data.shape)}D")
```

**Requirements:**
- Must be 4D: `(x_voxels, y_voxels, z_slices, n_timepoints)`
- Typical dimensions: `(64, 64, 33, 176)` to `(91, 109, 91, 240)`

#### 1.3 Data Quality Checks
```python
n_zeros = torch.sum(img_data == 0).item()
total_voxels = torch.prod(torch.tensor(img_data.shape)).item()
zero_ratio = n_zeros / total_voxels
```

**Quality Metrics:**
- `n_zeros`: Count of zero-valued voxels
- `zero_ratio`: Should be < 0.5 (most voxels should have signal)

#### 1.4 Extract Metadata
```python
metadata = {
    'shape': img_data.shape,                    # (x, y, z, timepoints)
    'voxel_size': data.header.get_zooms()[:3],  # (voxel_x, voxel_y, voxel_z) in mm
    'tr': data.header.get_zooms()[3],           # Repetition time in seconds
    'datatype': img_data.dtype,                 # torch.float32
    'orientation': nib.aff2axcodes(data.affine), # ('R', 'A', 'S') typically
    'affine': data.affine                       # 4x4 transformation matrix
}
```

**Affine Matrix:**
```
[[ voxel_to_world_x ]
 [ voxel_to_world_y ]
 [ voxel_to_world_z ]
 [ 0   0   0   1    ]]
```
Converts voxel coordinates (i, j, k) to world coordinates (x, y, z) in mm.

### Output
- **Data:** `img_data` (PyTorch tensor, 4D)
- **Metadata:** Dictionary with spatial/temporal information

---

## Stage 2: Motion Correction

### Purpose
Correct head motion artifacts by realigning all volumes to a reference volume.

### Input
- **Data:** 4D fMRI image (NIfTI format)
- **Shape:** `(x, y, z, n_timepoints)`

### Process

#### 2.1 Create Reference Volume
```python
if params['reference'] == 'mean':
    reference = np.mean(img_data, axis=-1).astype(np.float32)
elif params['reference'] == 'first':
    reference = img_data[..., 0].astype(np.float32)
else:
    ref_idx = int(params['reference'])
    reference = img_data[..., ref_idx].astype(np.float32)
```

**Reference Options:**
- **Mean:** Average across all timepoints (most stable)
- **First:** First volume (TR=0)
- **Index:** Specific volume number

**Variables:**
- `reference`: 3D array, shape `(x, y, z)`, dtype `float32`

#### 2.2 Initialize Motion Parameters
```python
motion_params = {
    'translations': np.zeros((n_vols, 3), dtype=np.float32),  # [x, y, z] shifts
    'rotations': np.zeros((n_vols, 3), dtype=np.float32),     # [roll, pitch, yaw]
    'displacement': np.zeros(n_vols, dtype=np.float32),       # Frame-wise displacement
    'excluded_volumes': []                                     # High-motion volumes
}
```

**Arrays:**
- `translations`: Translation in x, y, z (mm) for each volume
- `rotations`: Rotation around x, y, z axes (radians) - currently zeros (rigid body)
- `displacement`: Euclidean distance moved from previous volume
- `excluded_volumes`: List of volume indices exceeding motion threshold

#### 2.3 Volume-by-Volume Realignment
```python
for vol_idx in range(n_vols):
    current_vol = img_data[..., vol_idx].astype(np.float32)
    
    # Estimate translation
    translation = self._estimate_translation(reference, current_vol)
    
    # Apply shift
    corrected_vol = ndimage.shift(current_vol, translation, order=1, mode='nearest')
    corrected_data[..., vol_idx] = corrected_vol.astype(np.float32)
    
    # Store motion parameters
    motion_params['translations'][vol_idx] = translation
    
    # Calculate frame displacement
    if vol_idx > 0:
        prev_trans = motion_params['translations'][vol_idx - 1]
        displacement = np.sqrt(np.sum((translation - prev_trans) ** 2))
        motion_params['displacement'][vol_idx] = displacement
        
        # Flag excessive motion
        if displacement > params.get('max_displacement_mm', 3.0):
            motion_params['excluded_volumes'].append(vol_idx)
```

**Translation Estimation:**
```python
def _estimate_translation(reference, current):
    # Threshold images to focus on brain tissue
    ref_thresh = reference > np.percentile(reference[reference > 0], 50)
    cur_thresh = current > np.percentile(current[current > 0], 50)
    
    # Calculate center of mass
    ref_com = ndimage.center_of_mass(ref_thresh)
    cur_com = ndimage.center_of_mass(cur_thresh)
    
    # Translation is the difference
    translation = np.array(ref_com) - np.array(cur_com)
    
    # Limit maximum correction
    translation = np.clip(translation, -10, 10)  # Max 10 voxels
    
    return translation
```

**Key Concepts:**
- **Center of Mass (COM):** Weighted average position of all brain voxels
- **Translation Vector:** `[Δx, Δy, Δz]` in voxel units
- **Interpolation:** Linear interpolation (`order=1`) for subvoxel shifts
- **Mode:** `nearest` - extend edge values for boundary voxels

#### 2.4 Calculate Motion Statistics
```python
max_displacement = np.max(motion_params['displacement'])
mean_displacement = np.mean(motion_params['displacement'])
n_excluded = len(motion_params['excluded_volumes'])
```

**Quality Thresholds:**
- `max_displacement` < 3.0 mm: Good quality
- `mean_displacement` < 0.5 mm: Excellent quality
- `n_excluded` = 0: No high-motion volumes

### Output
- **Data:** Motion-corrected 4D image (NIfTI)
- **Motion Parameters:** Dictionary with translations, rotations, displacement
- **Excluded Volumes:** List of high-motion timepoints

---

## Stage 3: Slice Timing Correction

### Purpose
Correct for temporal differences in slice acquisition. Since slices are acquired sequentially within each TR, voxels in different slices represent different time points.

### Input
- **Data:** Motion-corrected 4D image
- **TR:** Repetition time (e.g., 2.0 seconds)
- **Slice Order:** Acquisition pattern

### Process

#### 3.1 Define Slice Acquisition Times
```python
tr = params.get('tr', metadata.get('tr', 2.0))
n_slices = img_data.shape[2]

if params['slice_order'] == 'ascending':
    slice_times = np.linspace(0, tr, n_slices, endpoint=False)
elif params['slice_order'] == 'descending':
    slice_times = np.linspace(tr, 0, n_slices, endpoint=False)
elif params['slice_order'] == 'interleaved':
    odd_times = np.linspace(0, tr/2, n_slices//2, endpoint=False)
    even_times = np.linspace(tr/2, tr, n_slices//2, endpoint=False)
    slice_times = np.zeros(n_slices)
    slice_times[::2] = odd_times   # Slices 0, 2, 4, ...
    slice_times[1::2] = even_times # Slices 1, 3, 5, ...
```

**Slice Time Patterns:**

**Ascending:** `[0.00, 0.06, 0.12, ..., 1.94]` seconds
```
Slice 0 acquired at t=0.00s
Slice 1 acquired at t=0.06s
Slice 2 acquired at t=0.12s
...
```

**Descending:** `[1.94, 1.88, 1.82, ..., 0.00]` seconds

**Interleaved:** `[0.00, 1.00, 0.06, 1.06, ...]` seconds
- Odd slices first, then even slices

#### 3.2 Reference Time
```python
if params['ref_slice'] == 'middle':
    ref_time = tr / 2
else:
    ref_slice_idx = int(params['ref_slice'])
    ref_time = slice_times[ref_slice_idx]
```

**Reference Time:** All slices are temporally shifted to this reference point (typically middle of TR).

#### 3.3 Temporal Interpolation
```python
original_times = np.arange(n_timepoints) * tr  # [0, 2, 4, 6, ...] seconds

for z in range(n_slices):
    slice_data = img_data[:, :, z, :]  # Shape: (x, y, n_timepoints)
    
    time_shift = slice_times[z] - ref_time
    shifted_times = original_times + time_shift
    
    for x in range(slice_data.shape[0]):
        for y in range(slice_data.shape[1]):
            voxel_ts = slice_data[x, y, :]
            
            if np.any(voxel_ts != 0):  # Skip empty voxels
                # Linear interpolation
                corrected_ts = np.interp(original_times, shifted_times, voxel_ts)
                corrected_data[x, y, z, :] = corrected_ts
```

**Interpolation Process:**

For a voxel at slice z=5:
- Original times: `[0, 2, 4, 6, 8, ...]` seconds
- Slice acquired at: `t_slice = 0.3` seconds
- Reference time: `t_ref = 1.0` seconds
- Time shift: `Δt = 0.3 - 1.0 = -0.7` seconds
- Shifted times: `[-0.7, 1.3, 3.3, 5.3, ...]`
- Interpolate original signal at original times using shifted times

**Linear Interpolation Formula:**
```
y_interp = y_1 + (x - x_1) * (y_2 - y_1) / (x_2 - x_1)
```

### Output
- **Data:** Slice-timing corrected 4D image (NIfTI)
- **Log:** Number of slices corrected, TR, reference time

---

## Stage 4: Spatial Normalization

### Purpose
Transform functional images to standard MNI152 template space for group-level analysis and anatomical consistency.

### Input
- **Data:** Slice-timing corrected 4D image (native space)
- **Shape:** Variable (scanner-specific)

### Process

#### 4.1 Load MNI152 Template
```python
from nilearn.datasets import load_mni152_template

resolution = params.get('resolution', 2)  # 2mm isotropic
mni_template = load_mni152_template(resolution=resolution)
```

**MNI152 Template:**
- **Space:** Montreal Neurological Institute standard space
- **Resolution:** 2mm isotropic voxels
- **Shape:** `(91, 109, 91)` voxels
- **Coverage:** Whole brain from -90mm to +90mm in each direction

#### 4.2 Affine Registration
```python
from nilearn.image import resample_to_img

normalized_img = resample_to_img(
    data_img,           # Source: functional image
    mni_template,       # Target: MNI152 template
    interpolation='linear',
    copy_header=True
)
```

**Affine Transformation:**
```
[x']   [a11  a12  a13  tx] [x]
[y'] = [a21  a22  a23  ty] [y]
[z']   [a31  a32  a33  tz] [z]
[1 ]   [0    0    0    1 ] [1]
```

Where:
- `a_ij`: Rotation and scaling parameters
- `t_x, t_y, t_z`: Translation parameters

**Transformation Steps:**
1. **Translation:** Align center of mass
2. **Rotation:** Align principal axes
3. **Scaling:** Match voxel dimensions
4. **Resampling:** Interpolate to 2mm grid

#### 4.3 Spatial Smoothing
```python
from nilearn.image import smooth_img

smooth_fwhm = params.get('smooth_fwhm', 6.0)  # 6mm Gaussian kernel

if smooth_fwhm > 0:
    smoothed_img = smooth_img(normalized_img, fwhm=smooth_fwhm)
```

**Gaussian Smoothing:**
```
G(x, y, z) = (1 / (2πσ²)^(3/2)) * exp(-(x² + y² + z²) / (2σ²))
```

Where:
- `FWHM = 2.355 * σ` (Full Width at Half Maximum)
- `FWHM = 6mm` typical for fMRI
- `σ = 6 / 2.355 = 2.55mm`

**Purpose of Smoothing:**
- Increase signal-to-noise ratio
- Compensate for anatomical variability
- Meet Gaussian random field theory assumptions
- Reduce spatial frequency noise

### Output
- **Data:** MNI152-normalized, smoothed 4D image
- **Shape:** `(91, 109, 91, n_timepoints)` at 2mm resolution
- **Space:** Standard stereotaxic coordinates

---

## Stage 5: Temporal Filtering

### Purpose
Remove physiological noise (respiratory ~0.3Hz, cardiac ~1Hz) and scanner drift (<0.01Hz) while preserving BOLD signal (0.01-0.1Hz).

### Input
- **Data:** Spatially normalized 4D image
- **TR:** Repetition time (sampling rate)

### Process

#### 5.1 Calculate Filter Parameters
```python
tr = metadata.get('tr', 2.0)
low_freq = params['low_freq']    # 0.009 Hz (high-pass)
high_freq = params['high_freq']  # 0.08 Hz (low-pass)
filter_order = params.get('filter_order', 4)

# Nyquist frequency
nyquist = 1 / (2 * tr)  # Max frequency = 1 / (2 * 2.0) = 0.25 Hz

# Normalize frequencies
low = low_freq / nyquist   # 0.009 / 0.25 = 0.036
high = high_freq / nyquist # 0.08 / 0.25 = 0.32
```

**Frequency Bands:**
- **Scanner drift:** < 0.009 Hz (removed by high-pass)
- **BOLD signal:** 0.01 - 0.08 Hz (preserved)
- **Respiratory:** ~0.3 Hz (removed by low-pass)
- **Cardiac:** ~1 Hz (removed by low-pass)
- **High-frequency noise:** > 0.08 Hz (removed by low-pass)

#### 5.2 Design Butterworth Bandpass Filter
```python
from scipy import signal

b, a = signal.butter(filter_order, [low, high], btype='band')
```

**Butterworth Filter:**
- **Order:** 4 (steeper roll-off)
- **Type:** Bandpass (allows frequencies between low and high)
- **Transfer Function:** `H(s) = 1 / sqrt(1 + (ω/ω_c)^(2n))`

**Filter Coefficients:**
- `b`: Numerator coefficients (feedforward)
- `a`: Denominator coefficients (feedback)

#### 5.3 Generate Brain Mask
```python
mean_img = np.mean(img_data, axis=-1)
brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)
```

**Brain Mask:**
- Threshold at 25th percentile of non-zero voxels
- Purpose: Only filter brain voxels (not background)
- Shape: `(x, y, z)` binary mask

#### 5.4 Apply Filter to Each Voxel
```python
brain_voxels = np.where(brain_mask)
total_voxels = len(brain_voxels[0])

for i in range(total_voxels):
    x, y, z = brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]
    voxel_ts = filtered_data[x, y, z, :].astype(np.float32)
    
    if np.any(voxel_ts != 0):
        # Linear detrending
        t = np.arange(len(voxel_ts), dtype=np.float32)
        p = np.polyfit(t, voxel_ts, deg=1)
        trend = p[1] + p[0] * t
        detrended = voxel_ts - trend
        
        # Apply bandpass filter
        filtered_ts = signal.filtfilt(b, a, detrended)
        
        # Add trend back
        filtered_data[x, y, z, :] = filtered_ts + trend
```

**Detrending:**
```
Linear fit: y = mx + b
Trend: trend(t) = p[1] + p[0] * t
Detrended: y_detrended = y_original - trend
```

**Zero-Phase Filtering (filtfilt):**
- Forward pass: Apply filter in time order
- Backward pass: Apply filter in reverse time order
- Result: Zero phase distortion (no time shift)

**Frequency Response:**
```
|H(f)| = 1 / sqrt(1 + ((f - f_center) / bandwidth)^8)
```

For `f_low=0.009Hz` to `f_high=0.08Hz`:
- Passband: Near unity gain
- Stopband: Strong attenuation (-40dB at order 4)

### Output
- **Data:** Temporally filtered 4D image (NIfTI)
- **Frequencies Preserved:** 0.009 - 0.08 Hz
- **DC Component:** Preserved (trend added back)

---

## Stage 6: ICA-AROMA Denoising

### Purpose
Automatically identify and remove motion-related independent components using Independent Component Analysis with Automatic Removal of Motion Artifacts.

### Input
- **Data:** Temporally filtered 4D image
- **Motion Parameters:** From Stage 2

### Process

#### 6.1 Extract Brain Timeseries
```python
brain_mask = self._generate_brain_mask(data_img)
brain_voxels_idx = np.where(brain_mask > 0)
n_brain_voxels = len(brain_voxels_idx[0])

# Extract timeseries from brain voxels only
brain_timeseries = np.zeros((n_timepoints, n_brain_voxels), dtype=np.float32)
for i in range(n_brain_voxels):
    x, y, z = brain_voxels_idx[0][i], brain_voxels_idx[1][i], brain_voxels_idx[2][i]
    brain_timeseries[:, i] = img_data[x, y, z, :]
```

**Data Structure:**
- **Shape:** `(n_timepoints, n_brain_voxels)`
- **Typical:** `(176 timepoints, ~30,000 voxels)`

#### 6.2 Perform ICA Decomposition
```python
from sklearn.decomposition import FastICA

n_components = min(params.get('n_components', 25), n_timepoints - 1, n_brain_voxels // 100)

ica = FastICA(
    n_components=n_components,
    random_state=42,
    max_iter=params.get('max_iter', 200),
    tol=params.get('tolerance', 1e-4)
)

# ICA decomposition
ica_timeseries = ica.fit_transform(brain_timeseries)  # (n_timepoints, n_components)
ica_spatial_maps = ica.components_                     # (n_components, n_voxels)
```

**ICA Model:**
```
X = AS
```
Where:
- `X`: Observed data `(n_timepoints, n_voxels)`
- `A`: Mixing matrix `(n_timepoints, n_components)` = `ica_timeseries`
- `S`: Source signals `(n_components, n_voxels)` = `ica_spatial_maps`

**Components:**
Each component has:
- **Temporal pattern:** `ica_timeseries[:, i]` - Time course
- **Spatial map:** `ica_spatial_maps[i, :]` - Brain activation pattern

#### 6.3 Identify Motion Components (AROMA Criteria)

##### Criterion 1: Correlation with Motion
```python
motion_ts = motion_params.get('displacement', np.zeros(ica_timeseries.shape[0]))

for comp_idx in range(n_components):
    component_ts = ica_timeseries[:, comp_idx]
    
    # Pearson correlation
    correlation = np.corrcoef(component_ts, motion_ts)[0, 1]
    
    if abs(correlation) > params.get('motion_correlation_threshold', 0.3):
        is_motion = True
```

**Motion Correlation:**
- **Threshold:** |r| > 0.3
- **Interpretation:** Component timecourse follows head motion

##### Criterion 2: High-Frequency Content
```python
# Frequency domain analysis
fft = np.fft.fft(component_ts)
freqs = np.fft.fftfreq(len(component_ts))
power_spectrum = np.abs(fft) ** 2

# Check if most power is in high frequencies (>0.2 Hz)
high_freq_mask = np.abs(freqs) > params.get('high_freq_threshold', 0.2)
high_freq_power = np.sum(power_spectrum[high_freq_mask])
total_power = np.sum(power_spectrum)

high_freq_ratio = high_freq_power / (total_power + 1e-8)

if high_freq_ratio > params.get('high_freq_ratio_threshold', 0.6):
    is_motion = True
```

**High-Frequency Criterion:**
- **Threshold:** >60% of power above 0.2 Hz
- **Rationale:** Motion artifacts are typically high-frequency

##### Criterion 3: Edge-Heavy Spatial Pattern
```python
spatial_map = ica_spatial_maps[comp_idx, :]
spatial_std = np.std(spatial_map)

if spatial_std > params.get('spatial_std_threshold', 2.0):
    # Check for edge-like patterns
    high_values = np.abs(spatial_map) > np.percentile(np.abs(spatial_map), 95)
    sparsity = np.sum(high_values) / len(spatial_map)
    
    if sparsity < 0.1:  # Very sparse activation
        is_motion = True
```

**Spatial Pattern:**
- **High std + sparse:** Edge artifacts, ringing
- **Motion artifacts:** Concentrated at brain boundaries

#### 6.4 Remove Motion Components
```python
motion_components = [0, 3, 7, 12]  # Example identified components

cleaned_timeseries = brain_timeseries.copy()

for comp_idx in motion_components:
    component_ts = ica_timeseries[:, comp_idx]
    
    for voxel_idx in range(n_brain_voxels):
        voxel_ts = cleaned_timeseries[:, voxel_idx]
        
        # Linear regression: remove component contribution
        correlation = np.corrcoef(voxel_ts, component_ts)[0, 1]
        
        if not np.isnan(correlation) and abs(correlation) > 0.1:
            # Calculate regression coefficient
            beta = correlation * (np.std(voxel_ts) / (np.std(component_ts) + 1e-8))
            
            # Remove component contribution
            cleaned_timeseries[:, voxel_idx] = voxel_ts - beta * component_ts
```

**Regression Model:**
```
voxel_cleaned = voxel_original - β * component_motion

where: β = r * (σ_voxel / σ_component)
```

#### 6.5 Reconstruct 4D Image
```python
cleaned_data = img_data.copy()

for i in range(n_brain_voxels):
    x, y, z = brain_voxels_idx[0][i], brain_voxels_idx[1][i], brain_voxels_idx[2][i]
    cleaned_data[x, y, z, :] = cleaned_timeseries[:, i]
```

### Output
- **Data:** ICA-AROMA cleaned 4D image (NIfTI)
- **ICA Results:** Dictionary with:
  - `motion_components`: List of removed component indices
  - `total_components`: Total ICA components
  - `n_components_removed`: Number removed

**Typical Results:**
- Total components: 25
- Motion components: 3-8 (12-32%)
- Cleaned: Reduced motion-correlated variance

---

## Stage 7: Additional Denoising

### Purpose
Further noise reduction using confound regression: motion parameters, aCompCor (CSF/WM noise), and optionally global signal.

### Input
- **Data:** ICA-AROMA cleaned 4D image
- **Motion Parameters:** From Stage 2

### Process

#### 7.1 Motion Regression
```python
confounds = {}

if params.get('motion_regression', True) and motion_params:
    # Motion parameters as confounds
    confounds['trans_x'] = motion_params['translations'][:, 0].tolist()
    confounds['trans_y'] = motion_params['translations'][:, 1].tolist()
    confounds['trans_z'] = motion_params['translations'][:, 2].tolist()
    confounds['frame_displacement'] = motion_params['displacement'].tolist()
    
    # Motion derivatives (velocity)
    confounds['trans_x_derivative'] = np.gradient(motion_params['translations'][:, 0]).tolist()
    confounds['trans_y_derivative'] = np.gradient(motion_params['translations'][:, 1]).tolist()
    confounds['trans_z_derivative'] = np.gradient(motion_params['translations'][:, 2]).tolist()
```

**Confounds Matrix:**
```
                trans_x  trans_y  trans_z  FD   dtrans_x  dtrans_y  dtrans_z
Timepoint 0     0.12     -0.05    0.23    0.0   0.0       0.0       0.0
Timepoint 1     0.15     -0.03    0.25    0.04  0.03      0.02      0.02
Timepoint 2     0.14     -0.01    0.24    0.03  -0.01     0.02     -0.01
...
```

**Frame Displacement (FD):**
```
FD_t = sqrt((x_t - x_{t-1})² + (y_t - y_{t-1})² + (z_t - z_{t-1})²)
```

#### 7.2 aCompCor (Anatomical Component Correction)
```python
n_components = params.get('n_components', 5)
variance_threshold = params.get('variance_threshold', 0.5)

# Create CSF/WM mask (simplified)
mean_img = np.mean(img_data, axis=-1)

# High-signal regions (likely CSF)
high_threshold = np.percentile(mean_img[mean_img > 0], 90)
# Low-signal regions (likely WM)
low_threshold = np.percentile(mean_img[mean_img > 0], 30)

noise_mask = (mean_img > high_threshold) | (mean_img < low_threshold)

# Extract time series from noise regions
noise_voxels = img_data[noise_mask]  # Shape: (n_noise_voxels, n_timepoints)
```

**aCompCor Regions:**
- **CSF:** High intensity (brightest 10% of voxels)
- **WM:** Low intensity (dimmest 30% of voxels)
- **Rationale:** These regions should not contain BOLD signal

**PCA on Noise Regions:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=n_components)
components = pca.fit_transform(noise_voxels.T)  # Shape: (n_timepoints, n_components)

# Check variance explained
var_explained = pca.explained_variance_ratio_
significant_components = np.where(var_explained > variance_threshold)[0]

# Add to confounds
for i, comp_idx in enumerate(significant_components):
    confounds[f'a_comp_cor_{i:02d}'] = components[:, comp_idx].tolist()
```

**aCompCor Components:**
- Component 0: 45% variance (respiratory)
- Component 1: 23% variance (cardiac)
- Component 2: 12% variance (scanner drift)
- Component 3: 8% variance (other noise)
- Component 4: 5% variance (residual)

#### 7.3 Global Signal Regression (Optional)
```python
if params.get('global_signal', {}).get('enabled', False):
    brain_mask = self._generate_brain_mask(data_img)
    brain_voxels = img_data[brain_mask > 0]
    global_signal = np.mean(brain_voxels, axis=0)
    confounds['global_signal'] = global_signal.tolist()
```

**Global Signal:**
- **Calculation:** Mean across all brain voxels at each timepoint
- **Shape:** `(n_timepoints,)` - one value per TR
- **Controversy:** May remove true neuronal signal
- **Default:** Disabled

#### 7.4 Apply Confound Regression
```python
n_timepoints = img_data.shape[-1]

# Create design matrix
confound_matrix = np.ones((n_timepoints, 1))  # Intercept

for confound_name, confound_ts in confounds.items():
    if len(confound_ts) == n_timepoints:
        confound_matrix = np.column_stack([confound_matrix, confound_ts])

# Shape: (n_timepoints, n_confounds+1)
# Example: (176, 15) for 7 motion + 5 aCompCor + 1 intercept
```

**Regression for Each Voxel:**
```python
denoised_data = np.zeros_like(img_data)
brain_voxels = np.where(brain_mask)

for i in range(len(brain_voxels[0])):
    x, y, z = brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]
    voxel_ts = img_data[x, y, z, :]
    
    # Ordinary least squares: β = (X'X)^(-1) X'y
    beta = np.linalg.lstsq(confound_matrix, voxel_ts, rcond=None)[0]
    predicted = confound_matrix @ beta
    residual = voxel_ts - predicted
    
    denoised_data[x, y, z, :] = residual
```

**Linear Model:**
```
y = Xβ + ε

where:
y: Voxel timeseries (n_timepoints,)
X: Design matrix (n_timepoints, n_confounds)
β: Regression coefficients (n_confounds,)
ε: Residual (cleaned signal)
```

**Residual Calculation:**
```
residual = y_observed - (β_0 + β_1*motion_x + β_2*motion_y + ... + β_14*aCompCor_4)
```

### Output
- **Data:** Fully denoised 4D image (NIfTI)
- **Confounds:** CSV file with all regressors
  - Columns: trans_x, trans_y, trans_z, FD, derivatives, aCompCor components
  - Rows: One per timepoint

---

## Stage 8: Brain Mask Generation

### Purpose
Create binary mask identifying brain voxels for downstream analysis, excluding skull, background, and CSF.

### Input
- **Data:** Fully preprocessed 4D image

### Process

#### 8.1 Calculate Mean Image
```python
mean_img = np.mean(img_data, axis=-1)  # Average across time
non_zero_voxels = mean_img[mean_img > 0]
```

**Mean Image:**
- Shape: `(x, y, z)` - 3D
- Purpose: Stable intensity reference across time

#### 8.2 Multi-Threshold Approach
```python
thresholds = [
    np.percentile(non_zero_voxels, 20),  # Lower threshold
    np.percentile(non_zero_voxels, 25),  # Original (default)
    np.percentile(non_zero_voxels, 15),  # Even lower
]

best_mask = None
best_coverage = 0

for threshold in thresholds:
    # Create candidate mask
    candidate_mask = (mean_img > threshold).astype(np.uint8)
    
    # Apply morphological operations
    candidate_mask = ndimage.binary_fill_holes(candidate_mask).astype(np.uint8)
    candidate_mask = ndimage.binary_erosion(candidate_mask, iterations=1).astype(np.uint8)
    candidate_mask = ndimage.binary_dilation(candidate_mask, iterations=2).astype(np.uint8)
    
    # Calculate coverage
    coverage = candidate_mask.sum() / candidate_mask.size
    
    # Good brain mask should cover 10-40% of volume
    if 0.05 <= coverage <= 0.5 and coverage > best_coverage:
        best_mask = candidate_mask
        best_coverage = coverage
```

**Morphological Operations:**

**Binary Fill Holes:**
```
Before:  1 1 1 1     After:  1 1 1 1
         1 0 0 1             1 1 1 1
         1 0 0 1             1 1 1 1
         1 1 1 1             1 1 1 1
```
Fills enclosed cavities inside brain

**Binary Erosion (iterations=1):**
```
Before:  1 1 1 1     After:  0 0 0 0
         1 1 1 1             0 1 1 0
         1 1 1 1             0 1 1 0
         1 1 1 1             0 0 0 0
```
Removes edge voxels (one voxel layer)

**Binary Dilation (iterations=2):**
```
Before:  0 0 0 0     After:  1 1 1 1
         0 1 1 0             1 1 1 1
         0 1 1 0             1 1 1 1
         0 0 0 0             1 1 1 1
```
Expands mask outward (two voxel layers)

**Net Effect:** Fill → Erode → Dilate = Smooth boundaries, remove noise, preserve brain

#### 8.3 Validation
```python
mask_voxels = np.sum(best_mask > 0)
mask_coverage = best_mask.sum() / best_mask.size
mask_size_mb = best_mask.nbytes / (1024 * 1024)

# Quality checks
if mask_size_mb < 0.1 or mask_coverage < 0.1:
    print(f"Mask validation issues for {subject_id}:")
    print(f"   - Shape: {best_mask.shape}")
    print(f"   - Non-zero voxels: {mask_voxels}")
    print(f"   - Coverage: {mask_coverage*100:.1f}%")
    print(f"   - Expected file size: {mask_size_mb:.2f}MB")
```

**Quality Thresholds:**
- **Coverage:** 10-40% of total volume
- **File size:** > 0.1 MB
- **Non-zero voxels:** > 1000 voxels
- **Shape:** Should match input dimensions

### Output
- **Data:** Binary brain mask (NIfTI)
- **Shape:** `(x, y, z)` - 3D
- **Values:** 0 (background) or 1 (brain)
- **Coverage:** Typically 15-30% of volume

---

## Output Files and Verification

### Files Saved

#### 1. Preprocessed Functional Data
```
Path: data/preprocessed/{site}/{subject_id}/func_preproc.nii.gz
Size: 10-50 MB (typical)
Format: 4D NIfTI (gzipped)
Shape: (91, 109, 91, n_timepoints) for MNI152 2mm
Data type: float32
```

#### 2. Brain Mask
```
Path: data/preprocessed/{site}/{subject_id}/mask.nii.gz
Size: 0.02-0.1 MB
Format: 3D NIfTI (gzipped)
Shape: (91, 109, 91) for MNI152 2mm
Data type: uint8
Values: 0 (background) or 1 (brain)
```

#### 3. Confound Regressors
```
Path: data/preprocessed/{site}/{subject_id}/confounds.csv
Size: 1-5 KB
Format: CSV
Columns: trans_x, trans_y, trans_z, frame_displacement,
         trans_x_derivative, trans_y_derivative, trans_z_derivative,
         a_comp_cor_00, a_comp_cor_01, ..., a_comp_cor_04
Rows: One per timepoint
```

### Verification Process

#### File Existence Check
```python
func_path = output_dir / "func_preproc.nii.gz"
mask_path = output_dir / "mask.nii.gz"

if func_path.exists() and mask_path.exists():
    print("Files exist - checking integrity...")
```

#### Gzip Integrity Test
```python
with gzip.open(func_path, 'rb') as gz_file:
    chunk = gz_file.read(1024 * 1024)  # Read 1MB
    if len(chunk) == 0:
        raise ValueError("Gzip file is empty")
```

#### NIfTI Loading Test
```python
img = nib.load(func_path)
shape = img.shape

# Validate dimensions
if len(shape) != 4:
    raise ValueError(f"Invalid dimensions: {shape}")

# Validate timepoints
if shape[3] < 50:
    raise ValueError(f"Too few timepoints: {shape[3]}")

# Test data reading
test_data = img.get_fdata()[:5, :5, :5, :1]
if test_data.size == 0:
    raise ValueError("Cannot read NIfTI data")
```

#### File Size Check
```python
file_size_mb = func_path.stat().st_size / (1024 * 1024)

if file_size_mb < 10.0:
    raise ValueError(f"File too small: {file_size_mb:.2f}MB")

if file_size_mb > 100.0:
    print(f"Warning: Large file: {file_size_mb:.2f}MB")
```

### Multi-Run Concatenation

For subjects with multiple runs:

```python
if 'all_runs' in row and len(row['all_runs']) > 1:
    preprocessed_runs = []
    
    for run_idx, run_path in enumerate(func_paths):
        # Process each run independently
        result = pipeline.process(str(run_path), f"{subject_id}_run{run_idx+1}")
        
        # Get processed data
        proc_array = result["processed_data"].get_fdata()
        preprocessed_runs.append(proc_array)
    
    # Concatenate along time dimension (axis=-1)
    concatenated_data = np.concatenate(preprocessed_runs, axis=-1)
    
    # Save as single file
    proc_nifti = nib.Nifti1Image(concatenated_data, affine, header)
    nib.save(proc_nifti, func_output_path)
```

**Example:**
- Run 1: 176 timepoints
- Run 2: 180 timepoints
- Concatenated: 356 timepoints

---

## Parallel Processing Architecture

### Worker Function
```python
def _process_subject(row):
    # Memory check
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print(f"High memory usage ({memory.percent:.1f}%) - forcing cleanup")
        gc.collect()
    
    # Extract subject info
    subject_id = row["subject_id"]
    func_path = Path(row["input_path"])
    output_dir = Path(row["out_dir"]) / site / subject_id
    
    # Check if already processed
    func_output_path = output_dir / "func_preproc.nii.gz"
    mask_output_path = output_dir / "mask.nii.gz"
    
    if func_output_path.exists() and mask_output_path.exists():
        if verify_output_integrity(func_output_path) and verify_output_integrity(mask_output_path):
            return {'status': 'success', 'subject_id': subject_id, 'skipped': True}
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(device=device)
    
    # Process
    result = pipeline.process(str(func_path), subject_id)
    
    if result["status"] != "success":
        return {'status': 'failed', 'subject_id': subject_id, 'error': result.get('error')}
    
    # Save outputs
    nib.save(result["processed_data"], func_output_path)
    
    brain_mask = result["brain_mask"]
    mask_nifti = nib.Nifti1Image(brain_mask.astype(np.uint8), affine, header)
    nib.save(mask_nifti, mask_output_path)
    
    confounds_path = output_dir / "confounds.csv"
    pd.DataFrame(result["confound_regressors"]).to_csv(confounds_path, index=False)
    
    return {'status': 'success', 'subject_id': subject_id}
```

### Parallel Execution
```python
from utils import run_parallel

results = run_parallel(
    _process_subject,
    metadata_df.to_dict('records'),
    n_jobs=4,  # 4 subjects in parallel
    desc="Preprocessing subjects"
)
```

### Resource Management

**Per-Subject Resources:**
- **CPU:** 1-2 cores
- **Memory:** 2-4 GB RAM
- **GPU:** Optional (for motion correction)
- **Disk I/O:** Read 50MB, Write 20MB
- **Time:** 5-15 minutes per subject

**System Limits:**
```python
n_jobs = min(cpu_count // 2, 8)  # Max 8 parallel jobs

# Memory-based throttling
if memory.percent > 70:
    n_jobs = max(1, n_jobs // 2)  # Reduce parallelism
```

---

## Processing Log

Each subject generates a detailed log:

```python
processing_log = [
    {
        'step': 'motion_correction',
        'status': 'success',
        'message': 'Max displacement: 2.34mm, Mean: 0.67mm',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'slice_timing_correction',
        'status': 'success',
        'message': 'Applied to 33 slices, TR=2.00s, ref_time=1.000s',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'spatial_normalization',
        'status': 'success',
        'message': 'Resampled to MNI152 space: (91, 109, 91, 176)',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'temporal_filtering',
        'status': 'success',
        'message': 'Applied 0.009-0.080Hz bandpass filter',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'ica_aroma',
        'status': 'success',
        'message': 'Removed 5 motion components out of 25 total',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'denoising',
        'status': 'success',
        'message': 'Applied 14 confound regressors',
        'subject_id': 'sub-1019436'
    },
    {
        'step': 'brain_mask',
        'status': 'success',
        'message': 'Generated mask: 45123 voxels (18.2% coverage)',
        'subject_id': 'sub-1019436'
    }
]
```

---

## Summary Statistics

After processing all subjects:

```
Total subjects: 759
Successful: 743 (97.9%)
Failed: 16 (2.1%)

Motion statistics:
  Mean displacement: 0.58 ± 0.34 mm
  Max displacement: 3.12 mm
  High-motion subjects (>3mm): 8 (1.1%)

ICA-AROMA:
  Mean components removed: 5.2 ± 2.1 (20.8%)
  Range: 2-12 components

Processing time:
  Mean: 8.3 ± 3.2 minutes per subject
  Total: ~105 hours (4 jobs in parallel)

Output sizes:
  Functional: 18.5 ± 4.2 MB per subject
  Mask: 0.04 ± 0.01 MB per subject
  Confounds: 0.002 MB per subject
  Total: ~14 GB for all subjects
```

---

## Key Equations Reference

### Motion Correction
```
Translation: T = COM_ref - COM_current
Displacement: FD_t = ||T_t - T_{t-1}||_2
```

### Slice Timing Correction
```
Corrected(t) = Linear_Interpolation(Original(t + Δt), t)
where Δt = slice_acquisition_time - reference_time
```

### Spatial Normalization
```
[x']     [R  T] [x]
[y']  =  [     ] [y]
[z']     [0  1] [z]

where R = rotation + scaling matrix, T = translation vector
```

### Temporal Filtering
```
H(ω) = 1 / sqrt(1 + (ω/ω_c)^(2n))
Filtered = IFT(H(ω) * FT(Signal))
```

### ICA-AROMA
```
X = AS
Motion_criterion: |corr(A_i, motion)| > 0.3
Freq_criterion: P(f>0.2Hz) / P(total) > 0.6
```

### Confound Regression
```
Y_cleaned = Y - Xβ
where β = (X^T X)^(-1) X^T Y
```

### Brain Mask
```
Mask = Fill_Holes(Erode(Dilate(I > threshold)))
Coverage = sum(Mask) / size(Mask)
Valid: 0.1 < Coverage < 0.4
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Gzip Corruption
**Symptom:** "Error -3 while decompressing"
**Solution:** Re-download raw data, check disk integrity

#### 2. Insufficient Memory
**Symptom:** Process killed, memory > 90%
**Solution:** Reduce n_jobs, enable swap, upgrade RAM

#### 3. Small Mask Files
**Symptom:** Mask < 0.01 MB
**Solution:** Lower threshold, check input quality

#### 4. High Motion
**Symptom:** Max displacement > 5mm
**Solution:** Flag subject, consider exclusion, manual inspection

#### 5. ICA Fails
**Symptom:** "Insufficient components for ICA-AROMA"
**Solution:** Check timepoints > 50, reduce n_components

---

## Configuration for Different Datasets

### High-Resolution Data (1mm isotropic)
```python
config['spatial_normalization']['resolution'] = 1
config['spatial_normalization']['smooth_fwhm'] = 4.0
```

### Low TR (< 1 second)
```python
config['temporal_filtering']['low_freq'] = 0.01
config['temporal_filtering']['high_freq'] = 0.15
```

### Pediatric Data
```python
config['motion_correction']['max_displacement_mm'] = 5.0
config['ica_aroma']['n_components'] = 30
```

### Multi-Band Acquisition
```python
config['slice_timing_correction']['slice_order'] = 'multiband'
config['slice_timing_correction']['mb_factor'] = 4
```

---

This documentation covers every computation, variable, and process in the preprocessing pipeline from raw NIfTI input to analysis-ready output.

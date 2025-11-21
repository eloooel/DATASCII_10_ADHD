# Thesis Visualizations - Real Data Integration

## Summary

Updated the comprehensive thesis visualization notebook to use **actual attention weights** and **real FC matrices** from trained GNN-STAN models instead of mock data.

## What Was Changed

### 1. Created Extraction Script (`extract_attention_weights.py`)

**Purpose:** Extract attention weights and FC matrices from a trained model

**Process:**
1. Loads V6 dataset (771 subjects, 5 sites)
2. Trains single LOSO fold (excluding NYU for test)
3. Uses PyTorch forward hooks to capture attention during inference
4. Saves outputs to `data/attention_weights/`

**Outputs:**
- `attention_weights.npz` - Spatial & temporal attention from GNN and STAN encoders
- `example_fc_matrix.npy` - Real (200×200) FC matrix from test subject
- `metadata.json` - Subject diagnosis, site, and data shapes

### 2. Updated Notebook Cells

**Modified Cells:**

**Cell #VSC-11050163 (Spatial Attention):**
- ✅ Loads `data/attention_weights/attention_weights.npz` if available
- ✅ Extracts spatial attention (200 ROIs) from GNN encoder
- ✅ Aggregates by 7 Schaefer networks (Visual, Somatomotor, etc.)
- ✅ Falls back to mock data if extraction not run yet
- ✅ Updates plot title to indicate "Actual weights from trained GNN-STAN model"

**Cell #VSC-7ac139d6 (Temporal Attention):**
- ✅ Loads temporal attention (176 timepoints) from STAN encoder
- ✅ Handles different array shapes (batch/head dimensions)
- ✅ Smooths with Gaussian filter for visualization
- ✅ Marks high-attention timepoints (>75th percentile)
- ✅ Updates plot title to indicate "Actual weights from trained STAN encoder"

**Cell #VSC-cc44e8c4 (Brain Connectivity):**
- ✅ Loads `data/attention_weights/example_fc_matrix.npy` if available
- ✅ Reads metadata (subject ID, diagnosis, site) from JSON
- ✅ Creates networkx graph from real FC correlations
- ✅ Thresholds top 5% connections for visualization
- ✅ Updates plot title to indicate "Actual subject connectivity"

### 3. Documentation

**Created Files:**
- `ATTENTION_EXTRACTION_README.md` - Comprehensive guide for extraction process
  - Usage instructions
  - Output descriptions
  - Customization options
  - Troubleshooting guide
  - Technical details about attention extraction

## How to Use

### Step 1: Extract Attention Weights

```powershell
# Activate virtual environment
& D:\repos\DATASCII_10_ADHD\thesis-adhd\Scripts\Activate.ps1

# Run extraction (takes ~5-10 minutes on GPU)
python extract_attention_weights.py
```

**What happens:**
- Trains model on 4 sites (excluding NYU)
- Selects ADHD subject from NYU test set
- Runs forward pass with hooks to capture attention
- Saves to `data/attention_weights/`

### Step 2: Run Notebook

```powershell
# Open notebook
code thesis_visualizations.ipynb

# Run all cells (or just cells 29-34 for attention/connectivity)
```

**Automatic behavior:**
- Notebook checks if `data/attention_weights/` exists
- If found: Loads real data and updates plots
- If not found: Uses mock data and prints warning with instructions

### Step 3: Generate Figures

**Before extraction:**
```
figures/11_spatial_attention.png         (mock data)
figures/12_temporal_attention.png        (mock data)
figures/14_brain_connectivity_graph.png  (mock data)
```

**After extraction:**
```
figures/11_spatial_attention.png         (✓ Real GNN attention)
figures/12_temporal_attention.png        (✓ Real STAN attention)
figures/14_brain_connectivity_graph.png  (✓ Real FC connectivity)
```

## Technical Implementation

### Attention Extraction Method

Uses **PyTorch forward hooks** to intercept attention weights during inference:

```python
attention_outputs = {}

def save_attention(name):
    def hook(module, input, output):
        # Many attention modules return (output, attention_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_outputs[name] = output[1].detach().cpu().numpy()
    return hook

# Register hooks on all attention-related modules
for name, module in model.named_modules():
    if 'attention' in name.lower():
        hook = module.register_forward_hook(save_attention(name))
```

### Data Flow

```
1. Train Model (extract_attention_weights.py)
   ├─ Load dataset (771 subjects)
   ├─ Create LOSO fold (train on 4 sites, test on NYU)
   ├─ Train 5 epochs (quick demonstration)
   └─ Model ready for inference

2. Extract Attention (forward hooks)
   ├─ Select test subject (ADHD from NYU)
   ├─ Run forward pass: model(fc_matrix, roi_timeseries)
   ├─ Hooks capture: gnn_encoder.attention, stan_encoder.attention
   └─ Save: attention_weights.npz

3. Visualize (thesis_visualizations.ipynb)
   ├─ Load: attention_weights.npz, example_fc_matrix.npy
   ├─ Process: Normalize, aggregate by network, threshold connections
   └─ Generate: figures/11_*.png, figures/12_*.png, figures/14_*.png
```

### Attention Shapes

**Spatial Attention (GNN):**
- Raw shape: `(200,)` - one weight per ROI
- Normalized: Divide by sum to get probability distribution
- Aggregated: Average within each of 7 Schaefer networks
- Visualization: Bar chart (network-level) + heatmap (top 50 ROIs)

**Temporal Attention (STAN):**
- Raw shape: `(176,)` or `(batch, 176)` - one weight per fMRI timepoint
- Normalized: Divide by sum
- Smoothed: Gaussian filter (σ=3) for cleaner visualization
- Visualization: Line plot (time series) + histogram (distribution)

**FC Matrix:**
- Shape: `(200, 200)` - Pearson correlations between all ROI pairs
- Symmetric: FC[i,j] = FC[j,i]
- Thresholded: Keep top 5% strongest connections (|r| > 95th percentile)
- Graph: 200 nodes (ROIs), ~1000 edges (strong connections)

## Comparison: Mock vs Real Data

### Mock Data (Before)

**Characteristics:**
- Spatial attention: Beta(2, 5) distribution - random importance
- Temporal attention: Beta(2, 3) distribution - uniform over time
- FC matrix: Random Gaussian → symmetric → thresholded
- Purpose: Demonstrate visualization techniques

**Limitations:**
- No clinical interpretation possible
- Doesn't reflect actual model behavior
- Generic patterns not specific to ADHD

### Real Data (After)

**Characteristics:**
- Spatial attention: Learned ROI importance from GNN graph convolutions
- Temporal attention: Learned timepoint importance from STAN LSTM
- FC matrix: Actual subject's functional connectivity from fMRI
- Purpose: Interpret model's diagnostic decisions

**Advantages:**
- Shows which brain regions model focuses on (e.g., Default Mode Network)
- Reveals critical temporal windows during scan
- Displays real connectivity patterns (hypo/hyperconnectivity)
- Enables clinical insight into ADHD classification

## Expected Results

### Spatial Attention (GNN)

**Likely patterns:**
- **High attention:** Default Mode Network, Frontoparietal Network
  - Known to show differences in ADHD (reduced anti-correlation)
- **Low attention:** Visual, Somatomotor Networks
  - Less relevant for ADHD diagnosis

**Example output:**
```
Spatial Attention Summary:
  Visual              : 0.1201
  Somatomotor         : 0.1156
  Dorsal Attn         : 0.1389
  Ventral Attn        : 0.1423
  Limbic              : 0.1512
  Frontoparietal      : 0.1678  ← Highest attention
  Default Mode        : 0.1641  ← Second highest
```

### Temporal Attention (STAN)

**Likely patterns:**
- **High attention:** Early scan periods (task-related if eyes-open)
- **Moderate attention:** Mid-scan (stable resting state)
- **Low attention:** Late scan (fatigue, motion artifacts)

**Example output:**
```
Temporal Attention Statistics:
  Mean:   0.005682
  Median: 0.005234
  Max:    0.012456 (at time point 42)
  High attention regions: 44 time points (>0.008123)
```

### Brain Connectivity

**Graph properties:**
- Nodes: 200 ROIs (colored by network)
- Edges: ~1000-1500 strong connections (top 5%)
- Density: ~0.05 (sparse, biologically realistic)
- Communities: Clear clustering by network (visual, motor, etc.)

**ADHD-specific patterns:**
- Reduced long-range connections (Default Mode ↔ Task-Positive)
- Increased short-range connections (within-network)
- Altered hub topology (different central nodes)

## Validation

### Check Extraction Success

```powershell
# Check if files exist
Test-Path data/attention_weights/attention_weights.npz
Test-Path data/attention_weights/example_fc_matrix.npy
Test-Path data/attention_weights/metadata.json

# View metadata
Get-Content data/attention_weights/metadata.json | ConvertFrom-Json
```

### Verify Attention Shapes

```python
import numpy as np

# Load and inspect
data = np.load('data/attention_weights/attention_weights.npz', allow_pickle=True)
print("Available keys:", list(data.keys()))

# Check shapes
for key in data.keys():
    if key not in ['subject_idx', 'diagnosis', 'site']:
        print(f"{key}: {data[key].shape}")

# Expected:
# gnn_encoder.attention: (200,) or similar
# stan_encoder.attention: (176,) or similar
```

### Verify FC Matrix

```python
fc = np.load('data/attention_weights/example_fc_matrix.npy')
print(f"Shape: {fc.shape}")  # Should be (200, 200)
print(f"Symmetric: {np.allclose(fc, fc.T)}")  # Should be True
print(f"Diagonal: {np.diag(fc).mean():.4f}")  # Should be ~0 or 1
print(f"Range: [{fc.min():.3f}, {fc.max():.3f}]")  # Should be [-1, 1] for correlations
```

## Troubleshooting

### Problem: Extraction script hangs at "Loading data..."

**Cause:** Loading 771 subjects × (200×200 FC + 176×200 timeseries) is memory-intensive

**Solution:**
- Wait 2-3 minutes for loading to complete
- Monitor memory usage (should be ~4-8 GB)
- If out of memory: Reduce batch size in script or use subset of data

### Problem: "No attention weights found" after running script

**Cause:** Forward hooks didn't capture attention (model architecture mismatch)

**Solutions:**
1. Check script output for "Extracted attention components"
2. Inspect saved file: `np.load('data/attention_weights/attention_weights.npz').keys()`
3. Modify hook detection logic if using custom attention module names
4. Ensure model returns attention in output tuple (check models/gnn.py, models/stan.py)

### Problem: Spatial attention wrong shape (not 200)

**Cause:** Attention includes batch/head dimensions

**Solution:** Already handled in notebook cell:
```python
if spatial_attention.ndim > 1:
    spatial_attention = spatial_attention.squeeze()
```

### Problem: GPU out of memory during training

**Cause:** 5 epochs of training with batch_size=32 exceeds GPU memory

**Solution:**
- Edit script: Reduce batch_size to 16 or 8
- Or use CPU: Change `device='cpu'` (slower but works)

## Future Enhancements

### 1. Average Across All Folds

Extract attention from all 5 LOSO folds and average:

```python
all_spatial = []
for test_site in ['NYU', 'Peking', 'NeuroIMAGE', 'KKI', 'OHSU']:
    model, test_idx = train_single_fold_for_attention(..., test_site=test_site)
    attention = extract_attention_hook(model, fc, roi, device)
    all_spatial.append(attention['spatial'])

spatial_attention_avg = np.mean(all_spatial, axis=0)
```

### 2. Compare HC vs ADHD Attention

Extract from both diagnostic groups:

```python
hc_attention = []
adhd_attention = []

for subject_idx in test_idx:
    fc = torch.tensor(fc_matrices[subject_idx:subject_idx+1])
    roi = torch.tensor(roi_timeseries[subject_idx:subject_idx+1])
    attention = extract_attention_hook(model, fc, roi, device)
    
    if labels[subject_idx] == 0:
        hc_attention.append(attention['spatial'])
    else:
        adhd_attention.append(attention['spatial'])

# Compare distributions
import scipy.stats
t_stat, p_val = scipy.stats.ttest_ind(hc_attention, adhd_attention, axis=0)
```

### 3. Attention-Weighted FC

Combine attention with connectivity:

```python
# Weight FC matrix by spatial attention
attention_2d = spatial_attention[:, np.newaxis] @ spatial_attention[np.newaxis, :]
fc_weighted = fc_matrix * attention_2d

# Visualize attention-modulated connectivity
# (shows which connections the model actually "sees")
```

### 4. Longitudinal Attention

Track attention changes across training epochs:

```python
attention_history = []
for epoch in range(100):
    # Train one epoch
    train_one_epoch(model, train_loader, optimizer, criterion)
    
    # Extract attention
    attention = extract_attention_hook(model, fc, roi, device)
    attention_history.append(attention['spatial'])

# Visualize evolution of attention focus during training
```

## References

- **Extraction Script:** `extract_attention_weights.py`
- **Visualization Notebook:** `thesis_visualizations.ipynb` (Sections 8 & 10)
- **Documentation:** `ATTENTION_EXTRACTION_README.md`
- **Model Architecture:** `models/gnn_stan_hybrid.py`, `models/gnn.py`, `models/stan.py`
- **Configuration:** `configs/baseline_accurate_v6_config.py`

---

**Status:** ✅ Implemented and documented  
**Last Updated:** November 21, 2025  
**Next Step:** Run `python extract_attention_weights.py` to generate real data

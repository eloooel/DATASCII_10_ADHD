# Attention Weight Extraction Guide

This guide explains how to extract actual attention weights and FC matrices from trained models for visualization in the thesis notebook.

## Overview

The `extract_attention_weights.py` script trains a single fold of the GNN-STAN model and extracts:
- **Spatial attention weights** from the GNN encoder (which ROIs are important)
- **Temporal attention weights** from the STAN encoder (which timepoints are important)
- **Example FC matrix** from a test subject for connectivity visualization

## Usage

### 1. Activate Virtual Environment

```powershell
& D:\repos\DATASCII_10_ADHD\thesis-adhd\Scripts\Activate.ps1
```

### 2. Run Extraction Script

```powershell
python extract_attention_weights.py
```

This will:
1. Load the V6 dataset (5 sites, 771 subjects)
2. Train a model on a single LOSO fold (excluding NYU, ~5 epochs for quick demo)
3. Extract attention weights using forward hooks
4. Save outputs to `data/attention_weights/`

**Expected runtime:** ~5-10 minutes on GPU, ~20-30 minutes on CPU

### 3. Outputs

The script creates `data/attention_weights/` directory with:

- **`attention_weights.npz`**: NumPy archive containing attention weights
  - Keys depend on model architecture (e.g., `gnn_encoder.attention`, `stan_encoder.attention`)
  - Also includes `subject_idx`, `diagnosis`, `site` metadata

- **`example_fc_matrix.npy`**: (200×200) FC matrix from test subject
  - Used for brain connectivity graph visualization
  - Symmetric matrix with correlations between ROIs

- **`metadata.json`**: Subject information
  ```json
  {
    "subject_idx": 123,
    "diagnosis": "ADHD",
    "site": "NYU",
    "attention_components": ["gnn_encoder.attention", "stan_encoder.attention"],
    "fc_matrix_shape": [200, 200],
    "roi_timeseries_shape": [176, 200]
  }
  ```

## Using in Thesis Notebook

After running the extraction script:

1. Open `thesis_visualizations.ipynb`
2. Run all cells - the notebook will automatically:
   - Load `data/attention_weights/attention_weights.npz`
   - Load `data/attention_weights/example_fc_matrix.npy`
   - Use actual data instead of mock data
3. Figures 11-14 will now show **real attention patterns** and **actual connectivity**

### Before Extraction (Mock Data)

```
⚠️  No attention weights found. Run extract_attention_weights.py first.
   Using mock data for demonstration...
```

### After Extraction (Real Data)

```
✓ Loading actual attention weights from trained model...
  Available attention components: ['gnn_encoder.attention', 'stan_encoder.attention']
  ✓ Found spatial attention: gnn_encoder.attention, shape: (200,)
  ✓ Found temporal attention: stan_encoder.attention, shape: (176,)

✓ Using actual spatial attention weights from trained model
```

## Customization

### Extract from Different Subject

Edit `extract_attention_weights.py`:

```python
# Line ~140: Choose different test subject
adhd_subjects = test_idx[labels[test_idx] == 1]
subject_idx = adhd_subjects[5]  # Change index to select different subject
```

### Train on Different Fold

```python
# Line ~123: Change test site
model, test_idx = train_single_fold_for_attention(
    fc_matrices, roi_timeseries, labels, sites,
    test_site='Peking',  # Change from 'NYU' to 'Peking', 'NeuroIMAGE', etc.
    device=device
)
```

### Extract from Full Training

For production-quality attention extraction:

1. Modify script to train for full 100 epochs
2. Use early stopping and best checkpoint
3. Average attention across all LOSO folds

```python
# Full training (replace quick 5-epoch training)
for epoch in range(TRAINING_CONFIG_BASELINE['max_epochs']):
    # ... full training loop with validation
```

## Technical Details

### Attention Extraction Method

The script uses PyTorch **forward hooks** to capture attention weights:

```python
def save_attention(name):
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attention_outputs[name] = output[1].detach().cpu().numpy()
    return hook

# Register hooks on attention modules
for name, module in model.named_modules():
    if 'attention' in name.lower():
        hook = module.register_forward_hook(save_attention(name))
```

### Spatial Attention (GNN Encoder)

- **Shape:** (200,) - one weight per ROI
- **Interpretation:** Higher weight = ROI receives more focus in graph convolution
- **Aggregation:** Network-level attention computed by averaging ROIs within each of 7 Schaefer networks

### Temporal Attention (STAN Encoder)

- **Shape:** (176,) - one weight per fMRI timepoint (TR)
- **Interpretation:** Higher weight = timepoint has more influence on classification
- **Usage:** Identifies critical temporal windows during scan

### FC Matrix

- **Shape:** (200, 200) - correlation matrix
- **Properties:** Symmetric, zero diagonal
- **Thresholding:** Top 5% connections kept for graph visualization
- **Networks:** 7 Schaefer networks (Visual, Somatomotor, Dorsal Attn, Ventral Attn, Limbic, Frontoparietal, Default Mode)

## Troubleshooting

### "No attention weights found"

**Problem:** Attention modules not detected by forward hooks

**Solutions:**
1. Check model architecture - ensure attention modules exist
2. Inspect available keys: `print(list(attention_data.keys()))`
3. Modify hook detection logic if attention uses custom names

### "Spatial attention wrong size"

**Problem:** Attention shape doesn't match 200 ROIs

**Cause:** Attention may be per-graph-node instead of per-ROI

**Solution:**
```python
# Reshape or aggregate attention
if spatial_attention.shape[0] > 200:
    # Batch dimension present, take first subject
    spatial_attention = spatial_attention[0]
elif spatial_attention.shape[0] < 200:
    # Multi-head attention, average across heads
    spatial_attention = spatial_attention.mean(axis=0)
```

### "Temporal attention not found"

**Problem:** STAN encoder doesn't return attention in output tuple

**Solution:** Modify `models/stan.py` to return attention:

```python
class EnhancedSTANBranch(nn.Module):
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_weights, context = self.attention(lstm_out)
        # Return both output and attention
        return output, attn_weights  # Add second return value
```

## Alternative: Load from Saved Checkpoints

If you have pre-trained checkpoints with `save_models=True`:

```python
# Load checkpoint
checkpoint = torch.load('data/trained/baseline_accurate_v6/run_1/fold_0.pt')
model.load_state_dict(checkpoint['model_state'])

# Run inference to extract attention
model.eval()
with torch.no_grad():
    logits = model(fc_matrix, roi_timeseries)
    attention = extract_attention_hook(model, fc_matrix, roi_timeseries, device)
```

## References

- **Model Architecture:** `models/gnn_stan_hybrid.py`
- **GNN Encoder:** `models/gnn.py` (EnhancedGNNBranch)
- **STAN Encoder:** `models/stan.py` (EnhancedSTANBranch)
- **Training Config:** `configs/baseline_accurate_v6_config.py`
- **Visualization Notebook:** `thesis_visualizations.ipynb` (Sections 8 & 10)

---

**Last Updated:** November 21, 2025  
**Status:** Ready for extraction ✓

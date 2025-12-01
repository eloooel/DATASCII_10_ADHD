"""
Extract attention weights from trained GNN-STAN model
Loads a subject, runs forward pass, and extracts spatial/temporal attention
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import GNNSTANHybrid
from configs.baseline_accurate_v6_config import MODEL_CONFIG_BASELINE, TRAINING_CONFIG_BASELINE
from utils.data_loader import load_features_and_labels


def extract_attention_from_model(model, fc_matrix, roi_timeseries, device):
    """
    Extract attention weights from a forward pass
    
    Args:
        model: Trained GNN-STAN model
        fc_matrix: Functional connectivity (1, n_rois, n_rois)
        roi_timeseries: ROI time series (1, n_timepoints, n_rois)
        device: torch device
        
    Returns:
        Dictionary with spatial and temporal attention weights
    """
    model.eval()
    
    with torch.no_grad():
        fc_matrix = fc_matrix.to(device)
        roi_timeseries = roi_timeseries.to(device)
        
        # Forward pass through model
        logits = model(fc_matrix, roi_timeseries)
        
        # Extract attention weights from model components
        attention_weights = {}
        
        # Try to extract spatial attention from GNN encoder
        if hasattr(model, 'gnn_encoder'):
            if hasattr(model.gnn_encoder, 'attention_weights'):
                attention_weights['spatial_attention'] = model.gnn_encoder.attention_weights.cpu().numpy()
            elif hasattr(model.gnn_encoder, 'last_attention'):
                attention_weights['spatial_attention'] = model.gnn_encoder.last_attention.cpu().numpy()
        
        # Try to extract temporal attention from STAN encoder
        if hasattr(model, 'stan_encoder'):
            if hasattr(model.stan_encoder, 'attention_weights'):
                attention_weights['temporal_attention'] = model.stan_encoder.attention_weights.cpu().numpy()
            elif hasattr(model.stan_encoder, 'last_attention'):
                attention_weights['temporal_attention'] = model.stan_encoder.last_attention.cpu().numpy()
        
        # Extract fusion attention if available
        if hasattr(model, 'fusion_layer'):
            if hasattr(model.fusion_layer, 'attention_weights'):
                attention_weights['fusion_attention'] = model.fusion_layer.attention_weights.cpu().numpy()
        
        return attention_weights, logits


def extract_attention_hook(model, fc_matrix, roi_timeseries, device):
    """
    Extract attention using forward hooks and aggregate multi-head attention
    """
    attention_outputs = {}
    
    def save_attention(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Many attention modules return (output, attention_weights)
                if len(output) > 1:
                    attention_outputs[name] = output[1].detach().cpu().numpy()
            elif 'attention' in name.lower():
                attention_outputs[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            hook = module.register_forward_hook(save_attention(name))
            hooks.append(hook)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(fc_matrix.to(device), roi_timeseries.to(device))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Post-process attention weights for visualization
    processed_attention = {}
    
    for name, weights in attention_outputs.items():
        # Store raw attention
        processed_attention[f'{name}_raw'] = weights
        
        # Aggregate multi-head attention if present
        if len(weights.shape) == 4 and 'attention' in name and 'dropout' not in name:
            # Shape: (batch, heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = weights.shape
            
            # Average across heads to get (batch, seq_len, seq_len)
            avg_attention = weights.mean(axis=1)
            processed_attention[f'{name}_avg_heads'] = avg_attention
            
            # Get attention weights for each position (average across all other positions)
            # This gives us the "importance" of each timepoint/ROI
            position_attention = avg_attention.mean(axis=-1)  # (batch, seq_len)
            processed_attention[f'{name}_position_importance'] = position_attention
    
    return processed_attention


def train_single_fold_for_attention(fc_matrices, roi_timeseries, labels, sites, 
                                     test_site='NYU', device='cuda'):
    """
    Train a single fold to get a model we can extract attention from
    """
    from training.dataset import ADHDDataset
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    print(f"Training model on fold excluding {test_site}...")
    
    # Create train/test split
    test_mask = sites == test_site
    train_mask = ~test_mask
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print(f"Train: {len(train_idx)} subjects, Test: {len(test_idx)} subjects")
    
    # Create datasets (use keyword arguments for array mode)
    train_dataset = ADHDDataset(
        fc_matrices=fc_matrices[train_idx],
        roi_timeseries=roi_timeseries[train_idx],
        labels=labels[train_idx]
    )
    
    # Create balanced sampler
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights_np = 1.0 / class_counts
    sample_weights = class_weights_np[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG_BASELINE['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = GNNSTANHybrid(**MODEL_CONFIG_BASELINE).to(device)
    
    # Setup training
    # Use CrossEntropyLoss with class weights for simplicity
    class_weights = torch.tensor([1.0, 4.0], device=device, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG_BASELINE['learning_rate'],
        weight_decay=TRAINING_CONFIG_BASELINE['weight_decay']
    )
    
    # Quick training (5 epochs for demonstration)
    print("Training for 5 epochs...")
    model.train()
    
    for epoch in range(5):
        epoch_loss = 0
        for batch in train_loader:
            fc = batch['fc_matrix'].to(device)
            roi = batch['roi_timeseries'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(fc, roi)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5: Loss = {epoch_loss/len(train_loader):.4f}")
    
    print("Success: Training complete")
    return model, test_idx


def main():
    """Extract attention weights and save for visualization"""
    
    print("="*80)
    print("ATTENTION WEIGHT EXTRACTION")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    feature_manifest_path = Path('data/features/feature_manifest.csv')
    manifest_df = pd.read_csv(feature_manifest_path)
    manifest_df = manifest_df[manifest_df['site'].isin(TRAINING_CONFIG_BASELINE['sites'])]
    
    fc_matrices, roi_timeseries, labels, sites = load_features_and_labels(manifest_df)
    print(f"   Loaded {len(labels)} subjects")
    
    # Train model on single fold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    print("\n2. Training model on single fold (excluding NYU for test)...")
    model, test_idx = train_single_fold_for_attention(
        fc_matrices, roi_timeseries, labels, sites,
        test_site='NYU', device=device
    )
    
    # Select a test subject (ADHD positive for interesting patterns)
    adhd_subjects = test_idx[labels[test_idx] == 1]
    if len(adhd_subjects) > 0:
        subject_idx = adhd_subjects[0]
    else:
        subject_idx = test_idx[0]
    
    print(f"\n3. Extracting attention from subject {subject_idx}...")
    print(f"   Diagnosis: {'ADHD' if labels[subject_idx] == 1 else 'HC'}")
    print(f"   Site: {sites[subject_idx]}")
    
    # Prepare subject data
    fc = torch.tensor(fc_matrices[subject_idx:subject_idx+1], dtype=torch.float32)
    roi = torch.tensor(roi_timeseries[subject_idx:subject_idx+1], dtype=torch.float32)
    
    # Extract attention using hooks
    attention_weights = extract_attention_hook(model, fc, roi, device)
    
    print(f"\n4. Extracted attention components:")
    for name, weights in attention_weights.items():
        print(f"   - {name}: shape {weights.shape}")
    
    # Save attention weights
    output_dir = Path('data/attention_weights')
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / 'attention_weights.npz',
        **attention_weights,
        subject_idx=subject_idx,
        diagnosis=labels[subject_idx],
        site=sites[subject_idx]
    )
    
    print(f"\nSuccess: Attention weights saved to: {output_dir / 'attention_weights.npz'}")
    
    # Also save the subject's FC matrix for connectivity visualization
    subject_fc = fc_matrices[subject_idx]
    np.save(output_dir / 'example_fc_matrix.npy', subject_fc)
    print(f"Success: FC matrix saved to: {output_dir / 'example_fc_matrix.npy'}")
    
    # Save metadata
    metadata = {
        'subject_idx': int(subject_idx),
        'diagnosis': 'ADHD' if labels[subject_idx] == 1 else 'HC',
        'site': str(sites[subject_idx]),
        'attention_components': list(attention_weights.keys()),
        'fc_matrix_shape': list(subject_fc.shape),
        'roi_timeseries_shape': list(roi_timeseries[subject_idx].shape)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Success: Metadata saved to: {output_dir / 'metadata.json'}")
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print("\nYou can now run the notebook cells to visualize:")
    print("  - Spatial attention weights (from GNN encoder)")
    print("  - Temporal attention weights (from STAN encoder)")
    print("  - Actual FC matrix connectivity graph")
    
    return attention_weights, subject_fc


if __name__ == '__main__':
    main()

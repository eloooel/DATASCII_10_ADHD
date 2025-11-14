import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path

class ADHDDataset(Dataset):
    """Dataset for ADHD classification with FC matrices and time series"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with columns: subject_id, site, fc_path, ts_path, diagnosis
        """
        self.data = data.reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            # Load FC matrix
            fc_path = Path(row['fc_path'])
            if fc_path.suffix == '.npy':
                fc_matrix = np.load(fc_path, allow_pickle=False).astype(np.float32)
            else:
                fc_matrix = pd.read_csv(fc_path, header=None).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
                if np.any(np.isnan(fc_matrix)):
                    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
            
            # Load time series
            ts_path = Path(row['ts_path'])
            if ts_path.suffix == '.npy':
                timeseries = np.load(ts_path, allow_pickle=False).astype(np.float32)
            else:
                # Load CSV and convert to numeric, handling any object columns
                ts_df = pd.read_csv(ts_path)
                # Convert all columns to numeric, coercing errors to NaN
                timeseries = ts_df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
                
                # Check for NaN values and replace with 0
                if np.any(np.isnan(timeseries)):
                    timeseries = np.nan_to_num(timeseries, nan=0.0)
            
            # Get label
            label = int(row['diagnosis'])  # 0 for control, 1 for ADHD
            
            # Ensure proper shapes and types
            fc_matrix = torch.from_numpy(fc_matrix.astype(np.float32)).float()
            timeseries = torch.from_numpy(timeseries.astype(np.float32)).float()
            
            # Validate shapes
            if fc_matrix.dim() != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
                raise ValueError(f"Invalid FC matrix shape: {fc_matrix.shape}, expected square matrix")
            if timeseries.dim() != 2:
                raise ValueError(f"Invalid timeseries shape: {timeseries.shape}, expected 2D array (timepoints, ROIs)")
            
            return {
                'fc_matrix': fc_matrix,
                'timeseries': timeseries,
                'label': torch.tensor(label, dtype=torch.long),
                'subject_id': row['subject_id'],
                'site': row.get('site', 'unknown')
            }
            
        except Exception as e:
            raise RuntimeError(f"Error loading data for subject {row['subject_id']}: {str(e)}")


def collate_fn(batch):
    """
    Custom collate function to handle variable-length timeseries
    Pads timeseries to the maximum length in the batch
    """
    # Separate components
    fc_matrices = [item['fc_matrix'] for item in batch]
    timeseries_list = [item['timeseries'] for item in batch]
    labels = [item['label'] for item in batch]
    subject_ids = [item['subject_id'] for item in batch]
    sites = [item['site'] for item in batch]
    
    # Stack FC matrices (all should be same size)
    fc_matrices = torch.stack(fc_matrices)
    
    # Pad timeseries to max length in batch
    # timeseries shape from dataset: (time_steps, n_rois)
    # Find max time_steps
    max_len = max(ts.shape[0] for ts in timeseries_list)
    n_rois = timeseries_list[0].shape[1]
    
    # Manually pad each timeseries to max_len
    padded_timeseries = []
    for ts in timeseries_list:
        current_len = ts.shape[0]
        if current_len < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - current_len, n_rois)
            ts_padded = torch.cat([ts, padding], dim=0)
        else:
            ts_padded = ts
        padded_timeseries.append(ts_padded)
    
    # Stack to (batch, time_steps, n_rois) then transpose to (batch, n_rois, time_steps)
    padded_timeseries = torch.stack(padded_timeseries).transpose(1, 2)
    
    # Stack labels
    labels = torch.stack(labels)
    
    return {
        'fc_matrix': fc_matrices,
        'timeseries': padded_timeseries,
        'label': labels,
        'subject_id': subject_ids,
        'site': sites
    }

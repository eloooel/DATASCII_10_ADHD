import torch
from torch.utils.data import Dataset
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
        
        # Load FC matrix
        fc_path = Path(row['fc_path'])
        if fc_path.suffix == '.npy':
            fc_matrix = np.load(fc_path)
        else:
            fc_matrix = pd.read_csv(fc_path, header=None).values
        
        # Load time series
        ts_path = Path(row['ts_path'])
        if ts_path.suffix == '.npy':
            timeseries = np.load(ts_path)
        else:
            timeseries = pd.read_csv(ts_path, header=None).values
        
        # Get label
        label = int(row['diagnosis'])  # 0 for control, 1 for ADHD
        
        return {
            'fc_matrix': torch.FloatTensor(fc_matrix),
            'timeseries': torch.FloatTensor(timeseries),
            'label': torch.LongTensor([label])[0],
            'subject_id': row['subject_id'],
            'site': row.get('site', 'unknown')
        }

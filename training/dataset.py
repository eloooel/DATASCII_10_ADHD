import torch
from torch.utils.data import Dataset
from typing import Optional
import numpy as np

class ADHDDataset(Dataset):
    """Dataset class for ADHD rs-fMRI data"""

    def __init__(
        self,
        fc_matrices: np.ndarray,
        roi_timeseries: np.ndarray,
        labels: np.ndarray,
        sites: np.ndarray,
        demographics: Optional[np.ndarray] = None,
        augment: bool = False
    ):
        self.fc_matrices = torch.FloatTensor(fc_matrices)
        self.roi_timeseries = torch.FloatTensor(roi_timeseries)
        self.labels = torch.LongTensor(labels)
        self.sites = sites
        self.demographics = demographics
        self.augment = augment
        self.training = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fc_matrix = self.fc_matrices[idx]
        roi_ts = self.roi_timeseries[idx]
        label = self.labels[idx]
        site = self.sites[idx]
        return {
            'fc_matrix': fc_matrix,
            'roi_timeseries': roi_ts,
            'label': label,
            'site': site
        }

    @staticmethod
    def collate_fn(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}

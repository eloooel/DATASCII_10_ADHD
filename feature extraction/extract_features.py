"""
Feature Extraction Module for ADHD rs-fMRI Pipeline
Computes functional connectivity matrices and saves ROI timeseries
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class FunctionalConnectivityExtractor:
    """Computes functional connectivity matrices from ROI time series"""
    
    def __init__(self, method: str = 'pearson', standardize: bool = True):
        self.method = method
        self.standardize = standardize

    def compute_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        n_timepoints, n_rois = timeseries.shape
        if self.standardize:
            scaler = StandardScaler()
            timeseries = scaler.fit_transform(timeseries)
        if self.method == 'pearson':
            conn = np.corrcoef(timeseries.T)
        elif self.method == 'partial_correlation':
            try:
                from sklearn.covariance import GraphicalLassoCV
                model = GraphicalLassoCV(cv=3, max_iter=100)
                model.fit(timeseries)
                conn = -model.precision_
            except:
                conn = np.corrcoef(timeseries.T)
        elif self.method == 'mutual_information':
            try:
                from sklearn.feature_selection import mutual_info_regression
                conn = np.zeros((n_rois, n_rois))
                for i in range(n_rois):
                    for j in range(i+1, n_rois):
                        val = mutual_info_regression(timeseries[:, [i]], timeseries[:, j])[0]
                        conn[i, j] = conn[j, i] = val
            except:
                conn = np.corrcoef(timeseries.T)
        else:
            raise ValueError(f"Unknown connectivity method: {self.method}")
        np.fill_diagonal(conn, 0.0)
        return np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureExtractor:
    """Wrapper to handle ROI timeseries -> connectivity -> save outputs"""

    def __init__(self, output_dir: Path, parcellation_labels: list, connectivity_method: str = 'pearson'):
        self.output_dir = output_dir
        self.labels = parcellation_labels
        self.conn_extractor = FunctionalConnectivityExtractor(connectivity_method)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_subject(self, unique_id: str, roi_timeseries: np.ndarray) -> dict:
        """Compute connectivity and save ROI timeseries + matrices"""
        outputs = {}
        # Save ROI timeseries CSV
        ts_path = self.output_dir / f"{unique_id}_roi_timeseries.csv"
        pd.DataFrame(roi_timeseries, columns=[f"ROI_{i+1}_{lbl}" for i, lbl in enumerate(self.labels)]).to_csv(ts_path, index=False)
        outputs['roi_timeseries'] = str(ts_path)
        # Compute connectivity
        conn = self.conn_extractor.compute_connectivity(roi_timeseries)
        # Save connectivity NPY
        npy_path = self.output_dir / f"{unique_id}_connectivity_matrix.npy"
        np.save(npy_path, conn)
        outputs['connectivity_npy'] = str(npy_path)
        # Save connectivity CSV
        csv_path = self.output_dir / f"{unique_id}_connectivity_matrix.csv"
        pd.DataFrame(conn, index=[f"ROI_{i+1}" for i in range(len(self.labels))],
                     columns=[f"ROI_{i+1}" for i in range(len(self.labels))]).to_csv(csv_path)
        outputs['connectivity_csv'] = str(csv_path)
        return outputs

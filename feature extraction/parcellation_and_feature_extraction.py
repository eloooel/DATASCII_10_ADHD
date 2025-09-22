"""
Parcellation Module for ADHD rs-fMRI Pipeline
Handles Schaefer-200 parcellation operations and ROI time series extraction
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Optional
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class SchaeferParcellation:
    """Handles Schaefer-200 parcellation operations"""
    
    def __init__(self, parcellation_path: Optional[str] = None):
        self.parcellation_path = parcellation_path
        self.atlas_data = None
        self.roi_labels = None
        self.n_rois = 200

    def load_parcellation(self) -> bool:
        """Load Schaefer-200 parcellation atlas"""
        try:
            if self.parcellation_path and Path(self.parcellation_path).exists():
                print(f"Loading Schaefer-200 parcellation from: {self.parcellation_path}")
                atlas_img = nib.load(self.parcellation_path)
                self.atlas_data = atlas_img.get_fdata().astype(int)
            else:
                print("Using simulated Schaefer-200 parcellation (placeholder)")
                self.atlas_data = self._create_placeholder_parcellation()

            self.roi_labels = self._generate_roi_labels()

            unique_rois = np.unique(self.atlas_data)
            unique_rois = unique_rois[unique_rois > 0]
            print(f"Loaded parcellation with {len(unique_rois)} ROIs, atlas shape: {self.atlas_data.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading parcellation: {e}")
            return False

    def _create_placeholder_parcellation(self) -> np.ndarray:
        """Create placeholder parcellation for testing"""
        shape = (91, 109, 91)
        parcellation = np.zeros(shape, dtype=int)
        x_div, y_div, z_div = 5, 8, 5
        roi_id = 1
        for i in range(x_div):
            for j in range(y_div):
                for k in range(z_div):
                    if roi_id <= 200:
                        x_start, x_end = int(i*shape[0]/x_div), int((i+1)*shape[0]/x_div)
                        y_start, y_end = int(j*shape[1]/y_div), int((j+1)*shape[1]/y_div)
                        z_start, z_end = int(k*shape[2]/z_div), int((k+1)*shape[2]/z_div)
                        parcellation[x_start:x_end, y_start:y_end, z_start:z_end] = roi_id
                        roi_id += 1
        return parcellation

    def _generate_roi_labels(self) -> List[str]:
        """Generate ROI labels for Schaefer-200 parcellation"""
        networks = {
            'Vis': 'Visual', 'SomMot': 'Somatomotor', 'DorsAttn': 'DorsalAttention',
            'SalVentAttn': 'SalVentAttn', 'Limbic': 'Limbic', 'Cont': 'Control', 'Default': 'Default'
        }
        network_sizes = {'Vis': 32, 'SomMot': 30, 'DorsAttn': 24, 'SalVentAttn': 28,
                         'Limbic': 14, 'Cont': 36, 'Default': 36}
        labels = []
        roi_idx = 1
        for net_abbrev, net_name in networks.items():
            size = network_sizes[net_abbrev]
            for i in range(size):
                hemisphere = 'LH' if i < size // 2 else 'RH'
                region_idx = (i % (size // 2)) + 1
                labels.append(f"{net_name}_{hemisphere}_{region_idx}")
                roi_idx += 1
                if roi_idx > 200:
                    break
            if roi_idx > 200:
                break
        while len(labels) < 200:
            labels.append(f"ROI_{len(labels)+1}")
        return labels[:200]

    def extract_roi_timeseries(self, fmri_data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract mean time series for each ROI"""
        if self.atlas_data is None:
            raise ValueError("Parcellation not loaded. Call load_parcellation() first.")

        if fmri_data.shape[:3] != self.atlas_data.shape:
            print("DEBUG: Spatial mismatch, resampling atlas...")
            zoom_factors = [fmri_data.shape[i] / self.atlas_data.shape[i] for i in range(3)]
            self.atlas_data = zoom(self.atlas_data.astype(float), zoom_factors, order=0).astype(int)
            print(f"DEBUG: Resampled atlas shape: {self.atlas_data.shape}")

        n_timepoints = fmri_data.shape[-1]
        roi_timeseries = np.zeros((n_timepoints, self.n_rois))

        mask_3d = brain_mask > 0 if brain_mask is not None else self.atlas_data > 0

        for roi_id in range(1, self.n_rois + 1):
            roi_mask = (self.atlas_data == roi_id) & mask_3d
            n_voxels = np.sum(roi_mask)
            roi_timeseries[:, roi_id - 1] = np.mean(fmri_data[roi_mask], axis=0) if n_voxels > 0 else np.zeros(n_timepoints)
        return roi_timeseries

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

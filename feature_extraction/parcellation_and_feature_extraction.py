"""
Feature Extraction Module for ADHD rs-fMRI Pipeline
Includes Schaefer-200 parcellation, ROI timeseries, and functional connectivity.
Supports subject-level batch parallel processing.
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import warnings

from utils.parallel_runner import run_parallel

warnings.filterwarnings('ignore', category=RuntimeWarning)


class SchaeferParcellation:
    """Handles Schaefer-200 parcellation operations"""
    
    def __init__(self, parcellation_path: Path = None):
        self.parcellation_path = parcellation_path
        self.atlas_data = None
        self.roi_labels = None
        self.n_rois = 200

    def load_parcellation(self) -> bool:
        try:
            if self.parcellation_path and self.parcellation_path.exists():
                atlas_img = nib.load(str(self.parcellation_path))
                self.atlas_data = atlas_img.get_fdata().astype(int)
            else:
                self.atlas_data = self._create_placeholder_parcellation()
            self.roi_labels = self._generate_roi_labels()
            return True
        except Exception as e:
            print(f"Error loading parcellation: {e}")
            return False

    def _create_placeholder_parcellation(self) -> np.ndarray:
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

    def _generate_roi_labels(self):
        labels = [f"ROI_{i+1}" for i in range(200)]
        return labels

    def extract_roi_timeseries(self, fmri_data: np.ndarray, brain_mask: np.ndarray = None) -> np.ndarray:
        if self.atlas_data.shape != fmri_data.shape[:3]:
            zoom_factors = [fmri_data.shape[i]/self.atlas_data.shape[i] for i in range(3)]
            self.atlas_data = zoom(self.atlas_data.astype(float), zoom_factors, order=0).astype(int)
        n_timepoints = fmri_data.shape[-1]
        roi_timeseries = np.zeros((n_timepoints, self.n_rois))
        mask_3d = brain_mask > 0 if brain_mask is not None else self.atlas_data > 0
        for roi_id in range(1, self.n_rois + 1):
            roi_mask = (self.atlas_data == roi_id) & mask_3d
            n_voxels = np.sum(roi_mask)
            roi_timeseries[:, roi_id-1] = np.mean(fmri_data[roi_mask], axis=0) if n_voxels > 0 else np.zeros(n_timepoints)
        return roi_timeseries


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
        else:
            conn = np.corrcoef(timeseries.T)
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
        outputs = {}
        ts_path = self.output_dir / f"{unique_id}_roi_timeseries.csv"
        pd.DataFrame(roi_timeseries, columns=[f"ROI_{i+1}_{lbl}" for i, lbl in enumerate(self.labels)]).to_csv(ts_path, index=False)
        outputs['roi_timeseries'] = str(ts_path)
        conn = self.conn_extractor.compute_connectivity(roi_timeseries)
        npy_path = self.output_dir / f"{unique_id}_connectivity_matrix.npy"
        np.save(npy_path, conn)
        outputs['connectivity_npy'] = str(npy_path)
        csv_path = self.output_dir / f"{unique_id}_connectivity_matrix.csv"
        pd.DataFrame(conn, index=[f"ROI_{i+1}" for i in range(len(self.labels))],
                     columns=[f"ROI_{i+1}" for i in range(len(self.labels))]).to_csv(csv_path)
        outputs['connectivity_csv'] = str(csv_path)
        return outputs


# ----------------- Batch Processing -----------------

def extract_features_worker(row, preproc_dir: Path, feature_out_dir: Path, atlas_labels: list):
    import numpy as np
    subject_id = row["subject_id"]
    subj_dir = preproc_dir / subject_id
    func_path = subj_dir / "func_preproc.npy"
    mask_path = subj_dir / "mask.npy"
    if not func_path.exists() or not mask_path.exists():
        return {"subject_id": subject_id, "status": "failed", "error": "Missing preprocessed files"}

    func_data = np.load(func_path)
    mask = np.load(mask_path)
    parcellation = SchaeferParcellation()
    parcellation.load_parcellation()
    roi_timeseries = parcellation.extract_roi_timeseries(func_data, mask)

    extractor = FeatureExtractor(feature_out_dir, parcellation_labels=atlas_labels)
    outputs = extractor.process_subject(subject_id, roi_timeseries)
    return {"subject_id": subject_id, "status": "success", "outputs": outputs}


def run_feature_extraction_parallel(metadata_csv: Path, preproc_dir: Path, feature_out_dir: Path,
                                    atlas_labels: list, max_workers: int = None):
    import pandas as pd
    metadata = pd.read_csv(metadata_csv)
    feature_out_dir.mkdir(parents=True, exist_ok=True)

    results = run_parallel(
        tasks=[row for _, row in metadata.iterrows()],
        worker_fn=lambda row: extract_features_worker(row, preproc_dir, feature_out_dir, atlas_labels),
        max_workers=max_workers,
        desc="Extracting features"
    )

    success = sum(1 for r in results if r["status"] == "success")
    print(f"Feature extraction complete. Success: {success}/{len(metadata)}")
    return results


def run_feature_extraction_stage(metadata_csv: Path, preproc_dir: Path, feature_out_dir: Path,
                                 atlas_labels: list, parallel: bool = True, max_workers: int = None):
    if parallel:
        run_feature_extraction_parallel(metadata_csv, preproc_dir, feature_out_dir, atlas_labels, max_workers)
    else:
        import pandas as pd
        metadata = pd.read_csv(metadata_csv)
        for _, row in metadata.iterrows():
            extract_features_worker(row, preproc_dir, feature_out_dir, atlas_labels)

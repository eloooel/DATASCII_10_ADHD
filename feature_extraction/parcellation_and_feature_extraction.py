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

from utils import run_parallel

warnings.filterwarnings('ignore', category=RuntimeWarning)


class SchaeferParcellation:
    """Handles Schaefer-200 parcellation operations"""
    
    def __init__(self, parcellation_path: Path = None):
        self.parcellation_path = parcellation_path
        self.atlas_data = None
        self.roi_labels = None
        self.n_rois = 200

    def load_parcellation(self) -> bool:
        """Load Schaefer parcellation - fails if not available"""
        try:
            if self.parcellation_path and self.parcellation_path.exists():
                atlas_img = nib.load(str(self.parcellation_path))
                self.atlas_data = atlas_img.get_fdata().astype(int)
                self.roi_labels = self._generate_roi_labels()
                return True
            else:
                # No placeholder - fail if real parcellation not available
                print("Error: No valid parcellation path provided")
                return False
        except Exception as e:
            print(f"Error loading parcellation: {e}")
            return False

    def _generate_roi_labels(self):
        """Generate ROI labels for Schaefer parcellation"""
        labels = [f"ROI_{i+1}" for i in range(200)]
        return labels

    def extract_roi_timeseries(self, fmri_data: np.ndarray, brain_mask: np.ndarray = None) -> np.ndarray:
        """Extract ROI timeseries - requires valid atlas data"""
        if self.atlas_data is None:
            raise ValueError("Atlas data not loaded. Cannot extract ROI timeseries.")
            
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

        # ROI timeseries file
        ts_path = self.output_dir / f"{unique_id}_roi_timeseries.csv"
        if not ts_path.exists():
            pd.DataFrame(
                roi_timeseries,
                columns=[f"ROI_{i+1}_{lbl}" for i, lbl in enumerate(self.labels)]
            ).to_csv(ts_path, index=False)
        outputs['roi_timeseries'] = str(ts_path)

        # Connectivity matrix
        npy_path = self.output_dir / f"{unique_id}_connectivity_matrix.npy"
        csv_path = self.output_dir / f"{unique_id}_connectivity_matrix.csv"
        if not (npy_path.exists() and csv_path.exists()):
            conn = self.conn_extractor.compute_connectivity(roi_timeseries)
            np.save(npy_path, conn)
            pd.DataFrame(
                conn,
                index=[f"ROI_{i+1}" for i in range(len(self.labels))],
                columns=[f"ROI_{i+1}" for i in range(len(self.labels))]
            ).to_csv(csv_path)

        outputs['connectivity_npy'] = str(npy_path)
        outputs['connectivity_csv'] = str(csv_path)

        return outputs



# ----------------- Batch Processing -----------------

def extract_features_worker(row, preproc_dir: Path, feature_out_dir: Path, atlas_labels: list, parcellation_path: Path = None):
    """Worker function for feature extraction - fails if parcellation unavailable"""
    subject_id = None
    site = None
    
    try:
        subject_id = row["subject_id"]
        # Consistent site extraction logic
        site = row.get("site", row.get("dataset", Path(row["input_path"]).parts[-5] if len(Path(row["input_path"]).parts) >= 5 else "UnknownSite"))
        
        # Update paths for NIfTI files with correct site handling
        func_path = preproc_dir / site / subject_id / "func_preproc.nii.gz"
        mask_path = preproc_dir / site / subject_id / "mask.nii.gz"

        # Create site-specific feature output directory
        site_feature_dir = feature_out_dir / site
        site_feature_dir.mkdir(parents=True, exist_ok=True)

        # Check preprocessing existence
        if not func_path.exists():
            return {
                "status": "failed", 
                "subject_id": subject_id, 
                "site": site,
                "error": f"Missing preprocessed file: {func_path}",
                "error_type": "missing_preprocessing"
            }

        if not mask_path.exists():
            return {
                "status": "failed", 
                "subject_id": subject_id, 
                "site": site,
                "error": f"Missing mask file: {mask_path}",
                "error_type": "missing_preprocessing"
            }

        # Load NIfTI data
        func_img = nib.load(func_path)
        mask_img = nib.load(mask_path)
        
        func_data = func_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Initialize parcellation with path - ✅ FIXED
        parcellation = SchaeferParcellation(parcellation_path)  # ✅ Pass the path!
        if not parcellation.load_parcellation():
            return {
                "status": "failed", 
                "subject_id": subject_id, 
                "site": site,
                "error": f"Failed to load Schaefer parcellation from {parcellation_path}",
                "error_type": "parcellation_unavailable"
            }
            
        # Extract ROI timeseries
        roi_timeseries = parcellation.extract_roi_timeseries(func_data, mask_data)

        # Save features
        extractor = FeatureExtractor(site_feature_dir, atlas_labels)
        outputs = extractor.process_subject(subject_id, roi_timeseries)

        return {
            "subject_id": subject_id, 
            "site": site, 
            "status": "success", 
            "outputs": outputs
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "subject_id": subject_id if 'subject_id' in locals() else "unknown",
            "site": site if 'site' in locals() else "unknown",
            "error": str(e),
            "error_type": "processing_error"
        }


def run_feature_extraction_stage(metadata_csv: Path, preproc_dir: Path, feature_out_dir: Path, atlas_labels: list, parallel: bool = True, max_workers: int = None):
    """
    Backend feature extraction function - no UI handling
    Used by main.py which handles all progress display
    """
    import pandas as pd
    metadata = pd.read_csv(metadata_csv)
    feature_out_dir.mkdir(parents=True, exist_ok=True)

    # Simple parallel/sequential execution without progress bars
    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                extract_features_worker, row, preproc_dir, feature_out_dir, atlas_labels
            ): row["subject_id"] for _, row in metadata.iterrows()}

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    subject_id = futures[future]
                    results.append({"subject_id": subject_id, "status": "failed", "error": str(e)})
    else:
        # Sequential execution
        results = []
        for _, row in metadata.iterrows():
            result = extract_features_worker(row, preproc_dir, feature_out_dir, atlas_labels)
            results.append(result)

    return results

def create_feature_manifest(feature_out_dir: Path, metadata: pd.DataFrame) -> Path:
    """Create a manifest CSV with paths to all extracted features for training"""
    manifest_data = []
    missing_count = 0
    
    for _, row in metadata.iterrows():
        subject_id = row['subject_id']
        site = row.get('site', row.get('dataset', 'unknown'))
        
        # Construct paths to feature files
        site_dir = feature_out_dir / site
        fc_path = site_dir / f"{subject_id}_connectivity_matrix.npy"
        ts_path = site_dir / f"{subject_id}_roi_timeseries.csv"
        
        # Only include subjects where both features exist
        if fc_path.exists() and ts_path.exists():
            manifest_data.append({
                'subject_id': subject_id,
                'site': site,
                'fc_path': str(fc_path),
                'ts_path': str(ts_path),
                'diagnosis': row.get('diagnosis', row.get('DX', 0))
            })
        else:
            missing_count += 1
    
    if not manifest_data:
        raise ValueError("No valid feature files found! Check feature extraction results.")
    
    # Create DataFrame and save
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = feature_out_dir / 'feature_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\nFeature Manifest Summary:")
    print(f"  ✅ Complete features: {len(manifest_df)} subjects")
    print(f"  ❌ Missing features: {missing_count} subjects")  # Less verbose
    print(f"  Sites: {manifest_df['site'].nunique()}")
    if 'diagnosis' in manifest_df.columns:
        print(f"  Controls: {(manifest_df['diagnosis'] == 0).sum()}")
        print(f"  ADHD: {(manifest_df['diagnosis'] == 1).sum()}")
    print(f"  Saved to: {manifest_path}")
    
    return manifest_path


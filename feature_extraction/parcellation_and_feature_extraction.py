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
    """Worker function for feature extraction"""
    subject_id = None
    site = None
    
    try:
        subject_id = row["subject_id"]
        site = row.get("site", row.get("dataset", Path(row["input_path"]).parts[-5] if len(Path(row["input_path"]).parts) >= 5 else "UnknownSite"))
        
        print(f"🔄 Processing {site}/{subject_id}")
        
        # Update paths for NIfTI files
        func_path = preproc_dir / site / subject_id / "func_preproc.nii.gz"
        mask_path = preproc_dir / site / subject_id / "mask.nii.gz"

        # Create site-specific feature output directory
        site_feature_dir = feature_out_dir / site
        site_feature_dir.mkdir(parents=True, exist_ok=True)

        # ✅ DETAILED FILE EXISTENCE CHECKS
        if not func_path.exists():
            print(f"❌ {subject_id}: Missing functional file")
            print(f"   Expected: {func_path}")
            print(f"   Parent dir exists: {func_path.parent.exists()}")
            if func_path.parent.exists():
                available_files = list(func_path.parent.glob("*.nii*"))
                print(f"   Available files: {[f.name for f in available_files]}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Missing preprocessed functional file",
                "error_type": "missing_preprocessing",
                "error_details": {
                    "missing_file": str(func_path),
                    "parent_exists": func_path.parent.exists(),
                    "available_files": [f.name for f in func_path.parent.glob("*.nii*")] if func_path.parent.exists() else []
                }
            }

        if not mask_path.exists():
            print(f"❌ {subject_id}: Missing mask file")
            print(f"   Expected: {mask_path}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Missing mask file",
                "error_type": "missing_preprocessing",
                "error_details": {
                    "missing_file": str(mask_path),
                    "func_exists": func_path.exists()
                }
            }

        # ✅ DETAILED NIFTI LOADING
        print(f"📁 Loading NIfTI files for {subject_id}")
        try:
            func_img = nib.load(func_path)
            func_data = func_img.get_fdata()
            print(f"   Functional data shape: {func_data.shape}")
            
            if func_data.ndim != 4:
                raise ValueError(f"Functional data should be 4D, got {func_data.ndim}D")
            
            if func_data.shape[-1] < 50:  # Check minimum timepoints
                raise ValueError(f"Too few timepoints: {func_data.shape[-1]} (minimum 50 expected)")
                
        except Exception as e:
            print(f"❌ {subject_id}: Error loading functional file")
            print(f"   Error: {str(e)}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Failed to load functional NIfTI: {str(e)}",
                "error_type": "nifti_loading_error",
                "error_details": {
                    "file_path": str(func_path),
                    "file_size_mb": func_path.stat().st_size / (1024*1024) if func_path.exists() else 0,
                    "detailed_error": str(e)
                }
            }

        try:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            print(f"   Mask data shape: {mask_data.shape}")
            
            if mask_data.shape != func_data.shape[:3]:
                raise ValueError(f"Mask shape {mask_data.shape} doesn't match functional shape {func_data.shape[:3]}")
                
        except Exception as e:
            print(f"❌ {subject_id}: Error loading mask file")
            print(f"   Error: {str(e)}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Failed to load mask NIfTI: {str(e)}",
                "error_type": "nifti_loading_error",
                "error_details": {
                    "file_path": str(mask_path),
                    "file_size_mb": mask_path.stat().st_size / (1024*1024) if mask_path.exists() else 0,
                    "detailed_error": str(e)
                }
            }

        # ✅ DETAILED PARCELLATION LOADING
        print(f"🧠 Loading parcellation for {subject_id}")
        try:
            parcellation = SchaeferParcellation(parcellation_path)
            if not parcellation.load_parcellation():
                print(f"❌ {subject_id}: Parcellation loading failed")
                print(f"   Atlas path: {parcellation_path}")
                print(f"   Atlas exists: {parcellation_path.exists() if parcellation_path else False}")
                
                return {
                    "status": "failed", 
                    "subject_id": subject_id, 
                    "site": site,
                    "error": f"Failed to load Schaefer parcellation",
                    "error_type": "parcellation_unavailable",
                    "error_details": {
                        "parcellation_path": str(parcellation_path) if parcellation_path else "None",
                        "path_exists": parcellation_path.exists() if parcellation_path else False
                    }
                }
        except Exception as e:
            print(f"❌ {subject_id}: Parcellation error")
            print(f"   Error: {str(e)}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Parcellation error: {str(e)}",
                "error_type": "parcellation_error",
                "error_details": {
                    "detailed_error": str(e),
                    "parcellation_path": str(parcellation_path) if parcellation_path else "None"
                }
            }
            
        # ✅ DETAILED ROI EXTRACTION
        print(f"🎯 Extracting ROI timeseries for {subject_id}")
        try:
            roi_timeseries = parcellation.extract_roi_timeseries(func_data, mask_data)
            print(f"   ROI timeseries shape: {roi_timeseries.shape}")
            
            if roi_timeseries.shape[1] != 200:
                raise ValueError(f"Expected 200 ROIs, got {roi_timeseries.shape[1]}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(roi_timeseries)):
                print(f"⚠️  {subject_id}: NaN values detected in ROI timeseries")
            
            if np.any(np.isinf(roi_timeseries)):
                print(f"⚠️  {subject_id}: Infinite values detected in ROI timeseries")
                
        except Exception as e:
            print(f"❌ {subject_id}: ROI extraction failed")
            print(f"   Error: {str(e)}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"ROI extraction failed: {str(e)}",
                "error_type": "roi_extraction_error",
                "error_details": {
                    "func_shape": func_data.shape,
                    "mask_shape": mask_data.shape,
                    "detailed_error": str(e)
                }
            }

        # ✅ DETAILED FEATURE SAVING
        print(f"💾 Saving features for {subject_id}")
        try:
            extractor = FeatureExtractor(site_feature_dir, atlas_labels)
            outputs = extractor.process_subject(subject_id, roi_timeseries)
            
            # Verify outputs were created
            for output_type, output_path in outputs.items():
                if not Path(output_path).exists():
                    raise FileNotFoundError(f"Failed to create {output_type}: {output_path}")
            
            print(f"✅ {subject_id}: Success")
            
        except Exception as e:
            print(f"❌ {subject_id}: Feature saving failed")
            print(f"   Error: {str(e)}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Feature saving failed: {str(e)}",
                "error_type": "feature_saving_error",
                "error_details": {
                    "output_dir": str(site_feature_dir),
                    "detailed_error": str(e)
                }
            }

        return {
            "subject_id": subject_id, 
            "site": site, 
            "status": "success", 
            "outputs": outputs,
            "processing_info": {
                "func_shape": func_data.shape,
                "mask_shape": mask_data.shape,
                "roi_timeseries_shape": roi_timeseries.shape,
                "has_nan": bool(np.any(np.isnan(roi_timeseries))),
                "has_inf": bool(np.any(np.isinf(roi_timeseries)))
            }
        }
        
    except Exception as e:
        print(f"❌ {subject_id or 'Unknown'}: Unexpected error")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        
        return {
            "status": "failed",
            "subject_id": subject_id if 'subject_id' in locals() else "unknown",
            "site": site if 'site' in locals() else "unknown",
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unexpected_error",
            "error_details": {
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "detailed_error": str(e)
            }
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


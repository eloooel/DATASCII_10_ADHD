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
    """Worker function for feature extraction - with comprehensive file corruption handling"""
    import gzip
    import os
    
    subject_id = None
    site = None
    
    try:
        subject_id = row["subject_id"]
        site = row.get("site", row.get("dataset", "UnknownSite"))
        
        print(f"🔄 Processing {site}/{subject_id}")
        
        func_path = preproc_dir / site / subject_id / "func_preproc.nii.gz"
        mask_path = preproc_dir / site / subject_id / "mask.nii.gz"

        site_feature_dir = feature_out_dir / site
        site_feature_dir.mkdir(parents=True, exist_ok=True)

        # ✅ COMPREHENSIVE FILE CORRUPTION DETECTION
        def validate_gzipped_nifti(file_path: Path) -> dict:
            """Comprehensive validation of gzipped NIfTI files"""
            validation = {
                'exists': file_path.exists(),
                'readable': False,
                'size_mb': 0,
                'gzip_valid': False,
                'nifti_loadable': False,
                'corruption_type': None,
                'error_message': None
            }
            
            if not validation['exists']:
                validation['error_message'] = f"File does not exist: {file_path}"
                return validation
            
            try:
                # Check file size
                file_size = file_path.stat().st_size
                validation['size_mb'] = file_size / (1024 * 1024)
                
                if file_size == 0:
                    validation['error_message'] = "File is empty (0 bytes)"
                    validation['corruption_type'] = 'empty_file'
                    return validation
                
                # Test gzip integrity first
                print(f"   🔍 Testing gzip integrity for {file_path.name}")
                try:
                    with gzip.open(file_path, 'rb') as gz_file:
                        # Try to read the entire file in chunks to detect corruption
                        chunk_size = 1024 * 1024  # 1MB chunks
                        total_read = 0
                        
                        while True:
                            chunk = gz_file.read(chunk_size)
                            if not chunk:
                                break
                            total_read += len(chunk)
                        
                        print(f"   ✅ Gzip decompression successful: {total_read / (1024*1024):.1f}MB decompressed")
                        validation['gzip_valid'] = True
                        
                except Exception as gz_error:
                    error_msg = str(gz_error)
                    print(f"   ❌ Gzip corruption detected: {error_msg}")
                    
                    # Categorize gzip errors
                    if "invalid stored block lengths" in error_msg:
                        validation['corruption_type'] = 'gzip_block_corruption'
                    elif "invalid block type" in error_msg:
                        validation['corruption_type'] = 'gzip_header_corruption'
                    elif "unexpected end of file" in error_msg:
                        validation['corruption_type'] = 'truncated_file'
                    else:
                        validation['corruption_type'] = 'unknown_gzip_error'
                    
                    validation['error_message'] = f"Gzip corruption: {error_msg}"
                    return validation
                
                # Test NIfTI loading
                print(f"   🧠 Testing NIfTI loading for {file_path.name}")
                try:
                    import nibabel as nib
                    img = nib.load(file_path)
                    data_shape = img.shape
                    
                    # Basic shape validation
                    if len(data_shape) < 3:
                        validation['error_message'] = f"Invalid NIfTI dimensions: {data_shape}"
                        validation['corruption_type'] = 'invalid_dimensions'
                        return validation
                    
                    # Try to access a small portion of data to ensure it's readable
                    if len(data_shape) == 4:
                        test_data = img.get_fdata()[:10, :10, :10, :5]  # Small sample
                    else:
                        test_data = img.get_fdata()[:10, :10, :10]  # Small sample
                    
                    print(f"   ✅ NIfTI loading successful: shape {data_shape}")
                    validation['nifti_loadable'] = True
                    validation['readable'] = True
                    
                except Exception as nii_error:
                    error_msg = str(nii_error)
                    print(f"   ❌ NIfTI loading failed: {error_msg}")
                    
                    if "invalid stored block lengths" in error_msg:
                        validation['corruption_type'] = 'gzip_block_corruption'
                    else:
                        validation['corruption_type'] = 'nifti_format_error'
                    
                    validation['error_message'] = f"NIfTI loading failed: {error_msg}"
                    return validation
                
            except Exception as e:
                validation['error_message'] = f"File validation error: {str(e)}"
                validation['corruption_type'] = 'validation_error'
                return validation
            
            return validation

        # Validate functional file
        print(f"📁 Validating functional file: {func_path.name}")
        func_validation = validate_gzipped_nifti(func_path)
        
        if not func_validation['readable']:
            print(f"❌ {subject_id}: Functional file corrupted")
            print(f"   File: {func_path}")
            print(f"   Size: {func_validation['size_mb']:.2f}MB")
            print(f"   Corruption type: {func_validation['corruption_type']}")
            print(f"   Error: {func_validation['error_message']}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Functional file corrupted: {func_validation['error_message']}",
                "error_type": "file_corruption",
                "error_details": {
                    "file_path": str(func_path),
                    "corruption_type": func_validation['corruption_type'],
                    "file_size_mb": func_validation['size_mb'],
                    "validation_result": func_validation
                }
            }

        # Validate mask file
        print(f"📁 Validating mask file: {mask_path.name}")
        mask_validation = validate_gzipped_nifti(mask_path)
        
        if not mask_validation['readable']:
            print(f"❌ {subject_id}: Mask file corrupted")
            print(f"   File: {mask_path}")
            print(f"   Corruption type: {mask_validation['corruption_type']}")
            
            return {
                "status": "failed",
                "subject_id": subject_id,
                "site": site,
                "error": f"Mask file corrupted: {mask_validation['error_message']}",
                "error_type": "file_corruption",
                "error_details": {
                    "file_path": str(mask_path),
                    "corruption_type": mask_validation['corruption_type'],
                    "validation_result": mask_validation
                }
            }

        # If we get here, files are valid - proceed with normal loading
        print(f"📊 Loading validated NIfTI files for {subject_id}")
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata()
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        print(f"   Functional shape: {func_data.shape}")
        print(f"   Mask shape: {mask_data.shape}")

        # Continue with rest of processing (parcellation, ROI extraction, etc.)
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
        
        return {
            "status": "failed",
            "subject_id": subject_id if 'subject_id' in locals() else "unknown",
            "site": site if 'site' in locals() else "unknown",
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unexpected_error",
            "error_details": {
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


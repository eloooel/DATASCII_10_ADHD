"""
Debugged version of Parcellation and Feature Extraction Module for ADHD Classification using GNN-STAN
Removes nested progress bars and adds debugging output to identify the hanging issue

Input: Preprocessed ADHD-200 dataset (from the preprocessing module)
Output: ROI time-series (.csv), static functional connectivity matrices (.npy)
Process: Uses Schaefer-200 parcellation to extract mean BOLD signals from 200 ROIs,
         then computes functional connectivity matrices for GNN input
"""

import argparse
import json
import csv
import sys
import pandas as pd
import numpy as np
import nibabel as nib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_OUTPUT_SUBDIR = "feature_extraction_outputs"
MANIFEST_FILENAME = "feature_extraction_manifest.csv"
SCHAEFER_200_URL = "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"


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
                # Create placeholder parcellation for testing
                self.atlas_data = self._create_placeholder_parcellation()
            
            # Generate ROI labels
            self.roi_labels = self._generate_roi_labels()
            
            unique_rois = np.unique(self.atlas_data)
            unique_rois = unique_rois[unique_rois > 0]  # Remove background
            
            print(f"Loaded parcellation with {len(unique_rois)} ROIs")
            print(f"Atlas shape: {self.atlas_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading parcellation: {e}")
            return False
    
    def _create_placeholder_parcellation(self) -> np.ndarray:
        """Create placeholder parcellation for testing"""
        # Create a simple 3D parcellation with 200 regions
        shape = (91, 109, 91)  # Standard MNI152 2mm shape
        parcellation = np.zeros(shape, dtype=int)
        
        # Divide brain space into regions (simplified)
        x_div = 5  # divisions along x
        y_div = 8  # divisions along y  
        z_div = 5  # divisions along z (5*8*5 = 200)
        
        roi_id = 1
        for i in range(x_div):
            for j in range(y_div):
                for k in range(z_div):
                    if roi_id <= 200:
                        x_start = int(i * shape[0] / x_div)
                        x_end = int((i + 1) * shape[0] / x_div)
                        y_start = int(j * shape[1] / y_div)
                        y_end = int((j + 1) * shape[1] / y_div)
                        z_start = int(k * shape[2] / z_div)
                        z_end = int((k + 1) * shape[2] / z_div)
                        
                        parcellation[x_start:x_end, y_start:y_end, z_start:z_end] = roi_id
                        roi_id += 1
        
        return parcellation
    
    def _generate_roi_labels(self) -> List[str]:
        """Generate ROI labels for Schaefer-200 parcellation"""
        # Schaefer-200 uses 7 networks: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
        networks = {
            'Vis': 'Visual',
            'SomMot': 'Somatomotor', 
            'DorsAttn': 'DorsalAttention',
            'SalVentAttn': 'SalVentAttn',
            'Limbic': 'Limbic',
            'Cont': 'Control',
            'Default': 'Default'
        }
        
        # Approximate distribution in Schaefer-200
        network_sizes = {
            'Vis': 32, 'SomMot': 30, 'DorsAttn': 24, 'SalVentAttn': 28,
            'Limbic': 14, 'Cont': 36, 'Default': 36
        }
        
        labels = []
        roi_idx = 1
        
        for net_abbrev, net_name in networks.items():
            size = network_sizes[net_abbrev]
            for i in range(size):
                hemisphere = 'LH' if i < size // 2 else 'RH'
                region_idx = (i % (size // 2)) + 1
                label = f"{net_name}_{hemisphere}_{region_idx}"
                labels.append(label)
                roi_idx += 1
                if roi_idx > 200:
                    break
            if roi_idx > 200:
                break
        
        # Fill remaining if needed
        while len(labels) < 200:
            labels.append(f"ROI_{len(labels) + 1}")
        
        return labels[:200]
    
    def extract_roi_timeseries(self, fmri_data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract mean time series for each ROI
        
        Args:
            fmri_data: 4D fMRI data (x, y, z, time)
            brain_mask: Optional brain mask to restrict extraction
            
        Returns:
            2D array (time, ROIs) with mean time series for each ROI
        """
        if self.atlas_data is None:
            raise ValueError("Parcellation not loaded. Call load_parcellation() first.")
        
        print(f"DEBUG: fMRI shape: {fmri_data.shape}")
        print(f"DEBUG: Atlas shape: {self.atlas_data.shape}")
        
        # Handle spatial dimension mismatch
        if fmri_data.shape[:3] != self.atlas_data.shape:
            print(f"DEBUG: Spatial mismatch, resampling atlas...")
            from scipy.ndimage import zoom
            zoom_factors = [fmri_data.shape[i] / self.atlas_data.shape[i] for i in range(3)]
            print(f"DEBUG: Zoom factors: {zoom_factors}")
            
            resampled_atlas = zoom(self.atlas_data.astype(float), zoom_factors, order=0)
            resampled_atlas = resampled_atlas.astype(int)
            print(f"DEBUG: Resampled atlas shape: {resampled_atlas.shape}")
            
            # Verify ROI preservation
            unique_orig = len(np.unique(self.atlas_data)) - 1  # -1 for background
            unique_resamp = len(np.unique(resampled_atlas)) - 1
            print(f"DEBUG: Original ROIs: {unique_orig}, Resampled ROIs: {unique_resamp}")
            
            self.atlas_data = resampled_atlas
        
        n_timepoints = fmri_data.shape[-1]
        roi_timeseries = np.zeros((n_timepoints, self.n_rois))
        
        print(f"DEBUG: Extracting time series for {n_timepoints} timepoints, {self.n_rois} ROIs")
        
        # Apply brain mask if provided
        if brain_mask is not None:
            print(f"DEBUG: Using brain mask with {np.sum(brain_mask > 0)} active voxels")
            mask_3d = brain_mask > 0
        else:
            print("DEBUG: No brain mask provided, using atlas mask")
            mask_3d = self.atlas_data > 0
        
        # Extract time series for each ROI (with progress tracking)
        rois_processed = 0
        for roi_id in range(1, self.n_rois + 1):
            roi_mask = (self.atlas_data == roi_id) & mask_3d
            n_voxels = np.sum(roi_mask)
            
            if n_voxels > 0:
                # Extract mean time series for this ROI
                roi_voxels = fmri_data[roi_mask]  # (n_voxels, time)
                mean_timeseries = np.mean(roi_voxels, axis=0)
                roi_timeseries[:, roi_id - 1] = mean_timeseries
            else:
                # Fill with zeros if no voxels in ROI
                roi_timeseries[:, roi_id - 1] = np.zeros(n_timepoints)
            
            rois_processed += 1
            
            # Print progress every 50 ROIs
            if rois_processed % 50 == 0:
                print(f"DEBUG: Processed {rois_processed}/{self.n_rois} ROIs")
        
        print(f"DEBUG: Completed ROI extraction for all {self.n_rois} ROIs")
        return roi_timeseries


class FunctionalConnectivityExtractor:
    """Computes functional connectivity matrices from ROI time series"""
    
    def __init__(self, method: str = 'pearson', standardize: bool = True):
        self.method = method
        self.standardize = standardize
        
    def compute_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute functional connectivity matrix
        
        Args:
            timeseries: 2D array (time, ROIs)
            
        Returns:
            2D connectivity matrix (ROIs, ROIs)
        """
        n_timepoints, n_rois = timeseries.shape
        print(f"DEBUG: Computing connectivity for {n_timepoints} timepoints, {n_rois} ROIs")
        
        # Standardize time series if requested
        if self.standardize:
            print("DEBUG: Standardizing time series...")
            scaler = StandardScaler()
            timeseries_norm = scaler.fit_transform(timeseries)
        else:
            timeseries_norm = timeseries
        
        # Compute connectivity based on method
        print(f"DEBUG: Computing {self.method} connectivity...")
        if self.method == 'pearson':
            connectivity_matrix = np.corrcoef(timeseries_norm.T)
        elif self.method == 'partial_correlation':
            connectivity_matrix = self._partial_correlation(timeseries_norm)
        elif self.method == 'mutual_information':
            connectivity_matrix = self._mutual_information(timeseries_norm)
        else:
            raise ValueError(f"Unknown connectivity method: {self.method}")
        
        print("DEBUG: Handling NaN values...")
        # Handle NaN values
        connectivity_matrix = np.nan_to_num(connectivity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Set diagonal to zero (remove self-connections)
        np.fill_diagonal(connectivity_matrix, 0.0)
        
        print(f"DEBUG: Final connectivity matrix shape: {connectivity_matrix.shape}")
        return connectivity_matrix
    
    def _partial_correlation(self, timeseries: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix"""
        try:
            from sklearn.covariance import GraphicalLassoCV
            # Use GraphicalLasso for partial correlation estimation
            model = GraphicalLassoCV(cv=3, max_iter=100)
            model.fit(timeseries)
            return -model.precision_  # Convert precision to partial correlation
        except:
            # Fallback to regular correlation if GraphicalLasso fails
            print("DEBUG: GraphicalLasso failed, falling back to Pearson correlation")
            return np.corrcoef(timeseries.T)
    
    def _mutual_information(self, timeseries: np.ndarray) -> np.ndarray:
        """Compute mutual information matrix"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            n_rois = timeseries.shape[1]
            mi_matrix = np.zeros((n_rois, n_rois))
            
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    mi_val = mutual_info_regression(
                        timeseries[:, [i]], timeseries[:, j], 
                        discrete_features=False, random_state=42
                    )[0]
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val
            
            return mi_matrix
        except:
            # Fallback to correlation
            print("DEBUG: Mutual information failed, falling back to Pearson correlation")
            return np.corrcoef(timeseries.T)


class ParcellationFeatureExtractor:
    """Main class for parcellation-based feature extraction"""
    
    def __init__(self, output_dir: Path, parcellation_path: Optional[str] = None,
                 connectivity_method: str = 'pearson'):
        self.output_dir = output_dir
        self.parcellation = SchaeferParcellation(parcellation_path)
        self.connectivity_extractor = FunctionalConnectivityExtractor(method=connectivity_method)
        self.results = []
        
        # Load parcellation
        print("DEBUG: Loading parcellation...")
        if not self.parcellation.load_parcellation():
            raise RuntimeError("Failed to load parcellation atlas")
        print("DEBUG: Parcellation loaded successfully")
    
    def process_subject(self, subject_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single subject's preprocessed data
        
        Expected inputs from preprocessing module:
        - preprocessed_bold: 4D fMRI volume
        - brain_mask: 3D brain mask
        - confounds: confound regressors
        
        Outputs:
        - ROI time-series (.csv)
        - Static functional connectivity matrix (.npy)
        """
        unique_id = subject_info['unique_id']
        preprocessed_path = subject_info.get('preprocessed_bold')
        mask_path = subject_info.get('brain_mask')
        
        print(f"DEBUG: Processing subject {unique_id}")
        print(f"DEBUG: Preprocessed path: {preprocessed_path}")
        print(f"DEBUG: Mask path: {mask_path}")
        
        if not preprocessed_path or not Path(preprocessed_path).exists():
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
        
        # Create subject output directory
        dataset = subject_info['dataset']
        subject_id = subject_info['subject_id']
        session = subject_info.get('session', '')
        
        subject_out_dir = self.output_dir / dataset / subject_id
        if session:
            subject_out_dir = subject_out_dir / f"ses-{session}"
        subject_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"DEBUG: Output directory: {subject_out_dir}")
        
        try:
            start_time = time.time()
            
            # Step 1: Load preprocessed fMRI data
            print("DEBUG: Loading preprocessed fMRI data...")
            fmri_img = nib.load(preprocessed_path)
            fmri_data = fmri_img.get_fdata()
            print(f"DEBUG: fMRI data loaded, shape: {fmri_data.shape}")
            
            # Step 2: Load brain mask if available
            print("DEBUG: Loading brain mask...")
            brain_mask = None
            if mask_path and Path(mask_path).exists():
                mask_img = nib.load(mask_path)
                brain_mask = mask_img.get_fdata()
                print(f"DEBUG: Brain mask loaded, shape: {brain_mask.shape}")
            else:
                print("DEBUG: No brain mask found")
            
            # Step 3: Extract ROI time series
            print("DEBUG: Starting ROI time series extraction...")
            roi_timeseries = self.parcellation.extract_roi_timeseries(fmri_data, brain_mask)
            print(f"DEBUG: ROI time series extraction completed, shape: {roi_timeseries.shape}")
            
            # Step 4: Compute functional connectivity
            print("DEBUG: Starting functional connectivity computation...")
            connectivity_matrix = self.connectivity_extractor.compute_connectivity(roi_timeseries)
            print(f"DEBUG: Connectivity computation completed, shape: {connectivity_matrix.shape}")
            
            # Step 5: Save outputs
            print("DEBUG: Saving outputs...")
            outputs = self._save_outputs(
                subject_out_dir, unique_id, roi_timeseries, connectivity_matrix
            )
            print(f"DEBUG: Outputs saved: {list(outputs.keys())}")
            
            processing_time = time.time() - start_time
            
            # Quality metrics
            print("DEBUG: Computing quality metrics...")
            quality_metrics = self._compute_quality_metrics(roi_timeseries, connectivity_matrix)
            print(f"DEBUG: Quality metrics: {quality_metrics}")
            
            success_result = {
                'subject_info': subject_info,
                'unique_id': unique_id,
                'status': 'success',
                'processing_time': processing_time,
                'outputs': outputs,
                'quality_metrics': quality_metrics,
                'error': None
            }
            
            print(f"DEBUG: Successfully processed {unique_id} in {processing_time:.1f}s")
            return success_result
            
        except Exception as e:
            print(f"DEBUG: Error processing {unique_id}: {str(e)}")
            import traceback
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            
            error_result = {
                'subject_info': subject_info,
                'unique_id': unique_id,
                'status': 'failed',
                'processing_time': 0,
                'outputs': {},
                'quality_metrics': {},
                'error': str(e)
            }
            
            return error_result
    
    def _save_outputs(self, output_dir: Path, unique_id: str, 
                     roi_timeseries: np.ndarray, connectivity_matrix: np.ndarray) -> Dict[str, str]:
        """Save feature extraction outputs"""
        outputs = {}
        
        try:
            print(f"DEBUG: Saving outputs to {output_dir}")
            
            # 1. Save ROI time-series as CSV
            print("DEBUG: Saving ROI time series CSV...")
            timeseries_path = output_dir / f"{unique_id}_roi_timeseries.csv"
            timeseries_df = pd.DataFrame(
                roi_timeseries, 
                columns=[f"ROI_{i+1}_{label}" for i, label in enumerate(self.parcellation.roi_labels)]
            )
            timeseries_df.to_csv(timeseries_path, index=False)
            outputs['roi_timeseries'] = str(timeseries_path)
            print(f"DEBUG: Saved time series to {timeseries_path}")
            
            # 2. Save connectivity matrix as numpy array
            print("DEBUG: Saving connectivity matrix NPY...")
            connectivity_path = output_dir / f"{unique_id}_connectivity_matrix.npy"
            np.save(connectivity_path, connectivity_matrix)
            outputs['connectivity_matrix'] = str(connectivity_path)
            print(f"DEBUG: Saved connectivity matrix to {connectivity_path}")
            
            # 3. Save connectivity matrix as CSV (for inspection)
            print("DEBUG: Saving connectivity matrix CSV...")
            connectivity_csv_path = output_dir / f"{unique_id}_connectivity_matrix.csv"
            connectivity_df = pd.DataFrame(
                connectivity_matrix,
                index=[f"ROI_{i+1}" for i in range(len(self.parcellation.roi_labels))],
                columns=[f"ROI_{i+1}" for i in range(len(self.parcellation.roi_labels))]
            )
            connectivity_df.to_csv(connectivity_csv_path)
            outputs['connectivity_csv'] = str(connectivity_csv_path)
            print(f"DEBUG: Saved connectivity CSV to {connectivity_csv_path}")
            
            # 4. Save ROI labels
            print("DEBUG: Saving ROI labels...")
            labels_path = output_dir / f"{unique_id}_roi_labels.csv"
            labels_df = pd.DataFrame({
                'ROI_ID': range(1, len(self.parcellation.roi_labels) + 1),
                'ROI_Label': self.parcellation.roi_labels
            })
            labels_df.to_csv(labels_path, index=False)
            outputs['roi_labels'] = str(labels_path)
            print(f"DEBUG: Saved ROI labels to {labels_path}")
            
        except Exception as e:
            print(f"DEBUG: Error saving outputs for {unique_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return outputs
    
    def _compute_quality_metrics(self, roi_timeseries: np.ndarray, 
                               connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Compute quality control metrics"""
        try:
            # ROI coverage (percentage of ROIs with non-zero signal)
            non_zero_rois = np.sum(np.std(roi_timeseries, axis=0) > 0)
            roi_coverage = (non_zero_rois / roi_timeseries.shape[1]) * 100
            
            # Mean connectivity strength
            # Only consider upper triangle (excluding diagonal)
            upper_triangle = np.triu(connectivity_matrix, k=1)
            mean_connectivity = np.mean(np.abs(upper_triangle[upper_triangle != 0]))
            
            # Signal-to-noise ratio approximation
            mean_signal = np.mean(np.std(roi_timeseries, axis=0))
            
            # Connectivity density (percentage of connections above threshold)
            threshold = 0.1
            strong_connections = np.sum(np.abs(upper_triangle) > threshold)
            total_connections = np.sum(upper_triangle != 0)
            connectivity_density = (strong_connections / total_connections * 100) if total_connections > 0 else 0
            
            return {
                'roi_coverage': roi_coverage,
                'mean_connectivity': mean_connectivity,
                'mean_signal_std': mean_signal,
                'connectivity_density': connectivity_density,
                'n_timepoints': roi_timeseries.shape[0],
                'n_rois': roi_timeseries.shape[1]
            }
            
        except Exception as e:
            print(f"DEBUG: Error computing quality metrics: {e}")
            return {
                'roi_coverage': 0.0,
                'mean_connectivity': 0.0,
                'mean_signal_std': 0.0,
                'connectivity_density': 0.0,
                'n_timepoints': roi_timeseries.shape[0] if roi_timeseries.size > 0 else 0,
                'n_rois': roi_timeseries.shape[1] if roi_timeseries.size > 0 else 0
            }
    
    def save_manifest(self, results: List[Dict[str, Any]], manifest_path: Path):
        """Save processing manifest"""
        print(f"DEBUG: Saving manifest with {len(results)} results to {manifest_path}")
        manifest_data = []
        
        for result in results:
            subject_info = result['subject_info']
            outputs = result.get('outputs', {})
            quality = result.get('quality_metrics', {})
            
            manifest_entry = {
                'dataset': subject_info['dataset'],
                'subject_id': subject_info['subject_id'],
                'session': subject_info.get('session', ''),
                'run': subject_info.get('run', ''),
                'unique_id': result['unique_id'],
                'preprocessed_input': subject_info.get('preprocessed_bold', ''),
                'roi_timeseries_csv': outputs.get('roi_timeseries', ''),
                'connectivity_matrix_npy': outputs.get('connectivity_matrix', ''),
                'connectivity_matrix_csv': outputs.get('connectivity_csv', ''),
                'roi_labels_csv': outputs.get('roi_labels', ''),
                'status': result['status'],
                'processing_time': f"{result.get('processing_time', 0):.1f}",
                'roi_coverage_percent': f"{quality.get('roi_coverage', 0):.1f}",
                'mean_connectivity': f"{quality.get('mean_connectivity', 0):.3f}",
                'connectivity_density_percent': f"{quality.get('connectivity_density', 0):.1f}",
                'n_timepoints': quality.get('n_timepoints', 0),
                'error': result.get('error', '')
            }
            manifest_data.append(manifest_entry)
        
        df = pd.DataFrame(manifest_data)
        df.to_csv(manifest_path, index=False)
        print(f"DEBUG: Saved feature extraction manifest to {manifest_path}")


def load_preprocessing_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load preprocessing manifest to get input files"""
    try:
        print(f"DEBUG: Loading preprocessing manifest from {manifest_path}")
        df = pd.read_csv(manifest_path)
        print(f"DEBUG: Manifest loaded with {len(df)} rows")
        
        # Convert to list of dictionaries and filter successful preprocessing
        subjects = []
        for _, row in df.iterrows():
            if row['status'] == 'success' and row['preprocessed_bold']:
                subject_info = {
                    'dataset': row['dataset'],
                    'subject_id': row['subject_id'],
                    'session': row['session'] if pd.notna(row['session']) else '',
                    'run': row['run'] if pd.notna(row['run']) else '',
                    'unique_id': row['unique_id'],
                    'preprocessed_bold': row['preprocessed_bold'],
                    'brain_mask': row.get('brain_mask', ''),
                    'confounds': row.get('confounds', '')
                }
                subjects.append(subject_info)
        
        print(f"DEBUG: Found {len(subjects)} successful subjects")
        return subjects
        
    except Exception as e:
        print(f"DEBUG: Error loading preprocessing manifest: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=str, required=True,
                       help="Path to preprocessing manifest CSV")
    parser.add_argument("--parcellation", type=str, required=True,
                       help="Path to Schaefer-200 parcellation atlas")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_extraction",
                       help="Output directory for extracted features")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of subjects to process (for testing)")
    args = parser.parse_args()

    # Convert paths
    manifest_path = Path(args.input_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Feature Extraction Pipeline ===")
    print(f"Loading manifest: {manifest_path}")
    subjects = load_preprocessing_manifest(manifest_path)
    
    if args.limit:
        subjects = subjects[:args.limit]
    
    results = []
    with tqdm(total=len(subjects), desc="Processing subjects") as pbar:
        for subject in subjects:
            try:
                extractor = ParcellationFeatureExtractor(
                    output_dir=output_dir,
                    parcellation_path=args.parcellation
                )
                result = extractor.process_subject(subject)
                results.append(result)
            except Exception as e:
                print(f"\nError processing {subject['unique_id']}: {e}")
                results.append({
                    'unique_id': subject['unique_id'],
                    'status': 'failed',
                    'error': str(e)
                })
            pbar.update(1)

    # Save summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count}/{len(subjects)}")
    print(f"Results saved to: {output_dir}")
    parser = argparse.ArgumentParser(description="Parcellation and Feature Extraction Module (Debug Version)")
    parser.add_argument("--input-manifest", required=True,
                       help="Path to preprocessing manifest CSV from preprocessing module")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_SUBDIR,
                       help="Output directory for feature extraction results")
    parser.add_argument("--parcellation", default=None,
                       help="Path to Schaefer-200 parcellation atlas (.nii.gz)")
    parser.add_argument("--connectivity-method", default="pearson",
                       choices=["pearson", "partial_correlation", "mutual_information"],
                       help="Method for computing functional connectivity")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of subjects processed (0 means no limit)")
    parser.add_argument("--dataset", default=None,
                       help="Process only specific dataset (optional)")
    parser.add_argument("--subject", default=None,
                       help="Process only specific subject ID (optional)")
    
    args = parser.parse_args()
    
    # Validate input manifest
    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        print(f"Error: Input manifest {manifest_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== ADHD Parcellation and Feature Extraction Module (DEBUG) ===")
    print(f"Input manifest: {manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parcellation: {args.parcellation if args.parcellation else 'Simulated Schaefer-200'}")
    print(f"Connectivity method: {args.connectivity_method}")
    
    # Load preprocessing results
    print("\n--- Loading Preprocessing Results ---")
    subjects_to_process = load_preprocessing_manifest(manifest_path)
    
    if not subjects_to_process:
        print("No successfully preprocessed subjects found in manifest.", file=sys.stderr)
        sys.exit(1)
    
    # Filter subjects based on arguments
    if args.dataset:
        subjects_to_process = [s for s in subjects_to_process if s['dataset'] == args.dataset]
        print(f"DEBUG: Filtered by dataset '{args.dataset}': {len(subjects_to_process)} subjects")
    if args.subject:
        subjects_to_process = [s for s in subjects_to_process if s['subject_id'] == args.subject]
        print(f"DEBUG: Filtered by subject '{args.subject}': {len(subjects_to_process)} subjects")
    if args.limit and args.limit > 0:
        subjects_to_process = subjects_to_process[:args.limit]
        print(f"DEBUG: Limited to {args.limit} subjects: {len(subjects_to_process)} subjects")
    
    print(f"Found {len(subjects_to_process)} subjects ready for feature extraction")
    
    # Process subjects
    print("\n--- Feature Extraction Phase ---")
    try:
        print("DEBUG: Initializing feature extractor...")
        extractor = ParcellationFeatureExtractor(
            output_dir, 
            args.parcellation, 
            args.connectivity_method
        )
        print("DEBUG: Feature extractor initialized successfully")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    results = []
    start_time = time.time()
    
    # Simple progress tracking without nested bars
    print(f"\nProcessing {len(subjects_to_process)} subjects...")
    
    for i, subject_info in enumerate(subjects_to_process):
        unique_id = subject_info['unique_id']
        
        print(f"\n--- Processing Subject {i+1}/{len(subjects_to_process)}: {unique_id} ---")
        
        try:
            result = extractor.process_subject(subject_info)
            results.append(result)
            
            if result['status'] == 'success':
                quality = result['quality_metrics']
                print(f"✓ SUCCESS: {unique_id}")
                print(f"  Processing time: {result['processing_time']:.1f}s")
                print(f"  ROI coverage: {quality.get('roi_coverage', 0):.1f}%")
                print(f"  Mean connectivity: {quality.get('mean_connectivity', 0):.3f}")
                print(f"  Outputs: {len(result.get('outputs', {}))}")
            else:
                print(f"✗ FAILED: {unique_id}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ CRITICAL ERROR processing {unique_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error result
            error_result = {
                'subject_info': subject_info,
                'unique_id': unique_id,
                'status': 'failed',
                'processing_time': 0,
                'outputs': {},
                'quality_metrics': {},
                'error': f"Critical error: {str(e)}"
            }
            results.append(error_result)
        
        # Progress update
        successes = sum(1 for r in results if r.get('status') == 'success')
        failures = len(results) - successes
        elapsed = time.time() - start_time
        avg_time = elapsed / len(results) if results else 0
        remaining = (len(subjects_to_process) - len(results)) * avg_time
        
        print(f"Progress: {len(results)}/{len(subjects_to_process)} | "
              f"Success: {successes} | Failed: {failures} | "
              f"Avg: {avg_time:.1f}s | Est. remaining: {remaining/60:.1f}min")
    
    total_time = time.time() - start_time
    
    # Save feature extraction manifest
    print("\n--- Saving Results ---")
    feature_manifest_path = output_dir / MANIFEST_FILENAME
    try:
        extractor.save_manifest(results, feature_manifest_path)
        print(f"✓ Manifest saved: {feature_manifest_path}")
    except Exception as e:
        print(f"✗ Error saving manifest: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    successes = sum(1 for r in results if r.get('status') == 'success')
    failures = len(results) - successes
    
    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Success rate: {(successes/len(results)*100):.1f}%" if results else "0.0%")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per subject: {total_time/len(results):.1f}s" if results else "N/A")
    
    if successes > 0:
        # Compute average quality metrics
        successful_results = [r for r in results if r.get('status') == 'success']
        avg_roi_coverage = np.mean([r.get('quality_metrics', {}).get('roi_coverage', 0) 
                                   for r in successful_results])
        avg_connectivity = np.mean([r.get('quality_metrics', {}).get('mean_connectivity', 0) 
                                   for r in successful_results])
        avg_connectivity_density = np.mean([r.get('quality_metrics', {}).get('connectivity_density', 0) 
                                          for r in successful_results])
        
        print(f"\nQuality Metrics (Average):")
        print(f"  ROI coverage: {avg_roi_coverage:.1f}%")
        print(f"  Mean connectivity: {avg_connectivity:.3f}")
        print(f"  Connectivity density: {avg_connectivity_density:.1f}%")
    
    if failures > 0:
        print(f"\nFailed subjects:")
        failed_results = [r for r in results if r.get('status') == 'failed'][:5]  # Show first 5 failures
        for result in failed_results:
            error_msg = result.get('error', 'Unknown error')
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + '...'
            print(f"  - {result.get('unique_id', 'Unknown')}: {error_msg}")
        
        if failures > 5:
            print(f"  ... and {failures - 5} more failures (see manifest for details)")
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"Feature extraction manifest: {feature_manifest_path}")
    
    print("\nDEBUG: Script completed successfully")


if __name__ == "__main__":
    main()
"""
Parcellation and Feature Extraction Module for ADHD Classification using GNN-STAN
Converts preprocessed fMRI data to ROI time-series and functional connectivity matrices

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
    
    def __init__(self, parcellation_path: str):
        """
        Initialize with required parcellation path
        
        Args:
            parcellation_path: Path to Schaefer-200 parcellation atlas (.nii.gz)
        """
        if not parcellation_path:
            raise ValueError("Parcellation path is required. Please provide path to Schaefer-200 atlas.")
        
        self.parcellation_path = Path(parcellation_path)
        if not self.parcellation_path.exists():
            raise FileNotFoundError(f"Parcellation atlas not found: {parcellation_path}")
        
        self.atlas_data = None
        self.roi_labels = None
        self.n_rois = 200
        
    def load_parcellation(self) -> bool:
        """Load Schaefer-200 parcellation atlas"""
        try:
            print(f"Loading Schaefer-200 parcellation from: {self.parcellation_path}")
            atlas_img = nib.load(str(self.parcellation_path))
            self.atlas_data = atlas_img.get_fdata().astype(int)
            
            # Generate ROI labels
            self.roi_labels = self._generate_roi_labels()
            
            # Validate parcellation
            unique_rois = np.unique(self.atlas_data)
            unique_rois = unique_rois[unique_rois > 0]  # Remove background
            
            if len(unique_rois) != 200:
                print(f"Warning: Expected 200 ROIs but found {len(unique_rois)} in parcellation")
                # Update n_rois to actual number found
                self.n_rois = len(unique_rois)
                # Truncate or extend labels as needed
                if len(self.roi_labels) > self.n_rois:
                    self.roi_labels = self.roi_labels[:self.n_rois]
                elif len(self.roi_labels) < self.n_rois:
                    self.roi_labels.extend([f"ROI_{i+1}" for i in range(len(self.roi_labels), self.n_rois)])
            
            print(f"Successfully loaded parcellation with {len(unique_rois)} ROIs")
            print(f"Atlas shape: {self.atlas_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading parcellation: {e}")
            return False
    
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
        
        # Ensure data dimensions match
        if fmri_data.shape[:3] != self.atlas_data.shape:
            raise ValueError(f"fMRI data shape {fmri_data.shape[:3]} doesn't match "
                           f"parcellation shape {self.atlas_data.shape}")
        
        n_timepoints = fmri_data.shape[-1]
        roi_timeseries = np.zeros((n_timepoints, self.n_rois))
        
        # Apply brain mask if provided
        if brain_mask is not None:
            mask_3d = brain_mask > 0
        else:
            # Create basic mask from atlas (non-zero regions)
            mask_3d = self.atlas_data > 0
        
        # Get unique ROI IDs actually present in the atlas
        unique_roi_ids = np.unique(self.atlas_data)
        unique_roi_ids = unique_roi_ids[unique_roi_ids > 0]  # Remove background (0)
        
        # Extract time series for each ROI
        for i, roi_id in enumerate(unique_roi_ids):
            if i >= self.n_rois:  # Safety check
                break
                
            roi_mask = (self.atlas_data == roi_id) & mask_3d
            
            if np.sum(roi_mask) > 0:
                # Extract mean time series for this ROI
                roi_voxels = fmri_data[roi_mask]  # (n_voxels, time)
                mean_timeseries = np.mean(roi_voxels, axis=0)
                roi_timeseries[:, i] = mean_timeseries
            else:
                # Fill with NaN for ROIs with no valid voxels
                roi_timeseries[:, i] = np.nan
                print(f"Warning: No valid voxels found for ROI {roi_id}")
        
        # Check for ROIs with no signal
        empty_rois = np.sum(np.isnan(roi_timeseries), axis=0)
        if np.any(empty_rois == n_timepoints):
            n_empty = np.sum(empty_rois == n_timepoints)
            print(f"Warning: {n_empty} ROIs have no valid signal")
        
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
        
        # Handle NaN values by setting them to 0
        timeseries_clean = np.nan_to_num(timeseries, nan=0.0)
        
        # Check for ROIs with no variance (all zero or constant values)
        roi_std = np.std(timeseries_clean, axis=0)
        valid_roi_mask = roi_std > 1e-10  # Very small threshold for numerical stability
        
        if not np.any(valid_roi_mask):
            print("Warning: No ROIs with valid signal variance found")
            return np.zeros((n_rois, n_rois))
        
        # Standardize time series if requested
        if self.standardize:
            scaler = StandardScaler()
            timeseries_norm = scaler.fit_transform(timeseries_clean)
        else:
            timeseries_norm = timeseries_clean
        
        # Initialize connectivity matrix
        connectivity_matrix = np.zeros((n_rois, n_rois))
        
        # Compute connectivity based on method
        if self.method == 'pearson':
            # Use only valid ROIs for correlation computation
            valid_timeseries = timeseries_norm[:, valid_roi_mask]
            if valid_timeseries.shape[1] > 1:
                valid_corr = np.corrcoef(valid_timeseries.T)
                # Place valid correlations back in full matrix
                valid_indices = np.where(valid_roi_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        connectivity_matrix[idx_i, idx_j] = valid_corr[i, j]
                        
        elif self.method == 'partial_correlation':
            connectivity_matrix = self._partial_correlation(timeseries_norm, valid_roi_mask)
        elif self.method == 'mutual_information':
            connectivity_matrix = self._mutual_information(timeseries_norm, valid_roi_mask)
        else:
            raise ValueError(f"Unknown connectivity method: {self.method}")
        
        # Handle NaN values in connectivity matrix
        connectivity_matrix = np.nan_to_num(connectivity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Set diagonal to zero (remove self-connections)
        np.fill_diagonal(connectivity_matrix, 0.0)
        
        return connectivity_matrix
    
    def _partial_correlation(self, timeseries: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix"""
        n_rois = timeseries.shape[1]
        partial_corr_matrix = np.zeros((n_rois, n_rois))
        
        try:
            from sklearn.covariance import GraphicalLassoCV
            # Use only valid ROIs
            valid_timeseries = timeseries[:, valid_mask]
            if valid_timeseries.shape[1] > 1:
                model = GraphicalLassoCV(cv=3, max_iter=100)
                model.fit(valid_timeseries)
                valid_partial_corr = -model.precision_
                
                # Place back in full matrix
                valid_indices = np.where(valid_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        partial_corr_matrix[idx_i, idx_j] = valid_partial_corr[i, j]
                        
        except ImportError:
            print("Warning: sklearn not available for partial correlation. Using Pearson correlation.")
            # Fallback to regular correlation
            valid_timeseries = timeseries[:, valid_mask]
            if valid_timeseries.shape[1] > 1:
                valid_corr = np.corrcoef(valid_timeseries.T)
                valid_indices = np.where(valid_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        partial_corr_matrix[idx_i, idx_j] = valid_corr[i, j]
        except Exception as e:
            print(f"Warning: Partial correlation failed ({e}). Using Pearson correlation.")
            # Another fallback to regular correlation
            valid_timeseries = timeseries[:, valid_mask]
            if valid_timeseries.shape[1] > 1:
                valid_corr = np.corrcoef(valid_timeseries.T)
                valid_indices = np.where(valid_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        partial_corr_matrix[idx_i, idx_j] = valid_corr[i, j]
        
        return partial_corr_matrix
    
    def _mutual_information(self, timeseries: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute mutual information matrix"""
        n_rois = timeseries.shape[1]
        mi_matrix = np.zeros((n_rois, n_rois))
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            valid_indices = np.where(valid_mask)[0]
            
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    if i != j:  # Skip diagonal
                        mi_val = mutual_info_regression(
                            timeseries[:, [idx_i]], timeseries[:, idx_j], 
                            discrete_features=False, random_state=42
                        )[0]
                        mi_matrix[idx_i, idx_j] = mi_val
                        
        except ImportError:
            print("Warning: sklearn not available for mutual information. Using Pearson correlation.")
            # Fallback to correlation
            valid_timeseries = timeseries[:, valid_mask]
            if valid_timeseries.shape[1] > 1:
                valid_corr = np.corrcoef(valid_timeseries.T)
                valid_indices = np.where(valid_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        mi_matrix[idx_i, idx_j] = valid_corr[i, j]
        except Exception as e:
            print(f"Warning: Mutual information failed ({e}). Using Pearson correlation.")
            # Fallback to correlation
            valid_timeseries = timeseries[:, valid_mask]
            if valid_timeseries.shape[1] > 1:
                valid_corr = np.corrcoef(valid_timeseries.T)
                valid_indices = np.where(valid_mask)[0]
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        mi_matrix[idx_i, idx_j] = valid_corr[i, j]
        
        return mi_matrix


class ParcellationFeatureExtractor:
    """Main class for parcellation-based feature extraction"""
    
    def __init__(self, output_dir: Path, parcellation_path: str,
                 connectivity_method: str = 'pearson'):
        self.output_dir = output_dir
        self.parcellation = SchaeferParcellation(parcellation_path)
        self.connectivity_extractor = FunctionalConnectivityExtractor(method=connectivity_method)
        self.results = []
        
        # Load parcellation
        if not self.parcellation.load_parcellation():
            raise RuntimeError("Failed to load parcellation atlas")
    
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
        
        try:
            start_time = time.time()
            
            # Create progress bar for this subject's processing steps
            steps = [
                "Loading preprocessed fMRI data",
                "Loading brain mask",
                f"Extracting time series from {self.parcellation.n_rois} ROIs",
                "Computing functional connectivity matrix",
                "Saving outputs"
            ]
            
            with tqdm(total=len(steps), desc=f"Processing {unique_id}", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]',
                     leave=False) as pbar:
                
                # Step 1: Load preprocessed fMRI data
                pbar.set_description(f"Processing {unique_id} - Loading fMRI data")
                fmri_img = nib.load(preprocessed_path)
                fmri_data = fmri_img.get_fdata()
                
                if len(fmri_data.shape) != 4:
                    raise ValueError(f"Expected 4D fMRI data, got {len(fmri_data.shape)}D")
                
                pbar.update(1)
                
                # Step 2: Load brain mask if available
                pbar.set_description(f"Processing {unique_id} - Loading brain mask")
                brain_mask = None
                if mask_path and Path(mask_path).exists():
                    mask_img = nib.load(mask_path)
                    brain_mask = mask_img.get_fdata()
                    if brain_mask.shape != fmri_data.shape[:3]:
                        print(f"Warning: Brain mask shape {brain_mask.shape} doesn't match "
                              f"fMRI data shape {fmri_data.shape[:3]}. Ignoring mask.")
                        brain_mask = None
                pbar.update(1)
                
                # Step 3: Extract ROI time series
                pbar.set_description(f"Processing {unique_id} - Extracting ROI time series")
                roi_timeseries = self.parcellation.extract_roi_timeseries(fmri_data, brain_mask)
                pbar.update(1)
                
                # Step 4: Compute functional connectivity
                pbar.set_description(f"Processing {unique_id} - Computing connectivity")
                connectivity_matrix = self.connectivity_extractor.compute_connectivity(roi_timeseries)
                pbar.update(1)
                
                # Step 5: Save outputs
                pbar.set_description(f"Processing {unique_id} - Saving outputs")
                outputs = self._save_outputs(
                    subject_out_dir, unique_id, roi_timeseries, connectivity_matrix
                )
                pbar.update(1)
            
            processing_time = time.time() - start_time
            
            # Quality metrics
            quality_metrics = self._compute_quality_metrics(roi_timeseries, connectivity_matrix)
            
            success_result = {
                'subject_info': subject_info,
                'unique_id': unique_id,
                'status': 'success',
                'processing_time': processing_time,
                'outputs': outputs,
                'quality_metrics': quality_metrics,
                'error': None
            }
            
            return success_result
            
        except Exception as e:
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
            # 1. Save ROI time-series as CSV
            timeseries_path = output_dir / f"{unique_id}_roi_timeseries.csv"
            timeseries_df = pd.DataFrame(
                roi_timeseries, 
                columns=[f"ROI_{i+1}_{label}" for i, label in enumerate(self.parcellation.roi_labels)]
            )
            timeseries_df.to_csv(timeseries_path, index=False)
            outputs['roi_timeseries'] = str(timeseries_path)
            
            # 2. Save connectivity matrix as numpy array
            connectivity_path = output_dir / f"{unique_id}_connectivity_matrix.npy"
            np.save(connectivity_path, connectivity_matrix)
            outputs['connectivity_matrix'] = str(connectivity_path)
            
            # 3. Save connectivity matrix as CSV (for inspection)
            connectivity_csv_path = output_dir / f"{unique_id}_connectivity_matrix.csv"
            connectivity_df = pd.DataFrame(
                connectivity_matrix,
                index=[f"ROI_{i+1}" for i in range(len(self.parcellation.roi_labels))],
                columns=[f"ROI_{i+1}" for i in range(len(self.parcellation.roi_labels))]
            )
            connectivity_df.to_csv(connectivity_csv_path)
            outputs['connectivity_csv'] = str(connectivity_csv_path)
            
            # 4. Save ROI labels
            labels_path = output_dir / f"{unique_id}_roi_labels.csv"
            labels_df = pd.DataFrame({
                'ROI_ID': range(1, len(self.parcellation.roi_labels) + 1),
                'ROI_Label': self.parcellation.roi_labels
            })
            labels_df.to_csv(labels_path, index=False)
            outputs['roi_labels'] = str(labels_path)
            
        except Exception as e:
            print(f"Warning: Error saving outputs for {unique_id}: {e}")
        
        return outputs
    
    def _compute_quality_metrics(self, roi_timeseries: np.ndarray, 
                               connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """Compute quality control metrics"""
        try:
            # ROI coverage (percentage of ROIs with non-zero signal)
            roi_std = np.std(roi_timeseries, axis=0)
            non_zero_rois = np.sum(roi_std > 1e-10)  # Use small threshold for numerical stability
            roi_coverage = (non_zero_rois / roi_timeseries.shape[1]) * 100
            
            # Mean connectivity strength
            # Only consider upper triangle (excluding diagonal)
            upper_triangle = np.triu(connectivity_matrix, k=1)
            valid_connections = upper_triangle[upper_triangle != 0]
            mean_connectivity = np.mean(np.abs(valid_connections)) if len(valid_connections) > 0 else 0.0
            
            # Signal-to-noise ratio approximation
            mean_signal = np.mean(roi_std[roi_std > 0]) if np.any(roi_std > 0) else 0.0
            
            # Connectivity density (percentage of connections above threshold)
            threshold = 0.1
            strong_connections = np.sum(np.abs(upper_triangle) > threshold)
            total_possible_connections = upper_triangle.size
            connectivity_density = (strong_connections / total_possible_connections * 100) if total_possible_connections > 0 else 0
            
            return {
                'roi_coverage': roi_coverage,
                'mean_connectivity': mean_connectivity,
                'mean_signal_std': mean_signal,
                'connectivity_density': connectivity_density,
                'n_timepoints': roi_timeseries.shape[0],
                'n_rois': roi_timeseries.shape[1],
                'n_valid_rois': non_zero_rois
            }
            
        except Exception as e:
            print(f"Warning: Error computing quality metrics: {e}")
            return {
                'roi_coverage': 0.0,
                'mean_connectivity': 0.0,
                'mean_signal_std': 0.0,
                'connectivity_density': 0.0,
                'n_timepoints': roi_timeseries.shape[0] if roi_timeseries.size > 0 else 0,
                'n_rois': roi_timeseries.shape[1] if roi_timeseries.size > 0 else 0,
                'n_valid_rois': 0
            }
    
    def save_manifest(self, results: List[Dict[str, Any]], manifest_path: Path):
        """Save processing manifest"""
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
                'n_valid_rois': quality.get('n_valid_rois', 0),
                'error': result.get('error', '')
            }
            manifest_data.append(manifest_entry)
        
        df = pd.DataFrame(manifest_data)
        df.to_csv(manifest_path, index=False)
        print(f"Saved feature extraction manifest to {manifest_path}")


def load_preprocessing_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load preprocessing manifest to get input files"""
    try:
        df = pd.read_csv(manifest_path)
        
        # Convert to list of dictionaries and filter successful preprocessing
        subjects = []
        for _, row in df.iterrows():
            if row['status'] == 'success' and pd.notna(row['preprocessed_bold']) and row['preprocessed_bold']:
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
        
        return subjects
        
    except Exception as e:
        print(f"Error loading preprocessing manifest: {e}")
        return []


def main():

    print("--- Parcellation and Feature Extraction Phase ---")
    parser = argparse.ArgumentParser(description="Parcellation and Feature Extraction Module")
    parser.add_argument("--preprocessing-manifest", type=str, required=True,
                       help="Path to preprocessing manifest CSV")
    parser.add_argument("--parcellation-path", type=str, required=True,
                       help="Path to Schaefer-200 parcellation atlas (.nii.gz)")
    parser.add_argument("--output-dir", type=str, default="feature_extraction_outputs",
                       help="Output directory for extracted features")
    parser.add_argument("--connectivity-method", type=str, default="pearson",
                       choices=["pearson", "partial_correlation", "mutual_information"],
                       help="Method to compute functional connectivity")
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    manifest_path = Path(args.preprocessing_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load subject list from preprocessing manifest
    print("Loading preprocessing manifest...")
    subjects = load_preprocessing_manifest(manifest_path)
    print(f"Found {len(subjects)} subjects with successful preprocessing")
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = ParcellationFeatureExtractor(
        output_dir=output_dir,
        parcellation_path=args.parcellation_path,
        connectivity_method=args.connectivity_method
    )
    
    # Process each subject
    results = []
    print("\nProcessing subjects:")
    for subject in tqdm(subjects, desc="Overall Progress"):
        try:
            result = extractor.process_subject(subject)
            results.append(result)
        except Exception as e:
            print(f"\nError processing subject {subject['unique_id']}: {e}")
            results.append({
                'subject_info': subject,
                'unique_id': subject['unique_id'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Save manifest
    manifest_path = output_dir / MANIFEST_FILENAME
    extractor.save_manifest(results, manifest_path)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(subjects)} subjects")
    print(f"Results saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
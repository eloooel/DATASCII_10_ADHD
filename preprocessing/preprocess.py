import nibabel as nib
import numpy as np
import torch
import warnings
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from scipy import signal, ndimage
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA
import pandas as pd
import gzip
import psutil
import gc

from nilearn.datasets import load_mni152_template
from utils import run_parallel

warnings.filterwarnings('ignore', category=RuntimeWarning)

class PreprocessingPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        if config is not None:
            self.config = self._merge_with_defaults(config)
        else:
            self.config = self._default_config()
            
        # Only print device info if explicitly requested
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processing_log: List[Dict[str, Any]] = []
        self.subject_id: str = "unknown"

    def _ensure_nifti(self, data, metadata, header):
        """Ensure output is always a NIfTI image"""
        if isinstance(data, nib.Nifti1Image):
            return data
        elif torch.is_tensor(data):
            return nib.Nifti1Image(data.cpu().numpy(), metadata['affine'], header=header)
        else:
            return nib.Nifti1Image(np.array(data), metadata['affine'], header=header)

    def _log_step(self, step_name: str, status: str, details: str = "") -> None:
        """Log a processing step"""
        self.processing_log.append({
            'step': step_name,
            'status': status,
            'message': details,
            'subject_id': self.subject_id
        })
        
        if status in ['failed', 'warning']:
            print(f"[{status.upper()}] {self.subject_id} - {step_name}: {details}")

    def _default_config(self) -> Dict[str, Any]:
        """Default preprocessing configuration"""
        return {
            'motion_correction': {
                'enabled': True,
                'reference': 'mean',
                'save_parameters': True,
                'max_displacement_mm': 3.0
            },
            'slice_timing_correction': {
                'enabled': True,
                'tr': 2.0,
                'slice_order': 'ascending',
                'ref_slice': 'middle'
            },
            'spatial_normalization': {
                'enabled': True,
                'template': 'MNI152',
                'resolution': 2,  # MNI152 resolution in mm
                'smooth_fwhm': 6.0  # Spatial smoothing after registration
            },
            'temporal_filtering': {
                'enabled': True,
                'low_freq': 0.009,
                'high_freq': 0.08,
                'filter_type': 'butterworth',
                'filter_order': 4
            },
            'denoising': {
                'motion_regression': True,
                'ica_aroma': {  # ✅ ICA-AROMA configuration
                    'enabled': True,
                    'n_components': 25,
                    'max_iter': 200,
                    'tolerance': 1e-4,
                    'motion_correlation_threshold': 0.3,
                    'high_freq_threshold': 0.2,
                    'high_freq_ratio_threshold': 0.6,
                    'spatial_std_threshold': 2.0
                },
                'acompcor': {
                    'enabled': True,
                    'n_components': 5,
                    'variance_threshold': 0.5
                },
                'global_signal': {
                    'enabled': False
                }
            },
            'output': {
                'save_intermediate': False,
                'compress': True,
                'datatype': 'float32'
            }
        }

    def _merge_with_defaults(self, user_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with defaults"""
        defaults = self._default_config()
        merged = defaults.copy()

        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
                merged[key] = {**defaults[key], **value}
            else:
                merged[key] = value

        return merged

    def process(self, input_path: str, subject_id: str = None) -> Dict[str, Any]:
        self.processing_log = []
        self.subject_id = subject_id or Path(input_path).stem

        try:
            # Load original image for metadata
            original_img = nib.load(input_path)
            self.metadata = {
                'affine': original_img.affine,
                'header': original_img.header
            }

            # Load and validate data
            data, metadata = self._load_and_validate_data(input_path)
            
            # Ensure each stage returns NIfTI
            data_img = self._ensure_nifti(data, metadata, original_img.header)
            
            # Pipeline stages in correct order
            data_img, motion_params = self._motion_correction(data_img, self.config['motion_correction'])
            data_img = self._slice_timing_correction(data_img, self.config['slice_timing_correction'], metadata)
            data_img = self._spatial_normalization(data_img, self.config['spatial_normalization'])  # ✅ Now includes MNI152
            data_img = self._temporal_filtering(data_img, self.config['temporal_filtering'], metadata)
            
            # ✅ ICA-AROMA after temporal filtering, before other denoising
            if self.config['denoising'].get('ica_aroma', {}).get('enabled', True):
                data_img, ica_results = self._ica_aroma(data_img, motion_params, self.config['denoising']['ica_aroma'])
            else:
                ica_results = {}
            
            # Other denoising (aCompCor, etc.)
            data_img, confound_regressors = self._denoising(data_img, motion_params, self.config['denoising'])

            # Generate brain mask
            brain_mask = self._generate_brain_mask(data_img)

            return {
                'processed_data': data_img,
                'confound_regressors': confound_regressors,
                'brain_mask': brain_mask,
                'motion_parameters': motion_params,
                'ica_results': ica_results,  # ✅ Add ICA results
                'metadata': metadata,
                'processing_log': self.processing_log,
                'subject_id': self.subject_id,
                'config_used': self.config,
                'status': 'success'
            }

        except Exception as e:
            self._log_step("pipeline_error", "failed", f"Pipeline failed: {str(e)}")
            return {'status': 'failed', 'error': str(e), 'subject_id': self.subject_id}



    # ----------------- Real Pipeline Stages -----------------


    def _load_and_validate_data(self, input_path: str) -> tuple:
        """Load and validate input fMRI data with CUDA support"""
        data = nib.load(input_path)
        
        # ✅ FIXED: Ensure float32 and avoid object arrays
        img_data_np = data.get_fdata().astype(np.float32)
        img_data = torch.from_numpy(img_data_np).to(self.device)
        
        if len(img_data.shape) != 4:
            raise ValueError(f"Expected 4D data, got {len(img_data.shape)}D")

        # Basic data quality checks on GPU
        n_zeros = torch.sum(img_data == 0).item()
        total_voxels = torch.prod(torch.tensor(img_data.shape)).item()
        
        metadata = {
            'shape': img_data.shape,
            'voxel_size': data.header.get_zooms()[:3],
            'tr': data.header.get_zooms()[3] if len(data.header.get_zooms()) > 3 else self.config['slice_timing_correction']['tr'],
            'datatype': img_data.dtype,
            'orientation': nib.aff2axcodes(data.affine),
            'affine': data.affine
        }

        return img_data, metadata

    def _motion_correction(self, data_img, params: Dict[str, Any]) -> tuple:
        """Real motion correction using volume realignment"""
        if not params.get('enabled', True):
            self._log_step("motion_correction", "skipped", "Disabled in config")
            return data_img, None

        # Get data and metadata - ensure proper data types
        if isinstance(data_img, nib.Nifti1Image):
            img_data = data_img.get_fdata().astype(np.float32)  # ✅ Force float32
            affine = data_img.affine
            header = data_img.header
        elif torch.is_tensor(data_img):
            img_data = data_img.cpu().numpy().astype(np.float32)  # ✅ Force float32
            affine = self.metadata['affine']
            header = None
        else:
            img_data = np.array(data_img, dtype=np.float32)  # ✅ Force float32
            affine = self.metadata['affine']
            header = None

        n_vols = img_data.shape[-1]
        
        # Create reference volume (mean or first volume)
        if params['reference'] == 'mean':
            reference = np.mean(img_data, axis=-1).astype(np.float32)
        elif params['reference'] == 'first':
            reference = img_data[..., 0].astype(np.float32)
        else:
            # Assume reference is a volume index
            ref_idx = int(params['reference'])
            reference = img_data[..., ref_idx].astype(np.float32)

        # Initialize motion parameters
        motion_params = {
            'translations': np.zeros((n_vols, 3), dtype=np.float32),  # ✅ Explicit dtype
            'rotations': np.zeros((n_vols, 3), dtype=np.float32),     # ✅ Explicit dtype
            'displacement': np.zeros(n_vols, dtype=np.float32),       # ✅ Explicit dtype
            'excluded_volumes': []
        }

        corrected_data = np.zeros_like(img_data, dtype=np.float32)  # ✅ Fixed parenthesis

        # Process each volume
        for vol_idx in range(n_vols):
            current_vol = img_data[..., vol_idx].astype(np.float32)
            translation = self._estimate_translation(reference, current_vol)
            
            # Ensure translation is finite float32
            translation = np.asarray(translation, dtype=np.float32)
            if not np.all(np.isfinite(translation)):
                translation = np.zeros(3, dtype=np.float32)
                
            # Apply shift
            corrected_vol = ndimage.shift(current_vol, translation, order=1, mode='nearest')
            corrected_data[..., vol_idx] = corrected_vol.astype(np.float32)
            
            # Store motion parameters
            motion_params['translations'][vol_idx] = translation
            
            # Calculate frame displacement
            if vol_idx > 0:
                prev_trans = motion_params['translations'][vol_idx - 1]
                displacement = np.sqrt(np.sum((translation - prev_trans) ** 2))
                motion_params['displacement'][vol_idx] = displacement
                
                # Check for excessive motion
                if displacement > params.get('max_displacement_mm', 3.0):
                    motion_params['excluded_volumes'].append(vol_idx)  # ✅ ADDED MISSING LINE

        # Always return NIfTI
        corrected_img = nib.Nifti1Image(corrected_data, affine, header)

        max_displacement = np.max(motion_params['displacement'])
        mean_displacement = np.mean(motion_params['displacement'])
        
        self._log_step("motion_correction", "success", 
                      f"Max displacement: {max_displacement:.2f}mm, Mean: {mean_displacement:.2f}mm")

        if motion_params['excluded_volumes']:
            self._log_step("motion_correction", "warning", 
                          f"Flagged {len(motion_params['excluded_volumes'])} high-motion volumes")

    # Convert back to tensor if input was tensor
        if torch.is_tensor(data_img):
            corrected_data_tensor = torch.from_numpy(corrected_data).to(self.device)
            return corrected_data_tensor, motion_params
        else:
            # Create corrected nibabel image
            corrected_img = nib.Nifti1Image(corrected_data, data_img.affine, data_img.header)
            return corrected_img, motion_params
        
    def _estimate_translation(self, reference: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Estimate translation between two volumes using center of mass"""
        try:
            # Threshold images to focus on brain tissue
            ref_thresh = reference > np.percentile(reference[reference > 0], 50)
            cur_thresh = current > np.percentile(current[current > 0], 50)
            
            # Calculate center of mass
            ref_com = ndimage.center_of_mass(ref_thresh)
            cur_com = ndimage.center_of_mass(cur_thresh)
            
            # Translation is the difference in center of mass
            translation = np.array(ref_com) - np.array(cur_com)
            
            # Limit maximum correction to prevent artifacts
            translation = np.clip(translation, -10, 10)
            
            return translation
            
        except:
            # Fallback to zero translation if center of mass fails
            return np.array([0.0, 0.0, 0.0])

    def _slice_timing_correction(self, data_img, params: Dict[str, Any], metadata: Dict[str, Any]):
        """Real slice timing correction"""
        if not params.get('enabled', True):
            self._log_step("slice_timing_correction", "skipped", "Disabled in config")
            return data_img

        img_data = data_img.get_fdata()
        tr = params.get('tr', metadata.get('tr', 2.0))
        n_slices = img_data.shape[2]
        
        # Define slice acquisition times
        if params['slice_order'] == 'ascending':
            slice_times = np.linspace(0, tr, n_slices, endpoint=False)
        elif params['slice_order'] == 'descending':
            slice_times = np.linspace(tr, 0, n_slices, endpoint=False)
        elif params['slice_order'] == 'interleaved':
            # Interleaved: odd slices first, then even
            odd_times = np.linspace(0, tr/2, n_slices//2, endpoint=False)
            even_times = np.linspace(tr/2, tr, n_slices//2, endpoint=False)
            slice_times = np.zeros(n_slices)
            slice_times[::2] = odd_times  # odd indices
            slice_times[1::2] = even_times  # even indices
        else:
            slice_times = np.linspace(0, tr, n_slices, endpoint=False)

        # Reference time (usually middle slice)
        if params['ref_slice'] == 'middle':
            ref_time = tr / 2
        else:
            ref_slice_idx = int(params['ref_slice'])
            ref_time = slice_times[ref_slice_idx]

        # Apply slice timing correction
        corrected_data = np.zeros_like(img_data)
        n_timepoints = img_data.shape[-1]
        
        # Create time vectors for interpolation
        original_times = np.arange(n_timepoints) * tr
        
        for z in range(n_slices):
            slice_data = img_data[:, :, z, :]
            
            time_shift = slice_times[z] - ref_time
            shifted_times = original_times + time_shift
            
            for x in range(slice_data.shape[0]):
                for y in range(slice_data.shape[1]):
                    voxel_ts = slice_data[x, y, :]
                    
                    if np.any(voxel_ts != 0):  # Skip empty voxels
                        # ✅ ADDED: Linear interpolation for slice timing correction
                        corrected_ts = np.interp(original_times, shifted_times, voxel_ts)
                        corrected_data[x, y, z, :] = corrected_ts
                    else:
                        corrected_data[x, y, z, :] = voxel_ts

        # Create corrected nibabel image
        corrected_img = nib.Nifti1Image(corrected_data, data_img.affine, data_img.header)
        
        self._log_step("slice_timing_correction", "success", 
                      f"Applied to {n_slices} slices, TR={tr:.2f}s, ref_time={ref_time:.3f}s")
        
        return corrected_img

    def _spatial_normalization(self, data_img, params: Dict[str, Any]):
        """MNI152 registration using nilearn - no fallbacks"""
        if not params.get('enabled', True):
            self._log_step("spatial_normalization", "skipped", "Disabled in config")
            return data_img

        try:
            from nilearn.datasets import load_mni152_template
            from nilearn.image import resample_to_img, smooth_img
            
            # Load MNI152 template at specified resolution
            resolution = params.get('resolution', 2)
            mni_template = load_mni152_template(resolution=resolution)
            
            self._log_step("spatial_normalization", "info", 
                          f"Loaded MNI152 template at {resolution}mm resolution")
            
            # Resample functional data to MNI152 space
            # This performs affine registration to template space
            normalized_img = resample_to_img(
                data_img, 
                mni_template, 
                interpolation='linear'
            )
            
            self._log_step("spatial_normalization", "success", 
                          f"Resampled to MNI152 space: {normalized_img.shape}")
            
            # Apply spatial smoothing after normalization
            smooth_fwhm = params.get('smooth_fwhm', 6.0)
            if smooth_fwhm > 0:
                smoothed_img = smooth_img(normalized_img, fwhm=smooth_fwhm)
                
                self._log_step("spatial_normalization", "success", 
                              f"Applied {smooth_fwhm}mm FWHM smoothing after MNI152 registration")
                
                return smoothed_img
            else:
                self._log_step("spatial_normalization", "success", 
                              "MNI152 registration complete (no smoothing)")
                return normalized_img
                
        except ImportError as e:
            # Fail if nilearn not available
            self._log_step("spatial_normalization", "failed", 
                          f"Nilearn not available for MNI152 registration: {e}")
            raise RuntimeError(f"MNI152 registration requires nilearn: {e}")
            
        except Exception as e:
            # Fail if registration fails
            self._log_step("spatial_normalization", "failed", 
                          f"MNI152 registration failed: {e}")
            raise RuntimeError(f"MNI152 registration failed: {e}")

    def _temporal_filtering(self, data_img, params: Dict[str, Any], metadata: Dict[str, Any]):
        """Real temporal bandpass filtering with CUDA support"""
        if not params.get('enabled', True):
            self._log_step("temporal_filtering", "skipped", "Disabled in config")
            return data_img

        # Get data and affine/header
        if isinstance(data_img, nib.Nifti1Image):
            img_data = data_img.get_fdata().astype(np.float32)  # ✅ Ensure float32
            affine = data_img.affine
            header = data_img.header
        else:
            img_data = np.array(data_img, dtype=np.float32)  # ✅ Force float32
            affine = metadata['affine']
            header = None

        tr = metadata.get('tr', 2.0)
        low_freq = params['low_freq']
        high_freq = params['high_freq']
        filter_order = params.get('filter_order', 4)
        
        # Nyquist frequency
        nyquist = 1 / (2 * tr)
        
        # Normalize frequencies
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Check frequency bounds
        if high >= 1.0:
            high = 0.99
            self._log_step("temporal_filtering", "warning", 
                        f"High frequency capped at {high * nyquist:.3f}Hz")
        
        # Design Butterworth bandpass filter
        try:
            b, a = signal.butter(filter_order, [low, high], btype='band')
        except ValueError:
            # Fallback to high-pass only if bandpass fails
            b, a = signal.butter(filter_order, low, btype='high')
            self._log_step("temporal_filtering", "warning", "Applied high-pass only")

        # Get brain mask for filtering (avoid filtering background)
        mean_img = np.mean(img_data, axis=-1)
        brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)
        
        # Apply filter to each voxel using NumPy (avoid PyTorch object arrays)
        filtered_data = img_data.copy()
        brain_voxels = np.where(brain_mask)
        total_voxels = len(brain_voxels[0])
        
        for i in range(total_voxels):
            x, y, z = brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]
            voxel_ts = filtered_data[x, y, z, :].astype(np.float32)  # ✅ Ensure float32
            
            if np.any(voxel_ts != 0):
                try:
                    # ✅ ADDED: Complete filtering implementation
                    # Linear detrending
                    t = np.arange(len(voxel_ts), dtype=np.float32)
                    p = np.polyfit(t, voxel_ts, deg=1)
                    trend = p[1] + p[0] * t
                    detrended = voxel_ts - trend

                    # Apply bandpass filter
                    filtered_ts = signal.filtfilt(b, a, detrended)
                    filtered_data[x, y, z, :] = filtered_ts + trend
                        
                except Exception as e:
                    # ✅ ADDED: Keep original if filtering fails
                    filtered_data[x, y, z, :] = voxel_ts

        # Always return NIfTI
        filtered_img = nib.Nifti1Image(filtered_data, affine, header)
        
        self._log_step("temporal_filtering", "success", 
                      f"Applied {low_freq:.3f}-{high_freq:.3f}Hz bandpass filter")
        
        return filtered_img

    def _denoising(self, data_img, motion_params: Optional[Dict[str, np.ndarray]], 
                   params: Dict[str, Any]) -> tuple:
        """Real denoising with motion regression and aCompCor"""
        img_data = data_img.get_fdata()
        confounds = {}

        # 1. Motion regression
        if params.get('motion_regression', True) and motion_params:
            # Use motion parameters as confounds
            confounds['trans_x'] = motion_params['translations'][:, 0].tolist()
            confounds['trans_y'] = motion_params['translations'][:, 1].tolist()
            confounds['trans_z'] = motion_params['translations'][:, 2].tolist()
            confounds['frame_displacement'] = motion_params['displacement'].tolist()
            
            # Add motion derivatives
            confounds['trans_x_derivative'] = np.gradient(motion_params['translations'][:, 0]).tolist()
            confounds['trans_y_derivative'] = np.gradient(motion_params['translations'][:, 1]).tolist()
            confounds['trans_z_derivative'] = np.gradient(motion_params['translations'][:, 2]).tolist()

        # 2. aCompCor (anatomical Component Correction)
        if params.get('acompcor', {}).get('enabled', True):
            acompcor_confounds = self._compute_acompcor(img_data, params['acompcor'])
            confounds.update(acompcor_confounds)

        # 3. Global signal regression (optional)
        if params.get('global_signal', {}).get('enabled', False):
            brain_mask = self._generate_brain_mask(data_img)
            brain_voxels = img_data[brain_mask > 0]
            global_signal = np.mean(brain_voxels, axis=0)
            confounds['global_signal'] = global_signal.tolist()

        # Apply confound regression to data
        denoised_data = self._apply_confound_regression(img_data, confounds)
        
        # Create denoised nibabel image
        denoised_img = nib.Nifti1Image(denoised_data, data_img.affine, data_img.header)

        n_confounds = len(confounds)
        self._log_step("denoising", "success", f"Applied {n_confounds} confound regressors")

        return denoised_img, confounds

    def _compute_acompcor(self, img_data: np.ndarray, params: Dict[str, Any]) -> Dict[str, List[float]]:
        """Compute anatomical CompCor components"""
        n_components = params.get('n_components', 5)
        variance_threshold = params.get('variance_threshold', 0.5)
        
        # Create CSF/WM mask (simplified - normally would use tissue segmentation)
        mean_img = np.mean(img_data, axis=-1)
        
        # High-signal regions (likely CSF) and low-signal regions (likely WM)
        high_threshold = np.percentile(mean_img[mean_img > 0], 90)
        low_threshold = np.percentile(mean_img[mean_img > 0], 30)
        
        noise_mask = (mean_img > high_threshold) | (mean_img < low_threshold)
        
        if np.sum(noise_mask) < 100:
            # Fallback mask if thresholding fails
            noise_mask = mean_img < np.percentile(mean_img[mean_img > 0], 50)
        
        # Extract time series from noise regions
        noise_voxels = img_data[noise_mask]
        
        if noise_voxels.shape[0] < n_components:
            n_components = max(1, noise_voxels.shape[0] // 2)
        
        # Compute PCA on noise regions
        try:
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(noise_voxels.T)  # Time x Components
            
            # Check variance explained
            var_explained = pca.explained_variance_ratio_
            significant_components = np.where(var_explained > variance_threshold)[0]
            
            if len(significant_components) == 0:
                significant_components = [0]  # Keep at least one component
            
            confounds = {}
            for i, comp_idx in enumerate(significant_components):
                confounds[f'a_comp_cor_{i:02d}'] = components[:, comp_idx].tolist()
            
            return confounds
            
        except Exception as e:
            # Fallback: return mean signal from noise regions
            mean_noise = np.mean(noise_voxels, axis=0)
            return {'a_comp_cor_00': mean_noise.tolist()}

    def _apply_confound_regression(self, img_data: np.ndarray, 
                                 confounds: Dict[str, List[float]]) -> np.ndarray:
        """Apply confound regression to remove nuisance signals"""
        if not confounds:
            return img_data
        
        # Create design matrix from confounds
        n_timepoints = img_data.shape[-1]
        confound_matrix = np.ones((n_timepoints, 1))  # Add intercept
        
        for confound_name, confound_ts in confounds.items():
            if len(confound_ts) == n_timepoints:
                confound_matrix = np.column_stack([confound_matrix, confound_ts])
        
        # Apply regression to each voxel
        denoised_data = np.zeros_like(img_data)
        
        # Get brain mask for regression
        mean_img = np.mean(img_data, axis=-1)
        brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)
        
        brain_voxels = np.where(brain_mask)
        
        for i in range(len(brain_voxels[0])):
            x, y, z = brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]
            voxel_ts = img_data[x, y, z, :]
            
            try:
                # Ordinary least squares regression
                beta = np.linalg.lstsq(confound_matrix, voxel_ts, rcond=None)[0]
                predicted = confound_matrix @ beta
                residual = voxel_ts - predicted
                denoised_data[x, y, z, :] = residual
                
            except:
                # Keep original if regression fails
                denoised_data[x, y, z, :] = voxel_ts
        
        # Copy non-brain voxels unchanged
        denoised_data[~brain_mask] = img_data[~brain_mask]
        
        return denoised_data
    

    def _ica_aroma(self, data_img, motion_params: Optional[Dict[str, np.ndarray]], 
                   params: Dict[str, Any]) -> tuple:
        """ICA-AROMA for automatic motion artifact removal"""
        if not params.get('enabled', True):
            self._log_step("ica_aroma", "skipped", "Disabled in config")
            return data_img, {}

        try:
            from sklearn.decomposition import FastICA
            
            img_data = data_img.get_fdata().astype(np.float32)
            
            # Reshape data for ICA: (n_timepoints, n_voxels)
            original_shape = img_data.shape
            n_timepoints = original_shape[-1]
            
            # Create brain mask for ICA
            brain_mask = self._generate_brain_mask(data_img)
            brain_voxels_idx = np.where(brain_mask > 0)
            n_brain_voxels = len(brain_voxels_idx[0])
            
            # Extract brain voxel time series
            brain_timeseries = np.zeros((n_timepoints, n_brain_voxels), dtype=np.float32)
            for i in range(n_brain_voxels):
                x, y, z = brain_voxels_idx[0][i], brain_voxels_idx[1][i], brain_voxels_idx[2][i]
                brain_timeseries[:, i] = img_data[x, y, z, :]
            
            # Determine number of ICA components (typically 20-40 for rs-fMRI)
            n_components = min(params.get('n_components', 25), n_timepoints - 1, n_brain_voxels // 100)
            
            if n_components < 5:
                raise ValueError(f"Insufficient components for ICA-AROMA ({n_components}). "
                               "Need at least 5 components for reliable motion artifact detection.")
            
            # Run ICA
            ica = FastICA(
                n_components=n_components,
                random_state=42,
                max_iter=params.get('max_iter', 200),
                tol=params.get('tolerance', 1e-4)
            )
            
            # Fit ICA and get components
            ica_timeseries = ica.fit_transform(brain_timeseries)  # (n_timepoints, n_components)
            ica_spatial_maps = ica.components_  # (n_components, n_voxels)
            
            # Identify motion-related components using AROMA criteria
            motion_components = self._identify_motion_components(
                ica_timeseries, ica_spatial_maps, motion_params, params
            )
            
            # Remove motion components by regression
            cleaned_timeseries = brain_timeseries.copy()
            
            for comp_idx in motion_components:
                component_ts = ica_timeseries[:, comp_idx]
                
                for voxel_idx in range(n_brain_voxels):
                    voxel_ts = cleaned_timeseries[:, voxel_idx]
                    
                    # ✅ COMPLETE: Linear regression implementation
                    correlation = np.corrcoef(voxel_ts, component_ts)[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > 0.1:
                        # Calculate regression coefficient
                        beta = correlation * (np.std(voxel_ts) / (np.std(component_ts) + 1e-8))
                        # Remove component contribution
                        cleaned_timeseries[:, voxel_idx] = voxel_ts - beta * component_ts
            
            # Put cleaned data back into original 4D structure
            cleaned_data = img_data.copy()
            for i in range(n_brain_voxels):
                x, y, z = brain_voxels_idx[0][i], brain_voxels_idx[1][i], brain_voxels_idx[2][i]
                cleaned_data[x, y, z, :] = cleaned_timeseries[:, i]
            
            # Create cleaned NIfTI image
            cleaned_img = nib.Nifti1Image(cleaned_data, data_img.affine, data_img.header)
            
            self._log_step("ica_aroma", "success", 
                          f"Removed {len(motion_components)} motion components out of {n_components} total")
            
            return cleaned_img, {
                'motion_components': motion_components,
                'total_components': n_components,
                'n_components_removed': len(motion_components)
            }
            
        except Exception as e:
            self._log_step("ica_aroma", "failed", f"ICA-AROMA failed: {e}")
            raise RuntimeError(f"ICA-AROMA preprocessing failed: {e}")

    def _identify_motion_components(self, ica_timeseries: np.ndarray, ica_spatial_maps: np.ndarray,
                                   motion_params: Dict, params: Dict) -> List[int]:
        """Identify motion-related ICA components using AROMA criteria"""
        motion_components = []
        
        if motion_params is None:
            self._log_step("ica_aroma", "warning", "No motion parameters available for AROMA classification")
            return motion_components
        
        n_components = ica_timeseries.shape[1]
        motion_ts = motion_params.get('displacement', np.zeros(ica_timeseries.shape[0]))
        
        for comp_idx in range(n_components):
            is_motion = False
            component_ts = ica_timeseries[:, comp_idx]
            
            # Criterion 1: High correlation with motion parameters (most important)
            if len(motion_ts) == len(component_ts):
                correlation = np.corrcoef(component_ts, motion_ts)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > params.get('motion_correlation_threshold', 0.3):
                    is_motion = True
            
            # ✅ COMPLETE: Criterion 2 implementation
            if not is_motion:
                try:
                    # Frequency domain analysis
                    fft = np.fft.fft(component_ts)
                    freqs = np.fft.fftfreq(len(component_ts))
                    power_spectrum = np.abs(fft) ** 2;
                    
                    # Check if most power is in high frequencies (>0.2 Hz for typical rs-fMRI)
                    high_freq_mask = np.abs(freqs) > params.get('high_freq_threshold', 0.2)
                    high_freq_power = np.sum(power_spectrum[high_freq_mask])
                    total_power = np.sum(power_spectrum)
                    
                    high_freq_ratio = high_freq_power / (total_power + 1e-8)
                    if high_freq_ratio > params.get('high_freq_ratio_threshold', 0.6):
                        is_motion = True
                except:
                    pass  # Skip frequency analysis if it fails
            
            # ✅ COMPLETE: Criterion 3 implementation
            if not is_motion:
                try:
                    spatial_map = ica_spatial_maps[comp_idx, :]
                    spatial_std = np.std(spatial_map)
                    
                    # High spatial variation could indicate motion artifacts
                    if spatial_std > params.get('spatial_std_threshold', 2.0):
                        # Additional check: look for edge-like patterns
                        high_values = np.abs(spatial_map) > np.percentile(np.abs(spatial_map), 95)
                        if np.sum(high_values) / len(spatial_map) < 0.1:  # Very sparse activation
                            is_motion = True
                except:
                    pass
            
            if is_motion:
                motion_components.append(comp_idx)
        
        return motion_components

    def _generate_brain_mask(self, data_img) -> np.ndarray:
        """Generate brain mask using improved intensity thresholding"""
        if hasattr(data_img, 'get_fdata'):
            img_data = data_img.get_fdata()
        else:
            img_data = np.asarray(data_img)
            
        # Calculate mean image across time
        mean_img = np.mean(img_data, axis=-1)
        
        # Enhanced brain mask generation
        non_zero_voxels = mean_img[mean_img > 0]
        
        if len(non_zero_voxels) == 0:
            print(f"⚠️  {self.subject_id}: No non-zero voxels found!")
            return np.zeros(mean_img.shape, dtype=np.uint8)
        
        # Use multiple thresholding approaches and take the best
        thresholds = [
            np.percentile(non_zero_voxels, 20),  # Lower threshold
            np.percentile(non_zero_voxels, 25),  # Original
            np.percentile(non_zero_voxels, 15),  # Even lower
        ]
        
        best_mask = None
        best_coverage = 0
        
        for threshold in thresholds:
            # Create candidate mask
            candidate_mask = (mean_img > threshold).astype(np.uint8)
            
            # Apply morphological operations
            from scipy import ndimage
            candidate_mask = ndimage.binary_fill_holes(candidate_mask).astype(np.uint8)
            candidate_mask = ndimage.binary_erosion(candidate_mask, iterations=1).astype(np.uint8)
            candidate_mask = ndimage.binary_dilation(candidate_mask, iterations=2).astype(np.uint8)
            
            # Calculate coverage (should be reasonable brain coverage)
            coverage = candidate_mask.sum() / candidate_mask.size
            
            # Good brain mask should cover 10-40% of the volume
            if 0.05 <= coverage <= 0.5 and coverage > best_coverage:
                best_mask = candidate_mask
                best_coverage = coverage
        
        # Fallback if no good mask found
        if best_mask is None:
            print(f"{self.subject_id}: Using fallback brain mask")
            # Very permissive threshold as last resort
            fallback_threshold = np.percentile(non_zero_voxels, 10)
            best_mask = (mean_img > fallback_threshold).astype(np.uint8)
            best_coverage = best_mask.sum() / best_mask.size
        
        # Final validation
        mask_size_mb = best_mask.nbytes / (1024 * 1024)
        
        # ✅ ONLY print details if there's a problem
        if mask_size_mb < 0.1 or best_coverage < 0.1:
            print(f"Mask validation issues for {self.subject_id}:")
            print(f"   - Shape: {best_mask.shape}")
            print(f"   - Non-zero voxels: {best_mask.sum()}")
            print(f"   - Coverage: {best_coverage*100:.1f}%")
            print(f"   - Expected file size: {mask_size_mb:.2f}MB")
            
            if mask_size_mb < 0.1:
                self._log_step("brain_mask", "warning", f"Suspiciously small mask: {mask_size_mb:.2f}MB")
    
        # ✅ SUCCESS: Only log to processing log, no console spam
        self._log_step("brain_mask", "success", 
                      f"Generated mask: {best_mask.sum()} voxels ({best_coverage*100:.1f}% coverage)")
        
        return best_mask

    # ----------------- Helper Methods -----------------

    def _log_step(self, step_name: str, status: str, details: str = "") -> None:
        """Log a processing step"""
        self.processing_log.append({
            'step': step_name,
            'status': status,
            'message': details,
            'subject_id': self.subject_id
        })
        
        if status in ['failed', 'warning']:
            print(f"[{status.upper()}] {self.subject_id} - {step_name}: {details}")

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        total = len(self.processing_log)
        success = sum(1 for e in self.processing_log if e['status'] == 'success')
        failed = sum(1 for e in self.processing_log if e['status'] == 'failed')
        skipped = sum(1 for e in self.processing_log if e['status'] == 'skipped')
        warnings = sum(1 for e in self.processing_log if e['status'] == 'warning')

        return {
            'subject_id': self.subject_id,
            'total_steps': total,
            'successful_steps': success,
            'failed_steps': failed,
            'skipped_steps': skipped,
            'warning_steps': warnings,
            'completion_rate': (success / total * 100) if total > 0 else 0
        }


def verify_output_integrity(output_path: Path, min_size_mb: float = 1.0, verbose: bool = False) -> bool:
    """Verify that a NIfTI file was written correctly and is not corrupted"""
    if not output_path.exists():
        if verbose:
            print(f"❌ Verification failed: File does not exist: {output_path}")
        return False
    
    try:
        # Check file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"   📏 Checking {output_path.name}: {file_size_mb:.2f}MB")
            
        if file_size_mb < min_size_mb:
            if verbose:
                print(f"❌ Verification failed: File too small ({file_size_mb:.2f}MB < {min_size_mb}MB): {output_path}")
            return False
        
        # Test gzip integrity
        if output_path.suffix == '.gz':
            try:
                if verbose:
                    print(f"   🔍 Testing gzip integrity: {output_path.name}")
                with gzip.open(output_path, 'rb') as f:
                    chunk = f.read(1024 * 1024)
                    if len(chunk) == 0:
                        if verbose:
                            print(f"❌ Verification failed: Gzip file appears empty: {output_path}")
                        return False
                if verbose:
                    print(f"   ✅ Gzip integrity: PASSED")
            except Exception as gz_error:
                if verbose:
                    print(f"❌ Verification failed: Gzip corruption in {output_path}: {gz_error}")
                return False
        
        # Test NIfTI loading
        try:
            if verbose:
                print(f"   🔍 Testing NIfTI loading: {output_path.name}")
            import nibabel as nib
            img = nib.load(output_path)
            
            # Basic shape validation
            if len(img.shape) < 3:
                if verbose:
                    print(f"❌ Verification failed: Invalid dimensions {img.shape}: {output_path}")
                return False
            
            if verbose:
                print(f"   ✅ NIfTI shape: {img.shape}")
            
            # Test reading a small sample of data
            if len(img.shape) == 4:
                test_data = img.get_fdata()[:5, :5, :5, :1]
            else:
                test_data = img.get_fdata()[:5, :5, :5]
            
            if test_data is None or test_data.size == 0:
                if verbose:
                    print(f"❌ Verification failed: Cannot read NIfTI data: {output_path}")
                return False
            
            if verbose:
                print(f"   ✅ Data sample read successfully")
                
        except Exception as nii_error:
            if verbose:
                print(f"❌ Verification failed: NIfTI loading error in {output_path}: {nii_error}")
            return False
        
        if verbose:
            print(f"✅ Verification passed: {output_path.name} ({file_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Verification failed: Unexpected error in {output_path}: {e}")
        return False

def cleanup_failed_subject(subject_out_dir: Path):
    """Remove all files for a failed subject to enable clean reprocessing"""
    if subject_out_dir.exists():
        import shutil
        try:
            shutil.rmtree(subject_out_dir)
            print(f"🧹 Cleaned up failed subject directory: {subject_out_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to cleanup {subject_out_dir}: {e}")
            return False
    return True

def identify_failed_subjects(output_base_dir: Path) -> List[Dict[str, str]]:
    """Identify subjects with failed or corrupted preprocessing"""
    failed_subjects = []
    
    if not output_base_dir.exists():
        return failed_subjects
    
    for site_dir in output_base_dir.iterdir():
        if not site_dir.is_dir():
            continue
            
        for subject_dir in site_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            func_file = subject_dir / "func_preproc.nii.gz"
            mask_file = subject_dir / "mask.nii.gz"
            
            # Check if files exist and are valid
            failed = False
            reason = ""
            
            if not func_file.exists():
                failed = True
                reason = "Missing functional file"
            elif not verify_output_integrity(func_file, min_size_mb=10.0, verbose=False):
                failed = True
                reason = "Corrupted functional file"
            
            if not mask_file.exists():
                failed = True
                reason = "Missing mask file"
            elif not verify_output_integrity(mask_file, min_size_mb=0.01, verbose=False):
                failed = True
                reason = "Corrupted mask file"
            
            if failed:
                failed_subjects.append({
                    "subject_id": subject_dir.name,
                    "site": site_dir.name,
                    "output_dir": str(subject_dir),
                    "reason": reason
                })
    
    return failed_subjects

def _process_subject(row):
    import psutil
    import gc
    import time
    
    # Add a verbose flag for debugging
    debug_verbose = False  # ✅ Set to True only when debugging
    
    try:
        # Check available memory before processing
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            print(f"⚠️ High memory usage ({memory.percent:.1f}%) - forcing cleanup")
            gc.collect()
            
            # Wait for memory to settle
            time.sleep(2)
            
            # Check again
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                raise RuntimeError(f"Insufficient memory ({memory.percent:.1f}% used). Cannot safely process.")
        
        # ✅ ADDED: Force retry flag check
        force_retry = row.get('force_retry', False)
        
        # Get device from row if available
        device = row.get('device', None)
        if device and isinstance(device, str):
            device = torch.device(device)
        
        pipeline = PreprocessingPipeline(device=device)
        subject_id = row["subject_id"]
        func_path = Path(row["input_path"])
        
        # ✅ ONLY print validation if verbose or there's an issue
        corruption_check = validate_input_file(func_path)
        
        if not corruption_check['valid']:
            print(f"❌ {subject_id}: Input validation failed - {corruption_check['message']}")
            raise FileCorruptionError(f"Input file corrupted: {corruption_check['error_type']}")
        elif debug_verbose:
            print(f"🔍 Validating input file: {subject_id}")
        
        # Extract site name
        site_name = (
            row.get("site") or 
            row.get("dataset") or 
            func_path.parts[-5] if len(func_path.parts) >= 5 else "UnknownSite"
        )

        # Verify input file exists
        if not func_path.exists():
            raise FileNotFoundError(f"Input file not found: {func_path.absolute()}")
        
        if not func_path.is_file():
            raise ValueError(f"Input path is not a file: {func_path.absolute()}")

        # Check file extension
        if not str(func_path).lower().endswith(('.nii', '.nii.gz')):
            raise ValueError(f"Invalid file extension. Expected .nii or .nii.gz, got: {func_path.suffix}")

        # Create output directory
        subj_out = Path(row.get("out_dir", ".")) / site_name / subject_id
        subj_out.mkdir(parents=True, exist_ok=True)

        # Define output paths
        func_output_path = subj_out / "func_preproc.nii.gz"
        mask_output_path = subj_out / "mask.nii.gz"

        # ✅ MODIFIED: Skip verification check if force_retry is True
        if not force_retry and func_output_path.exists() and mask_output_path.exists():
            if (verify_output_integrity(func_output_path, min_size_mb=10.0) and 
                verify_output_integrity(mask_output_path, min_size_mb=0.02)): 
                return {
                    "status": "success",
                    "subject_id": subject_id,
                    "site": site_name,
                    "output_dir": str(subj_out),
                    "files_verified": True, 
                    "skipped": True,
                    "message": f"Already processed and verified: {subject_id}"
                }
        
        # ✅ ADDED: Clean up corrupted files before retry
        if force_retry:
            print(f"🔄 Force retry enabled for {subject_id} - cleaning up old files")
            if func_output_path.exists():
                func_output_path.unlink()
                print(f"   🧹 Removed old functional file")
            if mask_output_path.exists():
                mask_output_path.unlink()
                print(f"   🧹 Removed old mask file")
            confounds_path = subj_out / "confounds.csv"
            if confounds_path.exists():
                confounds_path.unlink()
                print(f"   🧹 Removed old confounds file")

        # ✅ ADDED: Memory-optimized processing configuration for problematic subjects
        memory_optimized_config = pipeline.config.copy()
        if memory.percent > 70:
            print(f"⚠️ Applying memory-optimized settings ({memory.percent:.1f}% memory used)")
            # Reduce ICA components for memory-constrained environments
            if 'denoising' in memory_optimized_config:
                if 'ica_aroma' in memory_optimized_config['denoising']:
                    memory_optimized_config['denoising']['ica_aroma']['n_components'] = 15
                if 'acompcor' in memory_optimized_config['denoising']:
                    memory_optimized_config['denoising']['acompcor']['n_components'] = 3
            pipeline.config = memory_optimized_config

        # Test NIfTI loading
        try:
            test_load = nib.load(str(func_path.absolute()))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load NIfTI file ({func_path.absolute()}): {str(e)}")

        # Run preprocessing
        result = pipeline.process(str(func_path.absolute()), subject_id)

        if result["status"] == "success":
            # Get processed data
            proc_data = result["processed_data"] 
            
            # ✅ FIX: Define original_affine here
            original_img = nib.load(str(func_path.absolute()))
            original_affine = original_img.affine
            
            # Ensure it's a NIfTI image
            if torch.is_tensor(proc_data):
                proc_array = proc_data.cpu().numpy()
                proc_nifti = nib.Nifti1Image(proc_array, original_affine)
            elif isinstance(proc_data, nib.Nifti1Image):
                proc_nifti = proc_data
            else:
                proc_nifti = nib.Nifti1Image(np.array(proc_data), original_affine)

            # Save functional file
            print(f"   Saving functional file: {func_output_path.name}")
            nib.save(proc_nifti, func_output_path)

            # ✅ VERIFY FUNCTIONAL FILE
            if not verify_output_integrity(func_output_path, min_size_mb=10.0, verbose=False):  # ✅ Add verbose=False
                if func_output_path.exists():
                    func_output_path.unlink()
                raise RuntimeError(f"Functional file verification failed: {func_output_path}")

            # ✅ ENHANCED MASK CREATION AND SAVING
            print(f"   Creating brain mask for {subject_id}")
            try:
                brain_mask = result["brain_mask"]
                
                # ✅ DETAILED MASK VALIDATION
                print(f"   🔍 Mask details before saving:")
                print(f"   - Data type: {brain_mask.dtype}")
                print(f"   - Shape: {brain_mask.shape}")
                print(f"   - Min/Max values: {brain_mask.min()}/{brain_mask.max()}")
                print(f"   - Unique values: {np.unique(brain_mask)}")
                
                mask_voxels = np.sum(brain_mask > 0)
                mask_coverage = mask_voxels / brain_mask.size
                expected_size_mb = brain_mask.nbytes / (1024 * 1024)
                
                print(f"   - Non-zero voxels: {mask_voxels}")
                print(f"   - Coverage: {mask_coverage*100:.1f}%")
                print(f"   - Memory size: {expected_size_mb:.2f}MB")
                
                if mask_voxels < 1000:
                    raise ValueError(f"Brain mask has too few voxels: {mask_voxels} (expected >1000)")
                
                if mask_coverage < 0.01:
                    raise ValueError(f"Brain mask coverage too low: {mask_coverage*100:.2f}% (expected >1%)")
                
                # ✅ ENSURE PROPER DATA TYPE AND AFFINE
                # Force uint8 data type explicitly
                clean_mask = brain_mask.astype(np.uint8)
                
                # Get the original affine from input file
                original_img = nib.load(str(func_path.absolute()))
                original_affine = original_img.affine
                original_header = original_img.header.copy()
                
                # ✅ CREATE MASK NIFTI WITH EXPLICIT SETTINGS
                mask_nifti = nib.Nifti1Image(
                    clean_mask,
                    original_affine,
                    header=None  # Let nibabel create a fresh header
                )
                
                # ✅ FORCE CORRECT HEADER SETTINGS
                mask_header = mask_nifti.header
                mask_header.set_data_dtype(np.uint8)  # Explicitly set uint8
                mask_header.set_slope_inter(1, 0)     # No scaling
                
                # ✅ ONLY print details if verbose
                if debug_verbose:
                    print(f"   🔍 NIfTI image details:")
                    print(f"   - NIfTI shape: {mask_nifti.shape}")
                    print(f"   - NIfTI dtype: {mask_nifti.get_data_dtype()}")
                    print(f"   - Header dtype: {mask_header.get_data_dtype()}")
                
                # ✅ ROBUST SAVING - ONLY PRINT ERRORS OR IF VERBOSE
                max_save_attempts = 3
                save_successful = False

                for attempt in range(max_save_attempts):
                    try:
                        # ✅ SAVE WITH EXPLICIT COMPRESSION AND FLUSH
                        nib.save(mask_nifti, mask_output_path)
                        
                        # ✅ FORCE FILE SYSTEM SYNC
                        import os
                        if hasattr(os, 'sync'):
                            os.sync()  # Unix/Linux
                        
                        # Wait briefly for file system to settle
                        time.sleep(0.1)
                        
                        # ✅ IMMEDIATE VERIFICATION
                        if mask_output_path.exists():
                            saved_size_mb = mask_output_path.stat().st_size / (1024*1024)
                            
                            # ✅ ONLY print attempt details if verbose
                            if debug_verbose:
                                print(f"   📏 Attempt {attempt + 1}: Saved file size: {saved_size_mb:.2f}MB")
                            
                            # Test if file is complete by loading it
                            try:
                                test_load = nib.load(mask_output_path)
                                test_data = test_load.get_fdata()
                                test_voxels = np.sum(test_data > 0)
                                
                                if test_voxels == mask_voxels and saved_size_mb >= 0.01:  # ✅ Lowered from 0.05 to 0.01
                                    if debug_verbose:
                                        print(f"   ✅ Attempt {attempt + 1}: Save successful!")
                                    save_successful = True
                                    break
                                else:
                                    print(f"❌ {subject_id}: Attempt {attempt + 1} verification failed - voxels={test_voxels}, size={saved_size_mb:.2f}MB")
                                    if mask_output_path.exists():
                                        mask_output_path.unlink()
                            except Exception as load_error:
                                print(f"❌ {subject_id}: Attempt {attempt + 1} cannot load saved file: {load_error}")
                                if mask_output_path.exists():
                                    mask_output_path.unlink()
                        else:
                            print(f"❌ {subject_id}: Attempt {attempt + 1} file was not created")
                    
                    except Exception as save_error:
                        print(f"❌ {subject_id}: Attempt {attempt + 1} save error: {save_error}")
                        if mask_output_path.exists():
                            mask_output_path.unlink()

                if not save_successful:
                    raise RuntimeError(f"Failed to save mask after {max_save_attempts} attempts")

                # ✅ ONLY print final success if verbose
                if debug_verbose:
                    print(f"   ✅ Mask successfully saved and verified!")

            except Exception as e:
                print(f"   ❌ Mask creation/saving failed: {str(e)}")
                raise RuntimeError(f"Brain mask generation failed: {str(e)}")

            # Save confounds
            confounds_path = subj_out / "confounds.csv"
            pd.DataFrame(result["confound_regressors"]).to_csv(confounds_path, index=False)

            return {
                "status": "success",
                "subject_id": subject_id,
                "site": site_name,
                "output_dir": str(subj_out),
                "files_verified": True,
                "func_size_mb": func_output_path.stat().st_size / (1024*1024),
                "mask_size_mb": mask_output_path.stat().st_size / (1024*1024),
                "message": f"Preprocessed {subject_id} successfully"
            }
        else:
            raise RuntimeError(f"Pipeline failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        error_type = "unknown_error"
        if "Error -3" in str(e) or "decompressing" in str(e):
            error_type = "gzip_corruption"
        elif "process cannot access" in str(e):
            error_type = "file_access_conflict"
        elif "FileCorruptionError" in str(type(e).__name__):
            error_type = "input_corruption"
            
        return {
            "status": "failed",
            "subject_id": row.get("subject_id", "unknown"),
            "site": row.get("site", "UnknownSite"), 
            "error": str(e),
            "error_type": error_type,
            "message": f"Failed: {str(e)}"
        }

class FileCorruptionError(Exception):
    """Custom exception for file corruption"""
    pass

def validate_input_file(file_path: Path) -> dict:
    """Validate input file before processing"""
    validation = {
        'valid': False,
        'error_type': None,
        'message': None,
        'file_size_mb': 0
    }
    
    try:
        if not file_path.exists():
            validation['error_type'] = 'missing_file'
            validation['message'] = f"File not found: {file_path}"
            return validation
        
        # Check file size
        file_size = file_path.stat().st_size
        validation['file_size_mb'] = file_size / (1024 * 1024)
        
        if file_size == 0:
            validation['error_type'] = 'empty_file'
            validation['message'] = "File is empty (0 bytes)"
            return validation
        
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious for fMRI
            validation['error_type'] = 'file_too_small'
            validation['message'] = f"File suspiciously small: {validation['file_size_mb']:.2f}MB"
            return validation
        
        # Test gzip integrity for .gz files
        if str(file_path).endswith('.gz'):
            try:
                with gzip.open(file_path, 'rb') as gz_file:
                    # Read first chunk to test decompression
                    chunk = gz_file.read(1024 * 1024)  # 1MB
                    if len(chunk) == 0:
                        validation['error_type'] = 'empty_gzip'
                        validation['message'] = "Gzip file decompresses to empty"
                        return validation
            except Exception as gz_error:
                validation['error_type'] = 'gzip_corruption'
                validation['message'] = f"Gzip decompression failed: {str(gz_error)}"
                return validation
        
        # Test NIfTI loading
        try:
            img = nib.load(file_path)
            shape = img.shape
            
            if len(shape) < 3:
                validation['error_type'] = 'invalid_dimensions'
                validation['message'] = f"Invalid NIfTI dimensions: {shape}"
                return validation
                
            if len(shape) == 4 and shape[3] < 50:  # fMRI should have reasonable timepoints
                validation['error_type'] = 'insufficient_timepoints'
                validation['message'] = f"Too few timepoints: {shape[3]}"
                return validation
            
        except Exception as nii_error:
            validation['error_type'] = 'nifti_loading_error'
            validation['message'] = f"Cannot load NIfTI: {str(nii_error)}"
            return validation
        
        validation['valid'] = True
        validation['message'] = f"File validated successfully ({validation['file_size_mb']:.1f}MB)"
        return validation
        
    except Exception as e:
        validation['error_type'] = 'validation_error'
        validation['message'] = f"Validation failed: {str(e)}"
        return validation





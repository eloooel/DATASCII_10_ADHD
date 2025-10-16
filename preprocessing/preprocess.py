import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import argparse
import pandas as pd

from utils import run_parallel
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from scipy import signal, ndimage
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA
import warnings

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
                'max_displacement_mm': 3.0  # Exclusion threshold
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
                'resolution': 2.0,
                'smooth_fwhm': 6.0  # Spatial smoothing after normalization
            },
            'temporal_filtering': {
                'enabled': True,
                'low_freq': 0.009,  # Hz
                'high_freq': 0.08,  # Hz
                'filter_type': 'butterworth',
                'filter_order': 4
            },
            'denoising': {
                'motion_regression': True,
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
            
            # Pipeline stages
            data_img, motion_params = self._motion_correction(data_img, self.config['motion_correction'])
            data_img = self._slice_timing_correction(data_img, self.config['slice_timing_correction'], metadata)
            data_img = self._spatial_normalization(data_img, self.config['spatial_normalization'])
            data_img = self._temporal_filtering(data_img, self.config['temporal_filtering'], metadata)
            data_img, confound_regressors = self._denoising(data_img, motion_params, self.config['denoising'])

            # Generate brain mask
            brain_mask = self._generate_brain_mask(data_img)

            return {
                'processed_data': data_img,  # Now guaranteed to be NIfTI
                'confound_regressors': confound_regressors,
                'brain_mask': brain_mask,
                'motion_parameters': motion_params,
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
        img_data = torch.from_numpy(data.get_fdata()).to(self.device)
        
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

        # Get data and metadata
        if isinstance(data_img, nib.Nifti1Image):
            img_data = data_img.get_fdata()
            affine = data_img.affine
            header = data_img.header
        else:
            img_data = data_img.cpu().numpy() if torch.is_tensor(data_img) else np.array(data_img)
            affine = self.metadata['affine']
            header = None

        # Convert PyTorch tensor to numpy if needed
        if torch.is_tensor(data_img):
            img_data = data_img.cpu().numpy()
        else:
            img_data = data_img.get_fdata()

        n_vols = img_data.shape[-1]
        
        # Create reference volume (mean or first volume)
        if params['reference'] == 'mean':
            reference = np.mean(img_data, axis=-1)
        elif params['reference'] == 'first':
            reference = img_data[..., 0]
        else:
            # Assume reference is a volume index
            ref_idx = int(params['reference'])
            reference = img_data[..., ref_idx]

        # Initialize motion parameters
        motion_params = {
            'translations': np.zeros((n_vols, 3)),  # x, y, z translations
            'rotations': np.zeros((n_vols, 3)),     # roll, pitch, yaw rotations
            'displacement': np.zeros(n_vols),       # frame displacement
            'excluded_volumes': []
        }

        corrected_data = np.zeros_like(img_data)

        # Process each volume
        for vol_idx in range(n_vols):
            current_vol = img_data[..., vol_idx]
            # Ensure current_vol is a numpy array
            if torch.is_tensor(current_vol):
                current_vol = current_vol.cpu().numpy()
            translation = self._estimate_translation(reference, current_vol)
            # Ensure translation is a finite numpy array
            translation = np.asarray(translation, dtype=np.float32)
            if not np.all(np.isfinite(translation)):
                translation = np.zeros_like(translation)
            # Now apply shift
            corrected_vol = ndimage.shift(current_vol, translation, order=1, mode='nearest')
            corrected_data[..., vol_idx] = corrected_vol
            
            # Store motion parameters
            motion_params['translations'][vol_idx] = translation
            
            # Calculate frame displacement
            if vol_idx > 0:
                prev_trans = motion_params['translations'][vol_idx - 1]
                displacement = np.sqrt(np.sum((translation - prev_trans) ** 2))
                motion_params['displacement'][vol_idx] = displacement
                
                # Check for excessive motion
                if displacement > params.get('max_displacement_mm', 3.0):
                    motion_params['excluded_volumes'].append(vol_idx)

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
            corrected_data = torch.from_numpy(corrected_data).to(self.device)
            return corrected_data, motion_params
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
                    # Ensure numeric array
                    voxel_ts = np.asarray(slice_data[x, y, :], dtype=np.float32)
                    
                    # Skip empty voxels safely
                    if voxel_ts.size == 0:
                        corrected_data[x, y, z, :] = voxel_ts
                        continue

                    # Skip zero-background voxels
                    if not np.any(voxel_ts > 0):
                        corrected_data[x, y, z, :] = voxel_ts
                        continue

                    # Linear interpolation
                    corrected_ts = np.interp(original_times, shifted_times, voxel_ts)
                    corrected_data[x, y, z, :] = corrected_ts
                    
        # Create corrected nibabel image
        corrected_img = nib.Nifti1Image(corrected_data, data_img.affine, data_img.header)
        
        self._log_step("slice_timing_correction", "success", 
                      f"Applied to {n_slices} slices, TR={tr:.2f}s, ref_time={ref_time:.3f}s")
        
        return corrected_img

    def _spatial_normalization(self, data_img, params: Dict[str, Any]):
        """Spatial normalization with smoothing"""
        if not params.get('enabled', True):
            self._log_step("spatial_normalization", "skipped", "Disabled in config")
            return data_img

        img_data = data_img.get_fdata()
        
        # Apply spatial smoothing (in lieu of full registration to MNI)
        # This is a simplified version - real implementation would include registration
        smooth_fwhm = params.get('smooth_fwhm', 6.0)
        if smooth_fwhm > 0:
            # Convert FWHM to sigma for Gaussian filter
            sigma = smooth_fwhm / (2 * np.sqrt(2 * np.log(2)))
            
            # Get voxel sizes for anisotropic smoothing
            voxel_sizes = data_img.header.get_zooms()[:3]
            sigma_voxels = [sigma / vox_size for vox_size in voxel_sizes]
            
            smoothed_data = np.zeros_like(img_data)
            
            # Smooth each volume
            for t in range(img_data.shape[-1]):
                smoothed_data[..., t] = ndimage.gaussian_filter(
                    img_data[..., t], sigma=sigma_voxels, mode='nearest'
                )
            
            # Create smoothed nibabel image
            normalized_img = nib.Nifti1Image(smoothed_data, data_img.affine, data_img.header)
            
            self._log_step("spatial_normalization", "success", 
                          f"Applied {smooth_fwhm}mm FWHM smoothing")
        else:
            normalized_img = data_img
            self._log_step("spatial_normalization", "success", "No smoothing applied")

        return normalized_img

    def _temporal_filtering(self, data_img, params: Dict[str, Any], metadata: Dict[str, Any]):
        """Real temporal bandpass filtering with CUDA support"""
        if not params.get('enabled', True):
            self._log_step("temporal_filtering", "skipped", "Disabled in config")
            return data_img

        # Get data and affine/header
        if isinstance(data_img, nib.Nifti1Image):
            img_data = torch.from_numpy(data_img.get_fdata()).to(self.device)
            affine = data_img.affine
            header = data_img.header
        else:
            img_data = data_img if torch.is_tensor(data_img) else torch.from_numpy(data_img).to(self.device)
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
            b_torch = torch.from_numpy(b).to(self.device)
            a_torch = torch.from_numpy(a).to(self.device)
        except ValueError:
            # Fallback to high-pass only if bandpass fails
            b, a = signal.butter(filter_order, low, btype='high')
            b_torch = torch.from_numpy(b).to(self.device)
            a_torch = torch.from_numpy(a).to(self.device)
            self._log_step("temporal_filtering", "warning", "Applied high-pass only")

        # Get brain mask for filtering (avoid filtering background)
        mean_img = torch.mean(img_data, dim=-1)
        brain_mask = mean_img > torch.quantile(mean_img[mean_img > 0], 0.25)
        
        # Apply filter to each voxel on GPU
        filtered_data = img_data.clone()
        brain_voxels = torch.where(brain_mask)
        total_voxels = len(brain_voxels[0])
        
        for i in range(total_voxels):
            x, y, z = brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]
            voxel_ts = filtered_data[x, y, z, :]
            
            if torch.any(voxel_ts != 0):
                try:
                    # Detrend
                    t = torch.arange(voxel_ts.shape[0], dtype=torch.float32, device=self.device)
                    p = torch.polynomial.polynomial.polyfit(t, voxel_ts, deg=1)
                    trend = p[0] + p[1] * t
                    detrended = voxel_ts - trend

                    # FFT-based filtering
                    fft = torch.fft.rfft(detrended)
                    freqs = torch.fft.rfftfreq(voxel_ts.shape[0], d=tr)
                    mask = (freqs >= low_freq) & (freqs <= high_freq)
                    fft = fft * mask.to(self.device)
                    filtered = torch.fft.irfft(fft, n=voxel_ts.shape[0])
                    
                    # Add trend back
                    filtered_data[x, y, z, :] = filtered + trend
                except:
                    # Keep original if filtering fails
                    continue

        # Always return NIfTI
        filtered_array = filtered_data.cpu().numpy()
        filtered_img = nib.Nifti1Image(filtered_array, affine, header)
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

    def _generate_brain_mask(self, data_img) -> np.ndarray:
        """Generate brain mask using intensity thresholding"""
        if hasattr(data_img, 'get_fdata'):
            img_data = data_img.get_fdata()
        else:
            img_data = np.asarray(data_img)
            
        mean_img = np.mean(img_data, axis=-1)
        
        # Otsu-like thresholding
        non_zero_voxels = mean_img[mean_img > 0]
        
        if len(non_zero_voxels) == 0:
            return np.zeros(mean_img.shape, dtype=np.uint8)
        
        threshold = np.percentile(non_zero_voxels, 25)
        
        # Initial mask
        mask = (mean_img > threshold).astype(np.uint8)
        
        # Morphological operations to clean up mask
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        mask = ndimage.binary_erosion(mask, iterations=1).astype(np.uint8)
        mask = ndimage.binary_dilation(mask, iterations=2).astype(np.uint8)
        
        self._log_step("brain_mask", "success", 
                      f"Generated mask with {mask.sum()} voxels ({mask.sum()/mask.size*100:.1f}% of volume)")
        
        return mask

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

# def run_batch_cli():
#     import argparse
#     import nibabel as nib
#     import numpy as np

#     parser = argparse.ArgumentParser(description="Batch rs-fMRI Preprocessing")
#     parser.add_argument("--metadata", type=str, required=True,
#                         help="CSV file with columns: subject_id, input_path")
#     parser.add_argument("--out-dir", type=str, required=True,
#                         help="Directory to store preprocessed outputs")
#     parser.add_argument("--workers", type=int, default=None,
#                         help="Number of parallel workers (default = all cores)")
#     args = parser.parse_args()

#     metadata = pd.read_csv(args.metadata)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Add output dir to each row for worker
#     metadata["out_dir"] = str(out_dir)
#     results = []

#     # Run parallel processing using run_parallel
#     # Main process manages the progress bar
#     with tqdm(total=len(metadata), desc="Preprocessing subjects") as pbar:

#         def worker_wrapper(row):
#             res = _process_subject(row, pbar=pbar)
#             pbar.update(1)
#             tqdm.write(res["message"])
#             return res

#         results = run_parallel(
#             tasks=metadata.to_dict("records"),
#             worker_fn=worker_wrapper,
#             max_workers=args.workers,
#             desc=None
#         )

#     # Summary
#     success = sum(1 for r in results if r["status"] == "success")
#     failed = len(results) - success
#     print(f"\nFinished preprocessing. Success: {success}, Failed: {failed}")

def _process_subject(row, pbar=None):
    """Process a single subject; returns result dict, no printing inside."""
    try:
        # Get device from row if available, otherwise use default
        device = row.get('device', None)
        if device and isinstance(device, str):
            device = torch.device(device)
        
        pipeline = PreprocessingPipeline(device=device)
        subject_id = row["subject_id"]
        func_path = Path(row["input_path"])  # Convert to Path object

        # Extract site from input path - will get "OHSU" from the path structure
        site_name = func_path.parts[-5] if len(func_path.parts) >= 5 else row.get("site", "UnknownSite")

        if pbar is not None:
            pbar.set_postfix_str(f"Subject: {site_name} {subject_id}")
            pbar.refresh()  

        # Verify file exists and is readable
        if not func_path.exists():
            raise FileNotFoundError(f"Input file not found: {func_path.absolute()}")
        if not func_path.is_file():
            raise ValueError(f"Input path is not a file: {func_path.absolute()}")
        

        # Check file extension
        if not str(func_path).lower().endswith(('.nii', '.nii.gz')):
            raise ValueError(f"Invalid file extension. Expected .nii or .nii.gz, got: {func_path.suffix}")

        # Create output directory before processing
        subj_out = Path(row.get("out_dir", ".")) / site_name / subject_id
        subj_out.mkdir(parents=True, exist_ok=True)

        # Try to load the NIfTI file with explicit error handling
        try:
            test_load = nib.load(str(func_path.absolute()))
        except Exception as e:
            print(f"[ERROR] Failed to load NIfTI file: {func_path.absolute()}")
            print(f"Reason: {str(e)}")
            raise ValueError(f"Failed to load NIfTI file ({func_path.absolute()}): {str(e)}")

        # Run preprocessing
        result = pipeline.process(str(func_path.absolute()), subject_id)

        if result["status"] == "success":
            # Ensure processed_data is a Nifti1Image
            proc_data = result["processed_data"]
            if torch.is_tensor(proc_data):
                proc_array = proc_data.cpu().numpy()
                proc_nifti = nib.Nifti1Image(proc_array, np.eye(4))  # Use identity affine if unknown
            elif isinstance(proc_data, nib.Nifti1Image):
                proc_nifti = proc_data
            else:
                # fallback
                proc_nifti = nib.Nifti1Image(np.array(proc_data), np.eye(4))

            # Save preprocessed functional image
            nib.save(proc_nifti, subj_out / "func_preproc.nii.gz")

            # Save brain mask using the same affine as proc_nifti
            mask_nifti = nib.Nifti1Image(result["brain_mask"].astype(np.uint8),
                                        proc_nifti.affine)
            nib.save(mask_nifti, subj_out / "mask.nii.gz")

            # Save confound regressors
            pd.DataFrame(result["confound_regressors"]).to_csv(
                subj_out / "confounds.csv", index=False
            )

            return {
                "status": "success",
                "subject_id": subject_id,
                "site": site_name,
                "message": f"Preprocessed {subject_id} (Site: {site_name})"
            }
        else:
            raise RuntimeError(f"Pipeline failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        import traceback
        print(f"\nError processing subject {row.get('subject_id', 'unknown')}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return {
            "status": "failed",
            "subject_id": row.get("subject_id", "unknown"),
            "site": row.get("site", "UnknownSite"),
            "error": str(e),
            "message": f"Failed: {str(e)}"
        }

# if __name__ == "__main__":
#     run_batch_cli()
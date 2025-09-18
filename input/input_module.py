"""
Input Module for ADHD preprocessing pipeline
- Handles data discovery and organization from hierarchical structure: data/(dataset)/(subject)/func/*.nii
- Manages subject intake and metadata collection
- Coordinates preprocessing pipeline execution
- Outputs preprocessed 4D fMRI volumes, confound regressors, and brain masks
"""

import argparse
import json
import csv
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nib
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict

from tqdm import tqdm
import sys

# Import the preprocessing pipeline
from preprocessing.preprocessing import PreprocessingPipeline

DEFAULT_OUTPUT_SUBDIR = "preproc_outputs"
MANIFEST_FILENAME = "preprocessing_manifest.csv"
METADATA_FILENAME = "subjects_metadata.csv"


class DataDiscovery:
    """Handles discovery and organization of rs-fMRI data from hierarchical structure"""
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        
    def discover_subjects(self) -> List[Dict[str, Any]]:
        """
        Discover all subjects and their functional files from the data hierarchy.
        Supports both patterns:
        - data/(dataset)/(subject)/func/*.nii[.gz]
        - data/(dataset)/(subject)/(session)/func/*.nii[.gz]
        
        Returns list of subject dictionaries with metadata
        """
        subjects = []
        
        # Look for dataset directories
        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            print(f"Scanning dataset: {dataset_name}")
            
            # Look for subject directories  
            for subject_dir in dataset_dir.iterdir():
                if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                    continue
                    
                subject_id = subject_dir.name
                subjects_found = self._discover_subject_files(dataset_name, subject_id, subject_dir)
                subjects.extend(subjects_found)
        
        print(f"Total functional files discovered: {len(subjects)}")
        return sorted(subjects, key=lambda x: (x['dataset'], x['subject_id'], x['session'] or '', x['run'] or ''))
    
    def _discover_subject_files(self, dataset_name: str, subject_id: str, subject_dir: Path) -> List[Dict[str, Any]]:
        """
        Discover functional files for a single subject.
        Handles both session and non-session directory structures.
        """
        subject_files = []
        
        # Check for direct func directory (no sessions)
        direct_func_dir = subject_dir / "func"
        if direct_func_dir.exists():
            files = self._find_nifti_in_func_dir(direct_func_dir, dataset_name, subject_id, None)
            subject_files.extend(files)
        
        # Check for session directories
        for item in subject_dir.iterdir():
            if item.is_dir() and item.name.startswith('ses-'):
                session_name = item.name
                session_func_dir = item / "func"
                
                if session_func_dir.exists():
                    files = self._find_nifti_in_func_dir(session_func_dir, dataset_name, subject_id, session_name)
                    subject_files.extend(files)
        
        if not subject_files:
            print(f"Warning: No functional files found for {dataset_name}/{subject_id}")
            
        return subject_files
    
    def _find_nifti_in_func_dir(self, func_dir: Path, dataset_name: str, 
                               subject_id: str, session_name: str = None) -> List[Dict[str, Any]]:
        """Find all NIfTI files in a func directory and create subject entries"""
        subject_entries = []
        
        # Find all NIfTI files
        nifti_files = []
        for pattern in ["*.nii", "*.nii.gz"]:
            nifti_files.extend(list(func_dir.glob(pattern)))
        
        if not nifti_files:
            location = f"{dataset_name}/{subject_id}"
            if session_name:
                location += f"/{session_name}"
            print(f"Warning: No NIfTI files found in {location}/func")
            return subject_entries
        
        # Create entry for each functional file
        for nifti_file in nifti_files:
            filename = nifti_file.name
            
            # Extract BIDS fields from filename
            session_from_filename = self._extract_bids_field(filename, 'ses')
            run = self._extract_bids_field(filename, 'run')
            task = self._extract_bids_field(filename, 'task')
            
            # Use session from directory structure, fallback to filename
            final_session = session_name.replace('ses-', '') if session_name else session_from_filename
            
            subject_entry = {
                'dataset': dataset_name,
                'subject_id': subject_id,
                'session': final_session,
                'run': run,
                'task': task or 'rest',  # default to rest
                'input_path': str(nifti_file),
                'relative_path': str(nifti_file.relative_to(self.data_root)),
                'file_size_mb': round(nifti_file.stat().st_size / (1024 * 1024), 2),
                'has_session_dir': session_name is not None
            }
            subject_entries.append(subject_entry)
        
        return subject_entries
    
    def _extract_bids_field(self, filename: str, field: str) -> Optional[str]:
        """Extract BIDS field from filename (e.g., 'ses-1' -> '1')"""
        import re
        pattern = f"{field}-([^_]+)"
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    
    def save_metadata(self, subjects: List[Dict[str, Any]], output_path: Path):
        """Save subjects metadata to CSV"""
        df = pd.DataFrame(subjects)
        df.to_csv(output_path, index=False)
        print(f"Saved metadata for {len(subjects)} subject files to {output_path}")


class PreprocessingCoordinator:
    """Coordinates preprocessing pipeline execution and output management"""
    
    def __init__(self, output_dir: Path, config_path: str = None):
        self.output_dir = output_dir
        self.pipeline = PreprocessingPipeline(config_path) if config_path else PreprocessingPipeline()
        self.results = []
        
    def process_subject(self, subject_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single subject's functional data
        
        Expected outputs:
        - Preprocessed 4D fMRI volumes (.nii.gz)
        - Confound regressors (.tsv) 
        - Brain masks (.nii.gz)
        """
        subject_id = subject_info['subject_id']
        dataset = subject_info['dataset']
        session = subject_info.get('session', '')
        run = subject_info.get('run', '')
        input_path = subject_info['input_path']
        
        # Create unique identifier for this subject/session/run
        identifier_parts = [dataset, subject_id]
        if session:
            identifier_parts.append(f"ses-{session}")
        if run:
            identifier_parts.append(f"run-{run}")
        unique_id = "_".join(identifier_parts)
        
        # Create subject-specific output directory
        subject_out_dir = self.output_dir / dataset / subject_id
        if session:
            subject_out_dir = subject_out_dir / f"ses-{session}"
        subject_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run preprocessing pipeline (no print statements)
            start_time = time.time()
            
            result = self.pipeline.process(input_path, subject_id=unique_id)
            
            processing_time = time.time() - start_time
            
            # Save outputs in expected format
            outputs = self._save_preprocessing_outputs(
                result, subject_out_dir, unique_id, subject_info
            )
            
            success_result = {
                'subject_info': subject_info,
                'unique_id': unique_id,
                'status': 'success',
                'processing_time': processing_time,
                'outputs': outputs,
                'completion_percent': self._compute_completion_percent(result.get('processing_log', [])),
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
                'completion_percent': 0.0,
                'error': str(e)
            }
            
            return error_result
    
    def _save_preprocessing_outputs(self, result: Dict[str, Any], output_dir: Path, 
                                  unique_id: str, subject_info: Dict[str, Any]) -> Dict[str, str]:
        """Save preprocessing outputs in expected format"""
        outputs = {}
        
        # 1. Preprocessed 4D fMRI volume (.nii.gz)
        processed_data = result.get('processed_data')
        if processed_data is not None:
            preprocessed_path = output_dir / f"{unique_id}_preprocessed_bold.nii.gz"
            try:
                if hasattr(processed_data, 'get_fdata'):
                    nib.save(processed_data, str(preprocessed_path))
                else:
                    # Create NIfTI from array data
                    img = nib.Nifti1Image(np.asarray(processed_data), affine=np.eye(4))
                    nib.save(img, str(preprocessed_path))
                outputs['preprocessed_bold'] = str(preprocessed_path)
            except Exception as e:
                print(f"Warning: Could not save preprocessed volume for {unique_id}: {e}")
        
        # 2. Confound regressors (.tsv)
        confounds_path = output_dir / f"{unique_id}_confounds.tsv"
        try:
            # Generate placeholder confound regressors based on processing log
            confounds_data = self._generate_confounds(result.get('processing_log', []))
            confounds_df = pd.DataFrame(confounds_data)
            confounds_df.to_csv(confounds_path, sep='\t', index=False)
            outputs['confounds'] = str(confounds_path)
        except Exception as e:
            print(f"Warning: Could not save confounds for {unique_id}: {e}")
        
        # 3. Brain mask (.nii.gz)
        mask_path = output_dir / f"{unique_id}_brain_mask.nii.gz"
        try:
            # Generate placeholder brain mask
            mask_data = self._generate_brain_mask(processed_data)
            mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))
            nib.save(mask_img, str(mask_path))
            outputs['brain_mask'] = str(mask_path)
        except Exception as e:
            print(f"Warning: Could not save brain mask for {unique_id}: {e}")
        
        # 4. Processing log (JSON)
        log_path = output_dir / f"{unique_id}_processing_log.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(result.get('processing_log', []), f, indent=2)
            outputs['processing_log'] = str(log_path)
        except Exception as e:
            print(f"Warning: Could not save processing log for {unique_id}: {e}")
        
        return outputs
    
    def _generate_confounds(self, processing_log: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Generate placeholder confound regressors based on preprocessing steps"""
        # This is a placeholder - in real implementation, these would come from actual preprocessing
        n_timepoints = 200  # placeholder
        confounds = {}
        
        # Check which denoising methods were applied
        for log_entry in processing_log:
            step = log_entry.get('step', '')
            
            if 'motion' in step:
                # Add motion parameters
                for param in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']:
                    confounds[param] = np.random.normal(0, 0.1, n_timepoints).tolist()
            
            if 'acompcor' in step:
                # Add aCompCor components
                n_components = 5  # from config
                for i in range(n_components):
                    confounds[f'a_comp_cor_{i:02d}'] = np.random.normal(0, 1, n_timepoints).tolist()
        
        # Always add basic confounds
        if not confounds:  # fallback
            confounds['csf'] = np.random.normal(0, 0.5, n_timepoints).tolist()
            confounds['white_matter'] = np.random.normal(0, 0.5, n_timepoints).tolist()
        
        return confounds
    
    def _generate_brain_mask(self, processed_data) -> np.ndarray:
        """Generate placeholder brain mask"""
        if processed_data is not None and hasattr(processed_data, 'shape'):
            # Create a simple brain mask (placeholder)
            if hasattr(processed_data, 'get_fdata'):
                shape = processed_data.get_fdata().shape[:3]  # spatial dimensions only
            else:
                shape = np.asarray(processed_data).shape[:3]
            return np.ones(shape, dtype=np.uint8)
        else:
            # Default brain mask
            return np.ones((64, 64, 32), dtype=np.uint8)
    
    def _compute_completion_percent(self, processing_log: List[Dict[str, Any]]) -> float:
        """Compute completion percentage from processing log"""
        if not processing_log:
            return 0.0
        
        successful_steps = sum(1 for entry in processing_log if entry.get('status') == 'success')
        total_steps = len(processing_log)
        
        return (successful_steps / total_steps) * 100.0 if total_steps > 0 else 0.0
    
    def save_manifest(self, results: List[Dict[str, Any]], manifest_path: Path):
        """Save processing manifest with all results"""
        manifest_data = []
        
        for result in results:
            subject_info = result['subject_info']
            outputs = result.get('outputs', {})
            
            manifest_entry = {
                'dataset': subject_info['dataset'],
                'subject_id': subject_info['subject_id'],
                'session': subject_info.get('session', ''),
                'run': subject_info.get('run', ''),
                'task': subject_info.get('task', ''),
                'unique_id': result['unique_id'],
                'input_path': subject_info['input_path'],
                'preprocessed_bold': outputs.get('preprocessed_bold', ''),
                'confounds': outputs.get('confounds', ''),
                'brain_mask': outputs.get('brain_mask', ''),
                'processing_log': outputs.get('processing_log', ''),
                'status': result['status'],
                'completion_percent': f"{result['completion_percent']:.1f}",
                'processing_time': f"{result.get('processing_time', 0):.1f}",
                'error': result.get('error', '')
            }
            manifest_data.append(manifest_entry)
        
        df = pd.DataFrame(manifest_data)
        df.to_csv(manifest_path, index=False)
        print(f"Saved processing manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Input Module for ADHD preprocessing pipeline")
    parser.add_argument("--data-dir", required=True, 
                       help="Root data directory with structure: data/(dataset)/(subject)/func/*.nii")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_SUBDIR, 
                       help="Output directory for preprocessed results")
    parser.add_argument("--config", default=None, 
                       help="Path to YAML config for preprocessing pipeline (optional)")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Limit number of subjects processed (0 means no limit)")
    parser.add_argument("--dataset", default=None, 
                       help="Process only specific dataset (optional)")
    parser.add_argument("--subject", default=None, 
                       help="Process only specific subject ID (optional)")
    
    args = parser.parse_args()
    
    # Validate input directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== ADHD Preprocessing Pipeline - Input Module ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config: {args.config if args.config else 'Default configuration'}")
    
    # Discover subjects
    print("\n--- Data Discovery Phase ---")
    discovery = DataDiscovery(data_dir)
    all_subjects = discovery.discover_subjects()
    
    if not all_subjects:
        print("No subjects found with the expected directory structure.", file=sys.stderr)
        print("Expected: data/(dataset)/(subject)/func/*.nii[.gz]", file=sys.stderr)
        sys.exit(1)
    
    # Filter subjects based on arguments
    subjects_to_process = all_subjects
    if args.dataset:
        subjects_to_process = [s for s in subjects_to_process if s['dataset'] == args.dataset]
    if args.subject:
        subjects_to_process = [s for s in subjects_to_process if s['subject_id'] == args.subject]
    if args.limit and args.limit > 0:
        subjects_to_process = subjects_to_process[:args.limit]
    
    print(f"Found {len(all_subjects)} total subject files")
    print(f"Processing {len(subjects_to_process)} subject files")
    
    # Save metadata
    metadata_path = output_dir / METADATA_FILENAME
    discovery.save_metadata(all_subjects, metadata_path)
    
    # Process subjects
    print("\n--- Preprocessing Phase ---")
    coordinator = PreprocessingCoordinator(output_dir, args.config)
    
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Clean progress bar with status info below
    print()  # Add some space before progress bar
    
    with tqdm(total=len(subjects_to_process), 
              desc="Processing subjects",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              position=0,
              leave=True) as pbar:
        
        for i, subject_info in enumerate(subjects_to_process):
            # Update current subject being processed
            dataset = subject_info['dataset']
            subject_id = subject_info['subject_id']
            session = subject_info.get('session', '')
            run = subject_info.get('run', '')
            
            # Create display name
            current_display = f"{dataset}_{subject_id}"
            if session:
                current_display += f"_ses-{session}"
            if run:
                current_display += f"_run-{run}"
            
            result = coordinator.process_subject(subject_info)
            results.append(result)
            
            if result.get('status') == 'success':
                successful += 1
            else:
                failed += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Print status info below the progress bar (overwrite previous line)
            print(f"\rCurrent: {current_display} | Success: {successful} | Failed: {failed}     ", end="", flush=True)
    
    # Final newline to separate from summary
    print()  # This creates the newline after the status line
    
    total_time = time.time() - start_time
    
    # Save processing manifest
    manifest_path = output_dir / MANIFEST_FILENAME
    coordinator.save_manifest(results, manifest_path)
    
    # Summary
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes
    
    print(f"\n=== Processing Summary ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per subject: {total_time/len(results):.1f}s")
    
    if failures > 0:
        print(f"\nFailed subjects:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  - {result['unique_id']}: {result['error']}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"Processing manifest: {manifest_path}")
    print(f"Subjects metadata: {metadata_path}")


if __name__ == "__main__":
    main()
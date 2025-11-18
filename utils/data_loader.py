#  Acts as the Input Module for ADHD Preprocessing Pipeline

"""
Data Loader for ADHD Preprocessing Pipeline
- Handles discovery of rs-fMRI data from hierarchical structure: data/(dataset)/(subject)/func/*.nii[.gz]
- Provides subject metadata for downstream preprocessing
- Saves metadata CSV for reference
- Utilities for loading features and labels for training
"""

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class DataDiscovery:
    """Simple data discovery for ADHD-200 datasets"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def discover_subjects(self) -> List[Dict[str, Any]]:
        """Discovery that finds ALL functional runs per subject"""
        subjects = []
        
        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue

            dataset_name = dataset_dir.name
            print(f"Processing dataset: {dataset_name}")
            
            # ✅ Load participants info FIRST
            participants_info = self._load_participants_info(dataset_dir)
            print(f"📋 {dataset_name}: Loaded {len(participants_info)} participant diagnoses")
            
            dataset_subjects = []
            subject_dirs = [d for d in dataset_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('sub-')]
            
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                
                # ✅ Get proper diagnosis using existing logic
                diagnosis = self._get_diagnosis(subject_id, participants_info)
                
                # Get ALL functional files for this subject
                func_files = self._find_all_functional_files(subject_dir)
                
                if func_files:
                    # Create entry for EACH run
                    for i, func_file in enumerate(func_files):
                        # Extract run number from filename
                        import re
                        run_match = re.search(r'run-(\d+)', func_file.name)
                        run_number = run_match.group(1) if run_match else str(i+1)
                        
                        entry = {
                            'dataset': dataset_name,
                            'site': dataset_name,
                            'subject_id': subject_id,
                            'run': run_number,
                            'session': '1',
                            'task': 'rest',
                            'input_path': str(func_file),
                            'relative_path': str(func_file.relative_to(self.data_root)),
                            'file_size_mb': round(func_file.stat().st_size / (1024 * 1024), 2),
                            'diagnosis': diagnosis,
                            'has_session_dir': 'ses-' in str(func_file)
                        }
                        dataset_subjects.append(entry)
        
            # ✅ MOVE THESE INSIDE THE LOOP
            subjects.extend(dataset_subjects)
            
            # Count runs vs subjects for THIS dataset
            total_runs = len(dataset_subjects)
            unique_subjects = len(set(s['subject_id'] for s in dataset_subjects))
            print(f"Dataset {dataset_name}: {total_runs} runs from {unique_subjects} subjects")

        # ✅ Final summary AFTER the loop completes
        if subjects:
            total_runs = len(subjects)
            total_unique_subjects = len(set(s['subject_id'] for s in subjects))
            total_sites = len(set(s['site'] for s in subjects))
            
            print(f"\nData Loading Summary")
            print(f"Total: {total_runs} runs from {total_unique_subjects} subjects across {total_sites} sites")
            
            # Breakdown by site
            site_stats = {}
            for subj in subjects:
                site = subj['site']
                if site not in site_stats:
                    site_stats[site] = {'runs': 0, 'subjects': set()}
                site_stats[site]['runs'] += 1
                site_stats[site]['subjects'].add(subj['subject_id'])
            
            print("Site breakdown:")
            for site, stats in site_stats.items():
                print(f"  {site}: {stats['runs']} runs from {len(stats['subjects'])} subjects")
        else:
            print("\nNo subjects discovered across any sites!")

        return subjects

    def _find_all_functional_files(self, subject_dir: Path) -> List[Path]:
        """Find ALL functional files for a subject"""
        
        nii_files = list(subject_dir.rglob("*.nii*"))
        
        functional_files = []
        for file in nii_files:
            name_lower = file.name.lower()
            
            # Skip anatomical scans
            if any(skip in name_lower for skip in ['anat', 't1w', 't2w', 'flair', 'dwi']):
                continue
                
            # Include functional files
            if any(keyword in name_lower for keyword in ['rest', 'bold', 'func', 'task']):
                functional_files.append(file)
        
        # Sort by run number for consistency
        functional_files.sort(key=lambda x: x.name)
        
        return functional_files

    def _find_any_nii_file(self, subject_dir: Path) -> Path:
        """Find any .nii or .nii.gz file in the subject directory with enhanced search"""
        
        print(f"  Searching in: {subject_dir.name}")
        
        # Look for .nii files recursively in the subject directory
        nii_files = list(subject_dir.rglob("*.nii*"))
        
        print(f"    Found {len(nii_files)} .nii files total")
        
        if not nii_files:
            # Debug: show what directories exist
            subdirs = [d.name for d in subject_dir.iterdir() if d.is_dir()]
            print(f"    Available subdirectories: {subdirs}")
            return None
        
        # Debug: show first few files found
        for i, file in enumerate(nii_files[:3]):
            rel_path = file.relative_to(subject_dir)
            size_mb = file.stat().st_size / (1024*1024)
            print(f"      {i+1}. {rel_path} ({size_mb:.1f}MB)")
        
        # If multiple files, prefer larger ones (likely functional data)
        nii_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        # Filter out obviously non-functional files
        functional_candidates = []
        for file in nii_files:
            name_lower = file.name.lower()
            
            # Skip anatomical scans
            if any(skip in name_lower for skip in ['anat', 't1w', 't2w', 'flair', 'dwi']):
                continue
                
            # Prefer files with functional keywords
            if any(keyword in name_lower for keyword in ['rest', 'bold', 'func', 'task']):
                functional_candidates.append(file)
                print(f"    ✅ FUNCTIONAL candidate: {file.name}")
        
        # Return the first functional candidate, or the largest file if no clear functional files
        if functional_candidates:
            selected = functional_candidates[0]
            print(f"    🎯 SELECTED: {selected.relative_to(subject_dir)}")
            return selected
        elif nii_files:
            # Return largest file (likely functional if it's big enough)
            largest_file = nii_files[0]
            size_mb = largest_file.stat().st_size / (1024*1024)
            if size_mb > 50:  # > 50MB
                print(f"Selected (large file): {largest_file.name} ({size_mb:.1f}MB)")
                return largest_file
            else:
                print(f"Largest file too small: {largest_file.name} ({size_mb:.1f}MB)")
        
        return None

    def _get_diagnosis(self, subject_id: str, participants_info: Dict) -> int:
        """Get diagnosis for subject (0=Control, 1=ADHD)"""
        if subject_id in participants_info:
            return participants_info[subject_id]
        
        # Try without 'sub-' prefix
        alt_id = subject_id.replace('sub-', '')
        if alt_id in participants_info:
            return participants_info[alt_id]
        
        # Try with 'sub-' prefix
        alt_id = f"sub-{subject_id.replace('sub-', '')}"
        if alt_id in participants_info:
            return participants_info[alt_id]
        
        # Default to control (0) if diagnosis unknown
        return 0

    @staticmethod
    def save_metadata(subjects: List[Dict[str, Any]], output_path: Path):
        """Save subjects metadata to CSV"""
        df = pd.DataFrame(subjects)
        df.to_csv(output_path, index=False)
        print(f"Saved metadata for {len(subjects)} subjects to {output_path}")

    def _load_participants_info(self, dataset_dir: Path) -> Dict:
        """Load participants.tsv or phenotypic data for diagnosis information"""
        participants_info = {}
        
        # Common ADHD-200 metadata files
        metadata_files = [
            dataset_dir / "participants.tsv",
            dataset_dir / "participants.txt",
            dataset_dir / "phenotypic_data.tsv",
            dataset_dir / "phenotypic_data.txt",
            dataset_dir / f"{dataset_dir.name}_phenotypic.csv",
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    # Try different separators
                    for sep in ['\t', ',', ' ']:
                        try:
                            df = pd.read_csv(metadata_file, sep=sep)
                            
                            # Find participant ID column
                            id_col = None
                            for col in ['participant_id', 'Subject', 'ScanDir ID', 'ID', 'subject_id']:
                                if col in df.columns:
                                    id_col = col
                                    break
                            
                            if id_col is None:
                                continue
                            
                            # Find diagnosis column
                            dx_col = None
                            for col in ['DX', 'diagnosis', 'dx', 'group', 'Diagnosis']:
                                if col in df.columns:
                                    dx_col = col
                                    break
                            
                            if dx_col is not None:
                                for _, row in df.iterrows():
                                    subj_id = str(row[id_col])
                                    if not subj_id.startswith('sub-'):
                                        subj_id = f"sub-{subj_id}"
                                    
                                    # Convert diagnosis to binary (0=Control, 1=ADHD)
                                    dx_value = row[dx_col]
                                    
                                    # Handle missing/unknown values - SKIP these subjects
                                    if pd.isna(dx_value) or str(dx_value).strip().lower() in ['pending', 'unknown', 'n/a', '']:
                                        continue  # Skip subjects with unknown diagnosis
                                    
                                    # Convert to string for comparison
                                    dx_str = str(dx_value).strip()
                                    
                                    # ADHD-200 numeric codes: 0=Control, 1=ADHD-Combined, 2=ADHD-H/I, 3=ADHD-Inattentive
                                    if dx_value in [0, '0'] or dx_str in ['TDC', 'Control', 'control', 'TD', 'Typically Developing Children']:
                                        diagnosis = 0
                                    elif dx_value in [1, 2, 3, '1', '2', '3'] or 'ADHD' in dx_str:
                                        # Numeric: 1,2,3 = different ADHD subtypes
                                        # String: ADHD-Combined, ADHD-Inattentive, ADHD-Hyperactive/Impulsive
                                        diagnosis = 1
                                    else:
                                        print(f"  ⚠️  Unknown diagnosis value '{dx_value}' for {subj_id}, skipping")
                                        continue  # Skip unknown values
                                    
                                    participants_info[subj_id] = diagnosis
                                
                                return participants_info
                                
                        except:
                            continue
                        
                except Exception as e:
                    print(f"Error reading {metadata_file}: {e}")
                    continue
    
        return participants_info


def load_metadata(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load a previously saved metadata CSV"""
    df = pd.read_csv(manifest_path)
    return df.to_dict(orient='records')


def load_features_and_labels(
    manifest_df: pd.DataFrame,
    roi_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load functional connectivity matrices and ROI time series from feature manifest
    
    Args:
        manifest_df: DataFrame with columns [subject_id, site, diagnosis, fc_path, ts_path]
        roi_indices: Optional list of ROI indices to select (for ROI ranking)
        
    Returns:
        fc_matrices: (n_subjects, n_rois, n_rois) or (n_subjects, n_selected_rois, n_selected_rois)
        roi_timeseries: (n_subjects, n_rois, n_timepoints) or (n_subjects, n_selected_rois, n_timepoints)
        labels: (n_subjects,)
        sites: (n_subjects,) as string array
    """
    
    n_subjects = len(manifest_df)
    
    # Initialize lists to collect data
    fc_matrices_list = []
    roi_timeseries_list = []
    labels_list = []
    sites_list = []
    
    print(f"Loading features for {n_subjects} subjects...")
    
    # First pass to determine max timepoints
    max_timepoints = 0
    for idx, row in manifest_df.iterrows():
        ts_path = Path(row['ts_path'])
        if ts_path.suffix == '.csv':
            roi_ts = pd.read_csv(ts_path).values
            max_timepoints = max(max_timepoints, roi_ts.shape[0])
        else:
            roi_ts = np.load(ts_path)
            max_timepoints = max(max_timepoints, roi_ts.shape[-1])
    
    print(f"Max timepoints found: {max_timepoints}")
    
    # Second pass to load and pad/truncate
    for idx, row in manifest_df.iterrows():
        # Load FC matrix
        fc_path = Path(row['fc_path'])
        fc_matrix = np.load(fc_path)
        
        # Load ROI time series  
        ts_path = Path(row['ts_path'])
        # Handle both .npy and .csv formats
        if ts_path.suffix == '.csv':
            roi_ts = pd.read_csv(ts_path).values.T  # Transpose to (n_rois, n_timepoints)
        else:
            roi_ts = np.load(ts_path)
        
        # Pad or truncate to max_timepoints
        n_rois, n_timepoints = roi_ts.shape
        if n_timepoints < max_timepoints:
            # Pad with zeros
            padding = np.zeros((n_rois, max_timepoints - n_timepoints))
            roi_ts = np.concatenate([roi_ts, padding], axis=1)
        elif n_timepoints > max_timepoints:
            # Truncate
            roi_ts = roi_ts[:, :max_timepoints]
        
        # Select ROIs if specified
        if roi_indices is not None:
            fc_matrix = fc_matrix[np.ix_(roi_indices, roi_indices)]
            roi_ts = roi_ts[roi_indices, :]
        
        fc_matrices_list.append(fc_matrix)
        roi_timeseries_list.append(roi_ts)
        labels_list.append(row['diagnosis'])
        sites_list.append(row['site'])
    
    # Convert to numpy arrays
    fc_matrices = np.stack(fc_matrices_list, axis=0)
    roi_timeseries = np.stack(roi_timeseries_list, axis=0)
    labels = np.array(labels_list)
    sites = np.array(sites_list)
    
    print(f"Loaded shapes:")
    print(f"  FC matrices: {fc_matrices.shape}")
    print(f"  ROI timeseries: {roi_timeseries.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Sites: {sites.shape}")
    
    return fc_matrices, roi_timeseries, labels, sites


def load_roi_ranking_results(roi_ranking_path: Path, top_k: Optional[int] = None) -> List[int]:
    """
    Load ROI ranking results and return top-k ROI indices
    
    Args:
        roi_ranking_path: Path to ROI ranking CSV file
        top_k: Number of top ROIs to select (if None, uses all)
        
    Returns:
        List of ROI indices (0-indexed)
    """
    
    ranking_df = pd.read_csv(roi_ranking_path)
    
    if top_k is not None:
        ranking_df = ranking_df.head(top_k)
    
    # ROI indices are usually in a column like 'roi_id' or 'roi_index'
    if 'roi_index' in ranking_df.columns:
        roi_indices = ranking_df['roi_index'].tolist()
    elif 'roi_id' in ranking_df.columns:
        roi_indices = ranking_df['roi_id'].tolist()
    else:
        # Assume first column is ROI index
        roi_indices = ranking_df.iloc[:, 0].tolist()
    
    return roi_indices


def create_dataloaders_from_manifest(
    manifest_df: pd.DataFrame,
    batch_size: int = 16,
    num_workers: int = 4,
    roi_indices: Optional[List[int]] = None
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoaders from feature manifest
    
    Args:
        manifest_df: Feature manifest DataFrame
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        roi_indices: Optional ROI indices to select
        
    Returns:
        DataLoader for the data
    """
    
    from training.dataset import ADHDDataset, collate_fn
    
    # Load data
    fc_matrices, roi_timeseries, labels, sites = load_features_and_labels(
        manifest_df, roi_indices
    )
    
    # Create dataset
    dataset = ADHDDataset(
        fc_matrices=fc_matrices,
        roi_timeseries=roi_timeseries,
        labels=labels,
        sites=sites
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data discovery for ADHD preprocessing pipeline")
    parser.add_argument("--data-dir", required=True, help="Root data directory: data/(dataset)/(subject)/func/")
    parser.add_argument("--metadata-out", default="subjects_metadata.csv", help="CSV output path for metadata")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    discovery = DataDiscovery(data_root)
    subjects = discovery.discover_subjects()
    discovery.save_metadata(subjects, Path(args.metadata_out))

    print(f"Discovered {len(subjects)} subject files.")

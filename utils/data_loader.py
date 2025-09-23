#  Acts as the Input Module for ADHD Preprocessing Pipeline

"""
Data Loader for ADHD Preprocessing Pipeline
- Handles discovery of rs-fMRI data from hierarchical structure: data/(dataset)/(subject)/func/*.nii[.gz]
- Provides subject metadata for downstream preprocessing
- Saves metadata CSV for reference
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

class DataDiscovery:
    """Handles discovery and organization of rs-fMRI data from hierarchical structure"""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    def discover_subjects(self) -> List[Dict[str, Any]]:
        """Discover all subjects and their functional files"""
        subjects = []

        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            for subject_dir in dataset_dir.iterdir():
                if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                    continue

                subject_id = subject_dir.name
                subjects_found = self._discover_subject_files(dataset_name, subject_id, subject_dir)
                subjects.extend(subjects_found)

        return sorted(subjects, key=lambda x: (x['dataset'], x['subject_id'], x.get('session') or '', x.get('run') or ''))

    def _discover_subject_files(self, dataset_name: str, subject_id: str, subject_dir: Path) -> List[Dict[str, Any]]:
        """Discover functional files for a single subject"""
        subject_files = []

        # Direct func directory
        direct_func_dir = subject_dir / "func"
        if direct_func_dir.exists():
            subject_files.extend(self._find_nifti_in_func_dir(direct_func_dir, dataset_name, subject_id, None))

        # Session directories
        for item in subject_dir.iterdir():
            if item.is_dir() and item.name.startswith('ses-'):
                session_name = item.name
                session_func_dir = item / "func"
                if session_func_dir.exists():
                    subject_files.extend(self._find_nifti_in_func_dir(session_func_dir, dataset_name, subject_id, session_name))

        return subject_files

    def _find_nifti_in_func_dir(self, func_dir: Path, dataset_name: str, subject_id: str,
                                session_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all NIfTI files in a func directory and create subject entries"""
        subject_entries = []

        nifti_files = []
        for pattern in ["*.nii", "*.nii.gz"]:
            nifti_files.extend(list(func_dir.glob(pattern)))

        for nifti_file in nifti_files:
            filename = nifti_file.name
            session_from_filename = self._extract_bids_field(filename, 'ses')
            run = self._extract_bids_field(filename, 'run')
            task = self._extract_bids_field(filename, 'task')

            final_session = session_name.replace('ses-', '') if session_name else session_from_filename

            entry = {
                'dataset': dataset_name,
                'subject_id': subject_id,
                'session': final_session,
                'run': run,
                'task': task or 'rest',
                'input_path': str(nifti_file),
                'relative_path': str(nifti_file.relative_to(self.data_root)),
                'file_size_mb': round(nifti_file.stat().st_size / (1024 * 1024), 2),
                'has_session_dir': session_name is not None
            }
            subject_entries.append(entry)

        return subject_entries

    @staticmethod
    def _extract_bids_field(filename: str, field: str) -> Optional[str]:
        """Extract BIDS field from filename (e.g., 'ses-1' -> '1')"""
        import re
        pattern = f"{field}-([^_]+)"
        match = re.search(pattern, filename)
        return match.group(1) if match else None

    @staticmethod
    def save_metadata(subjects: List[Dict[str, Any]], output_path: Path):
        """Save subjects metadata to CSV"""
        df = pd.DataFrame(subjects)
        df.to_csv(output_path, index=False)
        print(f"Saved metadata for {len(subjects)} subjects to {output_path}")


def load_metadata(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load a previously saved metadata CSV"""
    df = pd.read_csv(manifest_path)
    return df.to_dict(orient='records')


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

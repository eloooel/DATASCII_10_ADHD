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
    """Enhanced data discovery for ADHD-200 datasets with flexible naming patterns"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def discover_subjects(self) -> List[Dict[str, Any]]:
        """Discover all subjects with flexible ADHD-200 naming patterns"""
        subjects = []
        
        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue

            dataset_name = dataset_dir.name
            
            # Load site-specific participants info if available
            participants_info = self._load_participants_info(dataset_dir)
            
            dataset_subjects = []
            for subject_dir in dataset_dir.iterdir():
                if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                    continue

                subject_id = subject_dir.name
                
                # Find functional file with flexible patterns
                func_file, session = self._find_functional_file(subject_dir, subject_id)
                
                if func_file:
                    # Extract diagnosis from participants info
                    diagnosis = self._get_diagnosis(subject_id, participants_info)
                    
                    entry = {
                        'dataset': dataset_name,
                        'site': dataset_name,  # Use dataset name as site
                        'subject_id': subject_id,
                        'session': session or '1',
                        'task': 'rest',
                        'input_path': str(func_file),
                        'relative_path': str(func_file.relative_to(self.data_root)),
                        'file_size_mb': round(func_file.stat().st_size / (1024 * 1024), 2),
                        'diagnosis': diagnosis,
                        'has_session_dir': 'ses-' in str(func_file)
                    }
                    dataset_subjects.append(entry)
        
        subjects.extend(dataset_subjects)

        # Single line output
        sites = len(set(s['site'] for s in subjects))
        print(f"Total discovered: {len(subjects)} subjects across {sites} sites")
        return subjects

    def _find_functional_file(self, subject_dir: Path, subject_id: str) -> tuple:
        """Find functional fMRI file with flexible ADHD-200 naming patterns"""
        
        # Define search patterns in priority order (most specific to most general)
        search_patterns = [
            # BIDS-compliant patterns
            (f"ses-*/func/{subject_id}_ses-*_task-rest*_bold.nii.gz", "bids_session"),
            (f"func/{subject_id}_ses-*_task-rest*_bold.nii.gz", "bids_no_session"),
            (f"ses-*/func/{subject_id}_*task-rest*.nii.gz", "bids_flexible"),
            
            # ADHD-200 common patterns
            (f"func/{subject_id}_task-rest*.nii.gz", "adhd200_task"),
            (f"func/{subject_id}_rest*.nii.gz", "adhd200_rest"),
            (f"func/{subject_id}_func*.nii.gz", "adhd200_func"),
            (f"ses-*/func/{subject_id}_*.nii.gz", "session_any"),
            (f"func/{subject_id}*.nii.gz", "func_any"),
            
            # Fallback patterns
            (f"**/*rest*.nii.gz", "fallback_rest"),
            (f"**/*bold*.nii.gz", "fallback_bold"),
            (f"**/*func*.nii.gz", "fallback_func"),
            (f"**/{subject_id}*.nii.gz", "fallback_subject"),
        ]
        
        for pattern, pattern_type in search_patterns:
            matches = list(subject_dir.glob(pattern))
            
            if matches:
                # Prefer files with 'rest' or 'bold' in the name
                best_match = self._select_best_functional_file(matches)
                
                if best_match:
                    session = self._extract_session_from_path(best_match)
                    return best_match, session
        
        return None, None

    def _select_best_functional_file(self, candidates: List[Path]) -> Path:
        """Select the best functional file from multiple candidates"""
        if len(candidates) == 1:
            return candidates[0]
        
        # Scoring system for file selection
        scored_files = []
        for file in candidates:
            score = 0
            name_lower = file.name.lower()
            
            # Prefer rest-state files
            if 'rest' in name_lower:
                score += 10
            if 'bold' in name_lower:
                score += 8
            if 'task-rest' in name_lower:
                score += 15
            
            # Prefer session 1
            if 'ses-1' in name_lower:
                score += 5
            
            # Prefer run 1
            if 'run-1' in name_lower:
                score += 3
            
            # Prefer shorter names (usually more standard)
            score -= len(file.name) * 0.01
            
            scored_files.append((score, file))
        
        # Return file with highest score
        scored_files.sort(key=lambda x: x[0], reverse=True)
        return scored_files[0][1]

    def _extract_session_from_path(self, file_path: Path) -> str:
        """Extract session from file path or name"""
        path_str = str(file_path)
        
        # Look for ses-X pattern
        import re
        ses_match = re.search(r'ses-(\d+)', path_str)
        if ses_match:
            return ses_match.group(1)
        
        # Default to session 1
        return '1'



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
        """Save subjects metadata to CSV with summary"""
        df = pd.DataFrame(subjects)
        df.to_csv(output_path, index=False)
        # No additional output

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
                            for col in ['participant_id', 'Subject', 'ScanDir ID', 'subject_id']:
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
                                    if pd.isna(dx_value):
                                        diagnosis = 0  # Default to control
                                    elif dx_value in [0, '0', 'TDC', 'Control', 'control', 'TD']:
                                        diagnosis = 0
                                    elif dx_value in [1, '1', 'ADHD', 'adhd', 'ADHD-C', 'ADHD-I', 'ADHD-H']:
                                        diagnosis = 1
                                    else:
                                        diagnosis = 0  # Default unknown to control
                                    
                                    participants_info[subj_id] = diagnosis
                                

                                return participants_info
                                
                        except:
                            continue
                        
                except Exception as e:
                    print(f"  ⚠️ Error reading {metadata_file}: {e}")
                    continue
    
        return participants_info


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

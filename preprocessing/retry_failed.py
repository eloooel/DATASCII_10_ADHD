"""
Utility script to identify and reprocess failed subjects
"""
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocess import (
    identify_failed_subjects,
    cleanup_failed_subject,
    _process_subject
)
from utils import run_parallel

def retry_failed_preprocessing(
    manifest_path: str,
    output_dir: str,
    cleanup: bool = True,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Identify and reprocess failed subjects
    
    Args:
        manifest_path: Path to original manifest CSV
        output_dir: Base output directory
        cleanup: Whether to cleanup corrupted files before retry
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with retry results
    """
    output_base = Path(output_dir)
    
    # Step 1: Identify failed subjects
    print("Identifying failed subjects...")
    failed_subjects = identify_failed_subjects(output_base)
    
    if not failed_subjects:
        print("Success: No failed subjects found!")
        return pd.DataFrame()
    
    print(f"\nFailed: Found {len(failed_subjects)} failed subjects:")
    for subj in failed_subjects:
        print(f"  - {subj['subject_id']} ({subj['site']}): {subj['reason']}")
    
    # Step 2: Load original manifest to get input paths
    manifest_df = pd.read_csv(manifest_path)
    
    # Step 3: Create retry list with input paths
    retry_list = []
    for failed in failed_subjects:
        # Find matching row in manifest
        match = manifest_df[manifest_df['subject_id'] == failed['subject_id']]
        
        if match.empty:
            print(f"Warning: {failed['subject_id']} not found in manifest")
            continue
        
        row = match.iloc[0].to_dict()
        row['force_retry'] = True
        row['out_dir'] = output_dir
        
        # Cleanup if requested
        if cleanup:
            cleanup_failed_subject(Path(failed['output_dir']))
        
        retry_list.append(row)
    
    if not retry_list:
        print("Error: No subjects to retry (couldn't match manifest)")
        return pd.DataFrame()
    
    print(f"\nRetrying {len(retry_list)} subjects...")
    
    # Step 4: Reprocess with parallel execution
    # FIX: Check if n_jobs > 1 for parallel, otherwise sequential
    if n_jobs > 1:
        results = run_parallel(_process_subject, retry_list, desc=f"Retrying {len(retry_list)} subjects")
    else:
        # Sequential processing
        print("Processing sequentially (n_jobs=1)...")
        results = []
        from tqdm import tqdm
        for row in tqdm(retry_list, desc="Retrying subjects"):
            result = _process_subject(row)
            results.append(result)
    
    # Step 5: Summarize results
    results_df = pd.DataFrame(results)
    
    success_count = len(results_df[results_df['status'] == 'success'])
    failed_count = len(results_df[results_df['status'] == 'failed'])
    
    print(f"\n{'='*60}")
    print(f"Retry Summary:")
    print(f"  Success: Successful: {success_count}/{len(retry_list)}")
    print(f"  Failed: Failed: {failed_count}/{len(retry_list)}")
    
    if failed_count > 0:
        print(f"\nStill failing:")
        for _, row in results_df[results_df['status'] == 'failed'].iterrows():
            print(f"  - {row['subject_id']}: {row.get('error', 'Unknown error')}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retry failed preprocessing")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup before retry")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs (1 for sequential)")
    
    args = parser.parse_args()
    
    results = retry_failed_preprocessing(
        manifest_path=args.manifest,
        output_dir=args.output,
        cleanup=not args.no_cleanup,
        n_jobs=args.jobs
    )
    
    # Save retry results
    results_path = Path(args.output) / "retry_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved: Retry results saved to: {results_path}")

"""
Script to run preprocessing only for WashU subjects
"""
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from preprocessing.preprocess import _process_subject
from utils import run_parallel

# Configuration
RAW_DIR = Path("./data/raw")
PREPROC_OUT = Path("./data/preprocessed")
METADATA_PATH = RAW_DIR / "subjects_metadata.csv"

def preprocess_washu_only(parallel: bool = True):
    """Run preprocessing only for WashU subjects"""
    
    print("="*70)
    print("PREPROCESSING: WashU ONLY")
    print("="*70)
    
    # Load full metadata
    if not METADATA_PATH.exists():
        print(f"❌ Metadata file not found: {METADATA_PATH}")
        print("Run with --stage preprocessing to generate metadata first")
        return
    
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Total subjects in metadata: {len(metadata)}")
    
    # Filter for WashU only
    washu_metadata = metadata[metadata['site'] == 'WashU'].copy()
    
    if len(washu_metadata) == 0:
        print("❌ No WashU subjects found in metadata!")
        print("\nAvailable sites:")
        print(metadata['site'].value_counts())
        return
    
    print(f"\n📊 Found {len(washu_metadata)} WashU entries (includes multiple runs per subject)")
    
    # WashU has multiple runs AND multiple sessions per subject
    # Strategy: Use only the FIRST run per subject (simplest approach)
    if 'run' in washu_metadata.columns:
        # Group by subject_id and take first run only
        grouped = washu_metadata.groupby('subject_id')
        total_runs = len(washu_metadata)
        
        print(f"\n📊 Subject breakdown:")
        print(f"  - Total unique subjects: {len(grouped)}")
        print(f"  - Total runs across all subjects: {total_runs}")
        print(f"  - Average runs per subject: {total_runs / len(grouped):.1f}")
        
        # Create a list with only the first run per subject
        subjects_first_run = []
        for subject_id, group in grouped:
            # Sort by session and run to ensure consistent "first" run
            group_sorted = group.sort_values(['session', 'run'])
            first_run = group_sorted.iloc[0].to_dict()
            subjects_first_run.append(first_run)
        
        washu_metadata = pd.DataFrame(subjects_first_run)
        print(f"✅ Selected first run for each of {len(washu_metadata)} subjects")
        print(f"   (Discarding {total_runs - len(subjects_first_run)} additional runs)")
    else:
        print("⚠️  No 'run' column found - using all entries")
    
    print(f"\nSubject IDs: {washu_metadata['subject_id'].unique().tolist()[:5]}{'...' if len(washu_metadata) > 5 else ''}")
    
    # Create output directory
    PREPROC_OUT.mkdir(parents=True, exist_ok=True)
    (PREPROC_OUT / "WashU").mkdir(exist_ok=True)
    
    # Prepare batch data
    batch_size = 8 if parallel else 1
    all_results = []
    
    print(f"\nProcessing {len(washu_metadata)} subjects (batch_size={batch_size})...")
    
    with tqdm(total=len(washu_metadata), desc="Preprocessing WashU", unit="subj") as pbar:
        for i in range(0, len(washu_metadata), batch_size):
            batch = washu_metadata.iloc[i:i+batch_size].copy()
            batch['device'] = 'cpu'
            batch['out_dir'] = str(PREPROC_OUT)
            
            batch_num = (i // batch_size) + 1
            total_batches = (len(washu_metadata) - 1) // batch_size + 1
            
            if parallel and len(batch) > 1:
                pbar.set_postfix_str(f"Batch {batch_num}/{total_batches}")
                results = run_parallel(
                    func=_process_subject,
                    items=batch.to_dict('records'),
                    desc="Processing batch"
                )
                pbar.update(len(batch))
            else:
                # Sequential processing
                results = []
                for _, row in batch.iterrows():
                    subject_id = row.get("subject_id", "unknown")
                    pbar.set_postfix_str(f"{subject_id}")
                    result = _process_subject(row)
                    results.append(result)
                    pbar.update(1)
            
            all_results.extend(results)
            
            # Pause between batches
            import time
            time.sleep(1)
    
    # Print summary
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"✅ Success: {success}/{len(washu_metadata)} subjects")
    print(f"❌ Failed:  {failed}/{len(washu_metadata)} subjects")
    
    if failed > 0:
        failed_subjects = [r for r in all_results if r["status"] == "failed"]
        print(f"\nFailed subjects:")
        for fail in failed_subjects[:10]:
            print(f"  - {fail['subject_id']}: {fail.get('error', 'unknown error')}")
        if len(failed_subjects) > 10:
            print(f"  ... and {len(failed_subjects) - 10} more")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = PREPROC_OUT / "WashU" / "preprocessing_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess WashU subjects only")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run preprocessing in parallel (default: True)")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel",
                       help="Disable parallel processing")
    args = parser.parse_args()
    
    preprocess_washu_only(parallel=args.parallel)

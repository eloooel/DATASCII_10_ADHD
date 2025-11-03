"""
Preprocess all sites with multi-run concatenation support
For sites with multiple runs per subject, automatically concatenates them
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import torch
from tqdm import tqdm
from preprocessing.preprocess import _process_subject
from utils.parallel_runner import run_parallel

# Configuration
RAW_DIR = Path("./data/raw")
PREPROC_OUT = Path("./data/preprocessed")
METADATA_PATH = RAW_DIR / "subjects_metadata.csv"

def group_runs_by_subject(metadata_df):
    """
    Group runs by subject for multi-run concatenation
    Returns a list where each entry has all_runs for subjects with multiple runs
    """
    subjects_list = []
    
    # Group by site and subject_id
    for site in metadata_df['site'].unique():
        site_data = metadata_df[metadata_df['site'] == site]
        
        # Group by subject within this site
        grouped = site_data.groupby('subject_id')
        
        for subject_id, group in grouped:
            # Sort by session and run for consistent ordering
            group_sorted = group.sort_values(['session', 'run'])
            
            # Get all run paths
            runs_list = group_sorted['input_path'].tolist()
            
            # Use first entry as template
            subject_entry = group_sorted.iloc[0].to_dict()
            
            # Add all_runs list if multiple runs exist
            if len(runs_list) > 1:
                subject_entry['all_runs'] = runs_list
                subject_entry['num_runs'] = len(runs_list)
            
            subjects_list.append(subject_entry)
    
    return subjects_list

def preprocess_all_sites(sites=None, parallel=True, batch_size=8):
    """
    Preprocess all sites with multi-run concatenation
    
    Args:
        sites: List of sites to process (None = all sites)
        parallel: Use parallel processing
        batch_size: Number of subjects per batch
    """
    
    print("="*70)
    print("PREPROCESSING: ALL SITES WITH MULTI-RUN SUPPORT")
    print("="*70)
    
    # Load metadata
    if not METADATA_PATH.exists():
        print(f"❌ Metadata file not found: {METADATA_PATH}")
        print("Run scripts/update_metadata_all_sites.py first")
        return
    
    metadata = pd.read_csv(METADATA_PATH)
    print(f"📊 Total entries in metadata: {len(metadata)}")
    print(f"📊 Total unique subjects: {metadata['subject_id'].nunique()}")
    
    # Filter by sites if specified
    if sites:
        metadata = metadata[metadata['site'].isin(sites)]
        print(f"\n🎯 Filtering to sites: {sites}")
        print(f"   Remaining entries: {len(metadata)}")
    
    if len(metadata) == 0:
        print("❌ No subjects found!")
        return
    
    # Show site breakdown
    print(f"\nSites to process:")
    for site in metadata['site'].unique():
        site_data = metadata[metadata['site'] == site]
        n_subjects = site_data['subject_id'].nunique()
        n_runs = len(site_data)
        avg_runs = n_runs / n_subjects
        
        print(f"  {site}: {n_subjects} subjects, {n_runs} runs (avg {avg_runs:.1f} runs/subject)")
        if avg_runs > 1.1:
            print(f"    ✅ Multi-run site - will concatenate runs")
    
    # Group runs by subject for concatenation
    print(f"\n🔄 Grouping runs by subject for multi-run concatenation...")
    subjects_to_process = group_runs_by_subject(metadata)
    
    print(f"✅ Prepared {len(subjects_to_process)} subjects for preprocessing")
    
    # Count multi-run subjects
    multi_run_count = sum(1 for s in subjects_to_process if 'all_runs' in s and len(s['all_runs']) > 1)
    single_run_count = len(subjects_to_process) - multi_run_count
    
    print(f"   - Single-run subjects: {single_run_count}")
    print(f"   - Multi-run subjects: {multi_run_count}")
    
    # Create output directories
    PREPROC_OUT.mkdir(parents=True, exist_ok=True)
    for site in metadata['site'].unique():
        (PREPROC_OUT / site).mkdir(exist_ok=True)
    
    # Convert to DataFrame for processing
    subjects_df = pd.DataFrame(subjects_to_process)
    
    # Process in batches
    all_results = []
    
    print(f"\n⚙️  Processing {len(subjects_df)} subjects (batch_size={batch_size})...")
    print(f"   Parallel: {parallel}")
    
    total_subjects = len(subjects_df)
    total_batches = (total_subjects - 1) // batch_size + 1
    
    with tqdm(total=total_subjects, desc="Overall Progress", unit="subject", position=0) as pbar:
        for i in range(0, len(subjects_df), batch_size):
            batch = subjects_df.iloc[i:i+batch_size].copy()
            batch['device'] = 'cpu'
            batch['out_dir'] = str(PREPROC_OUT)
            
            batch_num = (i // batch_size) + 1
            batch_size_current = len(batch)
            
            # Update progress bar description with batch info
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            
            if parallel and len(batch) > 1:
                # Parallel processing - update after batch completes
                results = run_parallel(
                    func=_process_subject,
                    items=batch.to_dict('records'),
                    desc=""  # Suppress inner progress bar
                )
                
                # Update overall progress with batch results
                success_in_batch = sum(1 for r in results if r["status"] == "success")
                failed_in_batch = sum(1 for r in results if r["status"] == "failed")
                pbar.set_postfix_str(f"✅ {success_in_batch} ❌ {failed_in_batch}")
                pbar.update(len(results))
            else:
                # Sequential processing - update after each subject
                results = []
                for idx, (_, row) in enumerate(batch.iterrows()):
                    subject_id = row.get("subject_id", "unknown")
                    num_runs = row.get("num_runs", 1)
                    site = row.get("site", "unknown")
                    
                    # Show current subject being processed
                    pbar.set_postfix_str(f"Processing {site}/{subject_id} ({num_runs} runs)")
                    
                    result = _process_subject(row)
                    results.append(result)
                    
                    # Update cumulative success/fail counts
                    total_processed = i + idx + 1
                    success_count = sum(1 for r in all_results + results if r["status"] == "success")
                    failed_count = sum(1 for r in all_results + results if r["status"] == "failed")
                    pbar.set_postfix_str(f"✅ {success_count} ❌ {failed_count}")
                    pbar.update(1)
            
            all_results.extend(results)
            
            # Memory cleanup between batches
            import time
            import gc
            gc.collect()
            time.sleep(1)
    
    # Print summary
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    skipped = sum(1 for r in all_results if r.get("skipped", False))
    
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"✅ Success: {success}/{len(subjects_df)} subjects")
    print(f"⏭️  Skipped: {skipped}/{len(subjects_df)} subjects (already processed)")
    print(f"❌ Failed:  {failed}/{len(subjects_df)} subjects")
    
    if failed > 0:
        failed_subjects = [r for r in all_results if r["status"] == "failed"]
        print(f"\nFailed subjects:")
        for fail in failed_subjects[:10]:
            print(f"  - {fail['subject_id']}: {fail.get('error', 'unknown error')}")
        if len(failed_subjects) > 10:
            print(f"  ... and {len(failed_subjects) - 10} more")
    
    # Save results by site
    results_df = pd.DataFrame(all_results)
    for site in results_df['site'].unique():
        site_results = results_df[results_df['site'] == site]
        results_path = PREPROC_OUT / site / "preprocessing_results.csv"
        site_results.to_csv(results_path, index=False)
        print(f"\n💾 {site} results saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess all sites with multi-run support")
    parser.add_argument("--sites", nargs="+", help="Specific sites to process (e.g., NYU WashU)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run preprocessing in parallel (default: True)")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel",
                       help="Disable parallel processing")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Number of subjects per batch (default: 8)")
    
    args = parser.parse_args()
    
    preprocess_all_sites(
        sites=args.sites,
        parallel=args.parallel,
        batch_size=args.batch_size
    )

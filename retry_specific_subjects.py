"""
Manual retry script for specific failed subjects
"""
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import gc

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.preprocess import _process_subject, cleanup_failed_subject

def retry_specific_subjects(subject_ids: list, manifest_path: str, output_dir: str, cleanup: bool = True):
    """
    Retry specific subjects by ID
    
    Args:
        subject_ids: List of subject IDs to retry
        manifest_path: Path to subjects metadata CSV
        output_dir: Output directory for preprocessing
        cleanup: Whether to cleanup old files before retry
    """
    print("="*70)
    print("MANUAL RETRY FOR SPECIFIC SUBJECTS")
    print("="*70)
    
    # Load manifest
    print(f"\n📂 Loading manifest from: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    print(f"   Found {len(manifest_df)} total subjects in manifest")
    
    # Remove duplicates from subject list
    subject_ids = list(set(subject_ids))
    print(f"\n🎯 Target subjects ({len(subject_ids)} unique):")
    for subj_id in subject_ids:
        print(f"   - {subj_id}")
    
    # Find subjects in manifest
    retry_list = []
    not_found = []
    
    for subj_id in subject_ids:
        match = manifest_df[manifest_df['subject_id'] == subj_id]
        
        if match.empty:
            not_found.append(subj_id)
            print(f"⚠️  {subj_id}: Not found in manifest")
            continue
        
        row = match.iloc[0].to_dict()
        row['force_retry'] = True
        row['out_dir'] = output_dir
        
        # Get site for cleanup
        site = row.get('site', row.get('dataset', 'UnknownSite'))
        subject_dir = Path(output_dir) / site / subj_id
        
        # Cleanup if requested
        if cleanup and subject_dir.exists():
            print(f"🧹 Cleaning up {subj_id} ({site})")
            cleanup_failed_subject(subject_dir)
        
        retry_list.append(row)
    
    if not_found:
        print(f"\n❌ Could not find {len(not_found)} subjects in manifest:")
        for subj_id in not_found:
            print(f"   - {subj_id}")
    
    if not retry_list:
        print("\n❌ No subjects to retry!")
        return []
    
    print(f"\n🔄 Processing {len(retry_list)} subjects sequentially...")
    print("="*70)
    
    # Process subjects one at a time with detailed output
    results = []
    
    for i, row in enumerate(retry_list, 1):
        subject_id = row['subject_id']
        site = row.get('site', row.get('dataset', 'UnknownSite'))
        
        print(f"\n[{i}/{len(retry_list)}] Processing {subject_id} ({site})")
        print("-"*70)
        
        try:
            result = _process_subject(row)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {subject_id}: SUCCESS")
                print(f"   - Functional: {result.get('func_size_mb', 'N/A'):.1f}MB")
                print(f"   - Mask: {result.get('mask_size_mb', 'N/A'):.3f}MB")
            else:
                print(f"❌ {subject_id}: FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                print(f"   Type: {result.get('error_type', 'unknown')}")
        
        except Exception as e:
            print(f"❌ {subject_id}: EXCEPTION - {str(e)}")
            results.append({
                'status': 'failed',
                'subject_id': subject_id,
                'site': site,
                'error': str(e),
                'error_type': 'exception'
            })
        
        # Force cleanup between subjects
        gc.collect()
        
        print("-"*70)
    
    # Summary
    print("\n" + "="*70)
    print("RETRY SUMMARY")
    print("="*70)
    
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\n✅ Successful: {len(success)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if success:
        print("\n✅ Successfully processed:")
        for r in success:
            print(f"   - {r['subject_id']} ({r['site']})")
    
    if failed:
        print("\n❌ Still failing:")
        for r in failed:
            print(f"   - {r['subject_id']} ({r['site']}): {r.get('error', 'Unknown')[:80]}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path(output_dir) / "manual_retry_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    # ✅ YOUR FAILED SUBJECTS
    FAILED_SUBJECTS = [
        "sub-0027016",
        "sub-0027017", 
        "sub-3304956",
        "sub-3449233",
        "sub-3566449"
    ]
    
    # Configuration
    MANIFEST_PATH = "data/raw/subjects_metadata.csv"
    OUTPUT_DIR = "data/preprocessed"
    
    # Run retry
    results = retry_specific_subjects(
        subject_ids=FAILED_SUBJECTS,
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        cleanup=True  # Set to False to keep existing files
    )
    
    print("\n" + "="*70)
    print("RETRY COMPLETE")
    print("="*70)

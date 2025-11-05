"""
Fix WashU metadata duplicates.

This script removes duplicate entries from the metadata CSV where the same
file path appears multiple times for the same subject.
"""

import pandas as pd
from pathlib import Path

def fix_washu_duplicates():
    """Remove duplicate WashU entries from metadata."""
    
    metadata_path = Path("data/raw/subjects_metadata.csv")
    backup_path = Path("data/raw/subjects_metadata_backup.csv")
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Total entries: {len(df)}")
    
    # Filter WashU entries
    washu_df = df[df['site'] == 'WashU'].copy()
    other_df = df[df['site'] != 'WashU'].copy()
    
    print(f"WashU entries: {len(washu_df)}")
    print(f"Other sites entries: {len(other_df)}")
    
    # Check for duplicates
    print("\nChecking for duplicates...")
    duplicates = washu_df[washu_df.duplicated(subset=['subject_id', 'input_path'], keep=False)]
    print(f"Duplicate entries found: {len(duplicates)}")
    
    if len(duplicates) > 0:
        # Show examples
        print("\nExample duplicates:")
        for subject_id in washu_df['subject_id'].unique()[:3]:
            subject_data = washu_df[washu_df['subject_id'] == subject_id]
            unique_files = subject_data['input_path'].nunique()
            total_entries = len(subject_data)
            if unique_files != total_entries:
                print(f"  {subject_id}: {total_entries} entries, {unique_files} unique files")
        
        # Remove duplicates (keep first occurrence)
        print("\nRemoving duplicates...")
        washu_dedup = washu_df.drop_duplicates(subset=['subject_id', 'input_path'], keep='first')
        
        print(f"WashU entries after deduplication: {len(washu_dedup)}")
        print(f"Removed {len(washu_df) - len(washu_dedup)} duplicate entries")
        
        # Combine back
        df_fixed = pd.concat([washu_dedup, other_df], ignore_index=True)
        
        # Sort by site and subject
        df_fixed = df_fixed.sort_values(['site', 'subject_id', 'run']).reset_index(drop=True)
        
        print(f"\nTotal entries after fix: {len(df_fixed)}")
        
        # Backup original
        print(f"\nBacking up original to {backup_path}...")
        df.to_csv(backup_path, index=False)
        
        # Save fixed version
        print(f"Saving fixed metadata to {metadata_path}...")
        df_fixed.to_csv(metadata_path, index=False)
        
        print("\n✅ Metadata fixed successfully!")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Original entries: {len(df)}")
        print(f"Fixed entries: {len(df_fixed)}")
        print(f"Duplicates removed: {len(df) - len(df_fixed)}")
        print(f"\nWashU subjects: {washu_dedup['subject_id'].nunique()}")
        print(f"WashU runs: {len(washu_dedup)}")
        print(f"Average runs per subject: {len(washu_dedup) / washu_dedup['subject_id'].nunique():.1f}")
        
    else:
        print("\n✅ No duplicates found!")

if __name__ == "__main__":
    fix_washu_duplicates()

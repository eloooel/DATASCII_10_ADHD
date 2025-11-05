"""
Validate and clean metadata CSV.

This script:
1. Checks for duplicate entries (same subject_id, run, input_path)
2. Removes duplicates if found
3. Validates data integrity
4. Creates backup before any changes

Run this anytime you suspect duplicate entries or before preprocessing.
"""

import pandas as pd
from pathlib import Path

def validate_and_clean_metadata(metadata_path: Path, fix: bool = False):
    """
    Validate metadata for duplicates and data integrity.
    
    Args:
        metadata_path: Path to metadata CSV
        fix: If True, removes duplicates and saves cleaned version
    """
    
    print("="*70)
    print("METADATA VALIDATION")
    print("="*70)
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return False
    
    # Load metadata
    print(f"\n📂 Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Total entries: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    
    # Check for duplicates
    print("\n" + "="*70)
    print("DUPLICATE CHECK")
    print("="*70)
    
    # Check by (subject_id, run, input_path) - most strict
    duplicate_mask = df.duplicated(subset=['subject_id', 'run', 'input_path'], keep=False)
    duplicates = df[duplicate_mask]
    
    if len(duplicates) > 0:
        print(f"⚠️  Found {len(duplicates)} duplicate entries!")
        
        # Show examples
        print("\nExample duplicates (first 5 subjects affected):")
        for subject_id in duplicates['subject_id'].unique()[:5]:
            subject_dupes = duplicates[duplicates['subject_id'] == subject_id]
            print(f"\n  {subject_id}:")
            for _, row in subject_dupes.iterrows():
                print(f"    Run {row['run']}: {row['input_path']}")
        
        # Count by site
        print("\nDuplicates by site:")
        print(duplicates['site'].value_counts())
        
        if fix:
            print("\n" + "="*70)
            print("FIXING DUPLICATES")
            print("="*70)
            
            # Backup original
            backup_path = metadata_path.parent / f"{metadata_path.stem}_backup{metadata_path.suffix}"
            print(f"💾 Creating backup: {backup_path}")
            df.to_csv(backup_path, index=False)
            
            # Remove duplicates (keep first occurrence)
            df_clean = df.drop_duplicates(
                subset=['subject_id', 'run', 'input_path'],
                keep='first'
            )
            
            print(f"Removed {len(df) - len(df_clean)} duplicate entries")
            
            # Save cleaned version
            df_clean.to_csv(metadata_path, index=False)
            print(f"✅ Cleaned metadata saved to: {metadata_path}")
            
            # New summary
            print("\n" + "="*70)
            print("CLEANED METADATA SUMMARY")
            print("="*70)
            print(f"Total entries: {len(df_clean)}")
            print(f"Unique subjects: {df_clean['subject_id'].nunique()}")
            print("\nEntries by site:")
            print(df_clean['site'].value_counts())
            
            return True
        else:
            print("\n💡 Run with --fix flag to remove duplicates automatically")
            return False
    else:
        print("✅ No duplicates found - metadata is clean!")
        
        # Additional validation
        print("\n" + "="*70)
        print("DATA INTEGRITY CHECK")
        print("="*70)
        
        # Check for missing files
        missing_files = []
        print("Checking if input files exist...")
        for _, row in df.iterrows():
            file_path = Path(row['input_path'])
            if not file_path.exists():
                missing_files.append(row['input_path'])
        
        if missing_files:
            print(f"⚠️  Found {len(missing_files)} entries with missing files")
            print("First 5 missing files:")
            for f in missing_files[:5]:
                print(f"  - {f}")
        else:
            print("✅ All input files exist")
        
        # Summary by site
        print("\n" + "="*70)
        print("SUMMARY BY SITE")
        print("="*70)
        for site in df['site'].unique():
            site_data = df[df['site'] == site]
            n_subjects = site_data['subject_id'].nunique()
            n_runs = len(site_data)
            avg_runs = n_runs / n_subjects if n_subjects > 0 else 0
            
            print(f"\n{site}:")
            print(f"  Subjects: {n_subjects}")
            print(f"  Total runs: {n_runs}")
            print(f"  Avg runs/subject: {avg_runs:.1f}")
        
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and clean metadata CSV")
    parser.add_argument(
        '--fix', 
        action='store_true',
        help='Automatically fix duplicates (creates backup first)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/raw/subjects_metadata.csv',
        help='Path to metadata CSV (default: data/raw/subjects_metadata.csv)'
    )
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata)
    success = validate_and_clean_metadata(metadata_path, fix=args.fix)
    
    if not success and not args.fix:
        print("\n" + "="*70)
        print("To fix duplicates, run:")
        print(f"  python {__file__} --fix")
        print("="*70)

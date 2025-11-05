"""Check Brown preprocessed data for duplicates and errors"""
import nibabel as nib
import pandas as pd
from pathlib import Path

brown_dir = Path('data/preprocessed/Brown')

print("="*70)
print("BROWN PREPROCESSED DATA VALIDATION")
print("="*70)

subjects = []
for subj_dir in sorted(brown_dir.glob('sub-*')):
    func_file = subj_dir / 'func_preproc.nii.gz'
    mask_file = subj_dir / 'mask.nii.gz'
    
    if func_file.exists() and mask_file.exists():
        img = nib.load(str(func_file))
        shape = img.shape
        timepoints = shape[3]
        file_size_mb = func_file.stat().st_size / (1024 * 1024)
        
        subjects.append({
            'subject': subj_dir.name,
            'timepoints': timepoints,
            'shape': f"{shape[0]}x{shape[1]}x{shape[2]}",
            'size_mb': round(file_size_mb, 1)
        })

df = pd.DataFrame(subjects)

print(f"\n✅ Successfully preprocessed: {len(df)} subjects\n")

print("Summary Statistics:")
print("-" * 70)
print(f"Timepoints:")
print(f"  Min: {df['timepoints'].min()}")
print(f"  Max: {df['timepoints'].max()}")
print(f"  Mean: {df['timepoints'].mean():.1f}")
print(f"  Median: {df['timepoints'].median():.1f}")
print(f"\nFile sizes (MB):")
print(f"  Min: {df['size_mb'].min()}")
print(f"  Max: {df['size_mb'].max()}")
print(f"  Mean: {df['size_mb'].mean():.1f}")

print(f"\nAll subjects:")
print("-" * 70)
print(df.to_string(index=False))

# Check for duplicates
print("\n" + "="*70)
print("DUPLICATE CHECK")
print("="*70)

expected_timepoints = 251  # Single run, no duplicates
duplicate_timepoints = 502  # Would indicate duplicates

if df['timepoints'].max() > 400:
    print("❌ DUPLICATES DETECTED!")
    duplicated = df[df['timepoints'] > 400]
    print(f"\nSubjects with >400 timepoints (likely duplicates):")
    print(duplicated.to_string(index=False))
elif df['timepoints'].std() > 10:
    print("⚠️  WARNING: High variance in timepoints")
    print(f"Standard deviation: {df['timepoints'].std():.1f}")
    print("\nTimepoint distribution:")
    print(df['timepoints'].value_counts().sort_index())
else:
    print("✅ NO DUPLICATES DETECTED!")
    print(f"\nAll subjects have similar timepoints (mean: {df['timepoints'].mean():.1f})")
    print("This matches the expected single-run data (no concatenation)")

# Compare with metadata
print("\n" + "="*70)
print("METADATA COMPARISON")
print("="*70)

metadata = pd.read_csv('data/raw/subjects_metadata.csv')
brown_meta = metadata[metadata['site'] == 'Brown']

print(f"Metadata entries: {len(brown_meta)}")
print(f"Unique subjects in metadata: {brown_meta['subject_id'].nunique()}")
print(f"Preprocessed subjects: {len(df)}")

if len(df) == brown_meta['subject_id'].nunique():
    print("✅ All metadata subjects have been preprocessed")
else:
    print(f"⚠️  Mismatch: {brown_meta['subject_id'].nunique() - len(df)} subjects missing")

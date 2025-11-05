"""
Check WashU preprocessing integrity
Validates preprocessed outputs against expected values from raw data
"""
import pandas as pd
from pathlib import Path

# Load the integrity report
report_path = Path("data/preprocessed/integrity_report.csv")

if not report_path.exists():
    print("❌ Integrity report not found. Run check_preprocessed_integrity.py first.")
    exit(1)

df = pd.read_csv(report_path)

# Filter for WashU only
washu_df = df[df['site'] == 'WashU'].copy()

print("="*70)
print("WASHU INTEGRITY REPORT")
print("="*70)

# Overall stats
total = len(washu_df)
valid = len(washu_df[washu_df['preproc_valid'] == True])
invalid = total - valid

print(f"\nTotal WashU subjects: {total}")
print(f"Valid preprocessed: {valid}")
print(f"Invalid/Missing: {invalid}")

# Timepoint analysis
if valid > 0:
    valid_subjects = washu_df[washu_df['preproc_valid'] == True]
    
    print(f"\nExpected timepoints (from raw files):")
    print(f"  Min: {valid_subjects['expected_timepoints'].min():.0f}")
    print(f"  Max: {valid_subjects['expected_timepoints'].max():.0f}")
    print(f"  Mean: {valid_subjects['expected_timepoints'].mean():.1f}")
    
    print(f"\nPreprocessed timepoints:")
    print(f"  Min: {valid_subjects['preproc_timepoints'].min():.0f}")
    print(f"  Max: {valid_subjects['preproc_timepoints'].max():.0f}")
    print(f"  Mean: {valid_subjects['preproc_timepoints'].mean():.1f}")

# File existence check
func_exists = washu_df['func_exists'].sum()
mask_exists = washu_df['mask_exists'].sum()

print(f"\nFile existence:")
print(f"  func_preproc.nii.gz: {func_exists}/{total}")
print(f"  mask.nii.gz: {mask_exists}/{total}")

# Duplicate detection
duplicate_subjects = washu_df[washu_df['issues'].str.contains('duplicate', case=False, na=False)]
if len(duplicate_subjects) > 0:
    print(f"\n❌ {len(duplicate_subjects)} subjects with DUPLICATE timepoints detected!")
    print("\nSubjects with duplicates:")
    for _, row in duplicate_subjects.iterrows():
        print(f"  {row['subject_id']}: expected {row['expected_timepoints']:.0f}, got {row['preproc_timepoints']:.0f}")
else:
    print(f"\n✅ No duplicate timepoints detected")

# Issues summary
issues_df = washu_df[washu_df['preproc_valid'] == False]
if len(issues_df) > 0:
    print(f"\nSubjects with issues ({len(issues_df)} subjects):")
    print(issues_df[['subject_id', 'expected_timepoints', 'preproc_timepoints', 'issues']].to_string(index=False))
else:
    print(f"\n✅ All subjects are valid!")

# File size statistics (for valid subjects only)
if valid > 0:
    valid_func = valid_subjects[valid_subjects['func_exists'] == True]
    valid_mask = valid_subjects[valid_subjects['mask_exists'] == True]
    
    if len(valid_func) > 0:
        print(f"\nFile sizes:")
        print(f"  func_preproc.nii.gz: {valid_func['func_size_mb'].min():.1f} - {valid_func['func_size_mb'].max():.1f} MB (avg: {valid_func['func_size_mb'].mean():.1f} MB)")
    
    if len(valid_mask) > 0:
        print(f"  mask.nii.gz: {valid_mask['mask_size_kb'].min():.1f} - {valid_mask['mask_size_kb'].max():.1f} KB (avg: {valid_mask['mask_size_kb'].mean():.1f} KB)")

print("\n" + "="*70)

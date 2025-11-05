"""Check NeuroIMAGE preprocessed data integrity"""
import pandas as pd

df = pd.read_csv('data/preprocessed/integrity_report.csv')
neuro = df[df['site'] == 'NeuroIMAGE'].copy()

print('='*70)
print('NEUROIMAGE INTEGRITY REPORT')
print('='*70)

print(f'\nTotal NeuroIMAGE subjects: {len(neuro)}')
print(f'Valid preprocessed: {neuro["preproc_valid"].sum()}')
print(f'Invalid/Missing: {(~neuro["preproc_valid"]).sum()}')

print(f'\nExpected timepoints (from raw files):')
print(f'  Min: {neuro["expected_timepoints"].min()}')
print(f'  Max: {neuro["expected_timepoints"].max()}')
print(f'  Mean: {neuro["expected_timepoints"].mean():.1f}')

if neuro["preproc_timepoints"].notna().any():
    valid_preproc = neuro[neuro["preproc_timepoints"].notna()]
    print(f'\nPreprocessed timepoints:')
    print(f'  Min: {valid_preproc["preproc_timepoints"].min():.0f}')
    print(f'  Max: {valid_preproc["preproc_timepoints"].max():.0f}')
    print(f'  Mean: {valid_preproc["preproc_timepoints"].mean():.1f}')

print(f'\nFile existence:')
print(f'  func_preproc.nii.gz: {neuro["func_exists"].sum()}/{len(neuro)}')
print(f'  mask.nii.gz: {neuro["mask_exists"].sum()}/{len(neuro)}')

# Check for duplicates
if neuro["preproc_timepoints"].notna().any():
    duplicates = neuro[neuro["preproc_timepoints"] > neuro["expected_timepoints"] * 1.5]
    if not duplicates.empty:
        print(f'\n⚠️  Subjects with DUPLICATE timepoints: {len(duplicates)}')
        print(duplicates[['subject_id','expected_timepoints','preproc_timepoints']].to_string(index=False))
    else:
        print(f'\n✅ No duplicate timepoints detected')

# Show issues
issues = neuro[neuro['issues'] != '']
if not issues.empty:
    print(f'\nSubjects with issues ({len(issues)} total):')
    print(issues[['subject_id','expected_timepoints','preproc_timepoints','issues']].to_string(index=False))
else:
    print('\n✅ All NeuroIMAGE subjects are valid!')

# File sizes
if neuro["func_size_mb"].notna().any():
    print(f'\nFile sizes:')
    print(f'  func_preproc.nii.gz: {neuro["func_size_mb"].min():.1f} - {neuro["func_size_mb"].max():.1f} MB (avg: {neuro["func_size_mb"].mean():.1f} MB)')
    print(f'  mask.nii.gz: {neuro["mask_size_kb"].min():.1f} - {neuro["mask_size_kb"].max():.1f} KB (avg: {neuro["mask_size_kb"].mean():.1f} KB)')

"""
Check preprocessed outputs for existence, loadability, duplicates and invalid outputs.

Produces: data/preprocessed/integrity_report.csv with columns:
- site, subject_id, expected_timepoints, preproc_timepoints, func_exists, mask_exists,
  func_size_mb, mask_size_kb, mask_nonzero_voxels, preproc_valid, issue

Usage:
  & "D:\repos\DATASCII_10_ADHD\thesis-adhd\Scripts\python.exe" scripts/check_preprocessed_integrity.py

This script expects the virtualenv to have pandas and nibabel installed.
"""

import sys
from pathlib import Path
import csv

try:
    import pandas as pd
    import nibabel as nib
    import numpy as np
except Exception as e:
    print("ERROR: missing dependency:", e)
    print("Run with the project's virtualenv Python where pandas and nibabel are installed.")
    sys.exit(2)

ROOT = Path.cwd()
META_PATH = ROOT / "data" / "raw" / "subjects_metadata.csv"
PREPROC_ROOT = ROOT / "data" / "preprocessed"
REPORT_PATH = PREPROC_ROOT / "integrity_report.csv"

if not META_PATH.exists():
    print(f"Metadata not found at {META_PATH}")
    sys.exit(1)

print("Loading metadata...")
df_meta = pd.read_csv(META_PATH)

# Ensure run and input_path are strings
if 'run' in df_meta.columns:
    df_meta['run'] = df_meta['run'].astype(str)

# Group runs by subject and site
subjects = []
for (site, subject), group in df_meta.groupby(['site', 'subject_id']):
    # unique input paths (dedup)
    unique_paths = group['input_path'].drop_duplicates().tolist()
    subjects.append({'site': site, 'subject_id': subject, 'raw_paths': unique_paths})

rows = []
print(f"Found {len(subjects)} subjects in metadata. Checking preprocessed outputs...")
for s in subjects:
    site = s['site']
    subject = s['subject_id']
    raw_paths = s['raw_paths']

    preproc_dir = PREPROC_ROOT / site / subject
    func_file = preproc_dir / 'func_preproc.nii.gz'
    mask_file = preproc_dir / 'mask.nii.gz'

    func_exists = func_file.exists()
    mask_exists = mask_file.exists()
    func_size_mb = None
    mask_size_kb = None
    preproc_timepoints = None
    mask_nonzero = None
    issues = []

    # expected timepoints = sum of timepoints of raw unique files
    expected_tp = 0
    for p in raw_paths:
        ppath = Path(p)
        if not ppath.exists():
            # try relative to repo root
            p2 = ROOT / p
            if p2.exists():
                ppath = p2
        try:
            if ppath.exists():
                img = nib.load(str(ppath))
                expected_tp += int(img.shape[3])
            else:
                issues.append('raw_missing')
        except Exception as e:
            issues.append('raw_load_error')

    if func_exists:
        try:
            func_size_mb = func_file.stat().st_size / (1024*1024)
            img = nib.load(str(func_file))
            preproc_timepoints = int(img.shape[3]) if len(img.shape) >= 4 else 0
        except Exception as e:
            issues.append('func_load_error')
    else:
        issues.append('func_missing')

    if mask_exists:
        try:
            mask_size_kb = mask_file.stat().st_size / 1024
            mimg = nib.load(str(mask_file))
            mdata = mimg.get_fdata()
            mask_nonzero = int(np.count_nonzero(mdata))
            # Consider tiny masks invalid
            if mask_nonzero < 100:
                issues.append('mask_too_small')
        except Exception as e:
            issues.append('mask_load_error')
    else:
        issues.append('mask_missing')

    # detect duplicates: preproc_timepoints much greater than expected
    if preproc_timepoints is not None and expected_tp > 0:
        if preproc_timepoints >= expected_tp * 1.5:
            issues.append('possible_duplicate_timepoints')
        elif preproc_timepoints < expected_tp:
            issues.append('incomplete_concatenation')

    preproc_valid = (len([x for x in issues if x not in ('raw_missing',)]) == 0) and func_exists and mask_exists
    # raw_missing not considered preproc invalid directly but noted

    rows.append({
        'site': site,
        'subject_id': subject,
        'expected_timepoints': expected_tp,
        'preproc_timepoints': preproc_timepoints,
        'func_exists': func_exists,
        'mask_exists': mask_exists,
        'func_size_mb': round(func_size_mb,1) if func_size_mb is not None else None,
        'mask_size_kb': round(mask_size_kb,1) if mask_size_kb is not None else None,
        'mask_nonzero_voxels': mask_nonzero,
        'preproc_valid': preproc_valid,
        'issues': ';'.join(issues) if issues else ''
    })

# Write report CSV
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Summarize
df_report = pd.DataFrame(rows)
total = len(df_report)
valid = df_report['preproc_valid'].sum()
invalid = total - valid
possible_dup = df_report['issues'].str.contains('possible_duplicate_timepoints').sum()
raw_missing = df_report['issues'].str.contains('raw_missing').sum()

print('\nIntegrity check complete')
print(f'Total subjects checked: {total}')
print(f'Valid preprocessed subjects: {valid}')
print(f'Invalid / issues: {invalid}')
print(f'Subjects with possible duplicate timepoints: {possible_dup}')
print(f'Subjects with raw files missing: {raw_missing}')
print(f'Report written to: {REPORT_PATH}')

# Print list of problematic subjects (first 30)
problems = df_report[df_report['preproc_valid'] == False]
if not problems.empty:
    print('\nList of problematic subjects (first 30):')
    print(problems[['site','subject_id','issues']].head(30).to_string(index=False))
else:
    print('\nNo problematic preprocessed subjects found.')

# Exit code 0 for success
sys.exit(0)

"""
Check WashU data structure - multiple runs per subject
"""
import pandas as pd
from pathlib import Path

METADATA_PATH = Path("data/raw/subjects_metadata.csv")

print("="*70)
print("WashU DATA STRUCTURE ANALYSIS")
print("="*70)

metadata = pd.read_csv(METADATA_PATH)
washu = metadata[metadata['site'] == 'WashU']

print(f"\n📊 Total WashU entries in metadata: {len(washu)}")
print(f"📊 Unique WashU subjects: {washu['subject_id'].nunique()}")
print(f"📊 Runs per subject: {len(washu) / washu['subject_id'].nunique():.1f} average")

print("\n" + "="*70)
print("RUNS PER SUBJECT BREAKDOWN")
print("="*70)

runs_per_subject = washu.groupby('subject_id').size()
print(f"\nDistribution of runs per subject:")
print(runs_per_subject.value_counts().sort_index())

print("\n" + "="*70)
print("SAMPLE: First 3 subjects with their runs")
print("="*70)

for subject_id in washu['subject_id'].unique()[:3]:
    subject_runs = washu[washu['subject_id'] == subject_id]
    print(f"\n{subject_id}: {len(subject_runs)} runs")
    for _, row in subject_runs.iterrows():
        filename = Path(row['full_path']).name
        print(f"  - Run {row['run']}: {filename}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("For preprocessing, you should:")
print("  1. Use only run-1 for each subject (60 subjects)")
print("  2. OR concatenate all runs per subject (more data per subject)")
print("  3. The preprocess_washu.py script is now configured to use run-1 only")
print("\nThis will give you 60 WashU subjects for analysis.")
print("="*70)

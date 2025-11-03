import pandas as pd

metadata = pd.read_csv('data/raw/subjects_metadata.csv')
washu = metadata[metadata['site'] == 'WashU']

print(f"Total WashU entries: {len(washu)}")
print(f"Unique WashU subjects: {washu['subject_id'].nunique()}")

# The run column might be read as string or int
print(f"\nRun column data type: {washu['run'].dtype}")
print(f"Run values: {sorted(washu['run'].unique())}")

# Filter to run == 1 (handle both int and string)
washu_run1 = washu[washu['run'] == 1]
print(f"\n✅ WashU entries with run==1: {len(washu_run1)}")
print(f"✅ Unique subjects with run==1: {washu_run1['subject_id'].nunique()}")

# Also try string comparison
washu_run1_str = washu[washu['run'].astype(str) == '1']
print(f"\n✅ WashU entries with run=='1' (string): {len(washu_run1_str)}")
print(f"✅ Unique subjects with run=='1' (string): {washu_run1_str['subject_id'].nunique()}")

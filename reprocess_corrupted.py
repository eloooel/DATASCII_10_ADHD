# reprocess_corrupted.py
import pandas as pd
from pathlib import Path

# List of corrupted subjects
corrupted_subjects = ["sub-1502229"]  # Add more if found

# Load metadata
metadata = pd.read_csv("data/raw/subjects_metadata.csv")

# Filter for corrupted subjects
corrupted_data = metadata[metadata['subject_id'].isin(corrupted_subjects)]

print(f"Re-processing {len(corrupted_data)} corrupted subjects:")
for _, row in corrupted_data.iterrows():
    print(f"  - {row['site']}/{row['subject_id']}")

# Re-run preprocessing for these subjects only
from preprocessing.preprocess import _process_subject

for _, row in corrupted_data.iterrows():
    print(f"\n🔄 Re-processing {row['subject_id']}")
    row_dict = row.to_dict()
    row_dict['out_dir'] = "data/preprocessed"
    row_dict['device'] = 'cpu'
    
    result = _process_subject(row_dict)
    print(f"Result: {result['status']}")
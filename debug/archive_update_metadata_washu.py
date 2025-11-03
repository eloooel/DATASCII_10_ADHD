"""
Update subjects_metadata.csv to include WashU subjects
"""
import pandas as pd
from pathlib import Path
from utils import DataDiscovery

RAW_DIR = Path("./data/raw")
METADATA_PATH = RAW_DIR / "subjects_metadata.csv"

print("="*70)
print("UPDATING METADATA TO INCLUDE WashU")
print("="*70)

# Load existing metadata
existing_metadata = pd.read_csv(METADATA_PATH)
print(f"\n📊 Existing metadata: {len(existing_metadata)} subjects")
print(existing_metadata['site'].value_counts())

# Discover WashU subjects
print("\n🔍 Discovering WashU subjects...")
washu_dir = RAW_DIR / "WashU"

if not washu_dir.exists():
    print(f"❌ WashU directory not found: {washu_dir}")
    exit(1)

# Use DataDiscovery to find WashU subjects
discovery = DataDiscovery(RAW_DIR)
all_subjects = discovery.discover_subjects()

# Filter for WashU only
washu_subjects_df = pd.DataFrame([s for s in all_subjects if s['site'] == 'WashU'])

if len(washu_subjects_df) == 0:
    print("❌ No WashU subjects discovered!")
    exit(1)

print(f"✅ Found {len(washu_subjects_df)} WashU subjects")
print(f"\nSample subjects:")
print(washu_subjects_df[['subject_id', 'site']].head() if len(washu_subjects_df) > 0 else "No subjects")

# Combine with existing metadata (avoid duplicates)
existing_ids = set(existing_metadata['subject_id'].tolist())
new_subjects = washu_subjects_df[~washu_subjects_df['subject_id'].isin(existing_ids)]

print(f"\n➕ Adding {len(new_subjects)} new WashU subjects to metadata")

# Merge
updated_metadata = pd.concat([existing_metadata, new_subjects], ignore_index=True)

# Save updated metadata
METADATA_PATH_BACKUP = RAW_DIR / "subjects_metadata_backup.csv"
existing_metadata.to_csv(METADATA_PATH_BACKUP, index=False)
print(f"💾 Backup saved to: {METADATA_PATH_BACKUP}")

updated_metadata.to_csv(METADATA_PATH, index=False)
print(f"💾 Updated metadata saved to: {METADATA_PATH}")

print("\n" + "="*70)
print("UPDATED METADATA SUMMARY")
print("="*70)
print(f"Total subjects: {len(updated_metadata)}")
print(f"\nSites:")
print(updated_metadata['site'].value_counts())
print(f"\nDiagnosis distribution (WashU only):")
washu_df = updated_metadata[updated_metadata['site'] == 'WashU']
print(washu_df['diagnosis'].value_counts())

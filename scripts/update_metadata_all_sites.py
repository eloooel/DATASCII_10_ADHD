"""
Discover and add all new sites to metadata
Sites with multiple runs will have all runs listed
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from utils.data_loader import DataDiscovery

RAW_DIR = Path("./data/raw")
METADATA_PATH = RAW_DIR / "subjects_metadata.csv"

print("="*70)
print("DISCOVERING ALL SITES AND UPDATING METADATA")
print("="*70)

# Load existing metadata
if METADATA_PATH.exists():
    existing_metadata = pd.read_csv(METADATA_PATH)
    print(f"\n📊 Existing metadata: {len(existing_metadata)} entries")
    print(existing_metadata['site'].value_counts())
    existing_ids_runs = set(zip(existing_metadata['subject_id'], existing_metadata['run']))
else:
    existing_metadata = pd.DataFrame()
    existing_ids_runs = set()
    print("\n⚠️  No existing metadata found, creating new")

# Discover all subjects from all sites
print("\n🔍 Discovering subjects from all sites...")
discovery = DataDiscovery(RAW_DIR)
all_subjects = discovery.discover_subjects()

# Convert to DataFrame
all_subjects_df = pd.DataFrame(all_subjects)

if len(all_subjects_df) == 0:
    print("❌ No subjects discovered!")
    exit(1)

print(f"\n✅ Discovered {len(all_subjects_df)} total entries across all sites")
print(f"\nSites discovered:")
print(all_subjects_df['site'].value_counts())

# Check which sites have multiple runs
print("\n" + "="*70)
print("MULTI-RUN ANALYSIS")
print("="*70)

for site in all_subjects_df['site'].unique():
    site_data = all_subjects_df[all_subjects_df['site'] == site]
    n_subjects = site_data['subject_id'].nunique()
    n_entries = len(site_data)
    avg_runs = n_entries / n_subjects if n_subjects > 0 else 0
    
    # Check if has run column and multiple runs
    if 'run' in site_data.columns:
        max_run = site_data['run'].max()
        print(f"\n{site}:")
        print(f"  - Subjects: {n_subjects}")
        print(f"  - Total entries: {n_entries}")
        print(f"  - Avg runs/subject: {avg_runs:.1f}")
        print(f"  - Max run number: {max_run}")
        
        if avg_runs > 1.1:
            print(f"  ✅ MULTI-RUN SITE - will concatenate runs per subject")
        else:
            print(f"  ℹ️  Single run per subject")
    else:
        print(f"\n{site}: {n_subjects} subjects (no run info)")

# Filter out entries that already exist (same subject_id AND run)
if len(existing_metadata) > 0:
    new_subjects = all_subjects_df[
        ~all_subjects_df.apply(lambda row: (row['subject_id'], row['run']) in existing_ids_runs, axis=1)
    ]
else:
    new_subjects = all_subjects_df

print(f"\n" + "="*70)
print(f"METADATA UPDATE")
print(f"="*70)
print(f"➕ Adding {len(new_subjects)} new entries to metadata")

if len(new_subjects) == 0:
    print("✅ Metadata is already up to date!")
    exit(0)

# Show new entries by site
print(f"\nNew entries by site:")
print(new_subjects['site'].value_counts())

# Merge with existing metadata
if len(existing_metadata) > 0:
    updated_metadata = pd.concat([existing_metadata, new_subjects], ignore_index=True)
    
    # Backup existing
    METADATA_PATH_BACKUP = RAW_DIR / "subjects_metadata_backup.csv"
    existing_metadata.to_csv(METADATA_PATH_BACKUP, index=False)
    print(f"\n💾 Backup saved to: {METADATA_PATH_BACKUP}")
else:
    updated_metadata = new_subjects

# Save updated metadata
updated_metadata.to_csv(METADATA_PATH, index=False)
print(f"💾 Updated metadata saved to: {METADATA_PATH}")

print("\n" + "="*70)
print("FINAL METADATA SUMMARY")
print("="*70)
print(f"Total entries: {len(updated_metadata)}")
print(f"Total unique subjects: {updated_metadata['subject_id'].nunique()}")

print(f"\nEntries by site:")
print(updated_metadata['site'].value_counts())

print(f"\nDiagnosis distribution:")
print(updated_metadata['diagnosis'].value_counts())

# Show multi-run sites
print("\n" + "="*70)
print("MULTI-RUN SITES (for concatenation)")
print("="*70)
for site in updated_metadata['site'].unique():
    site_data = updated_metadata[updated_metadata['site'] == site]
    n_subjects = site_data['subject_id'].nunique()
    n_entries = len(site_data)
    avg_runs = n_entries / n_subjects
    
    if avg_runs > 1.1:
        print(f"\n{site}:")
        print(f"  - {n_subjects} subjects with {n_entries} total runs")
        print(f"  - Average {avg_runs:.1f} runs per subject")
        print(f"  ✅ Will concatenate runs during preprocessing")

print("\n" + "="*70)
print("✅ METADATA UPDATE COMPLETE")
print("="*70)

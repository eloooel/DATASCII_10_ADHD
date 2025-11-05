# Preventing Duplicate Metadata Entries

## The Problem

When preprocessing is interrupted and the metadata update script (`update_metadata_all_sites.py`) is run multiple times, duplicate entries could be added to `subjects_metadata.csv`. This causes:
- ❌ Same files processed multiple times during preprocessing
- ❌ Doubled file sizes (duplicate timepoints concatenated)
- ❌ Invalid training data
- ❌ Wasted computational resources

## Safeguards Implemented

### 1. Enhanced Deduplication in Update Script

**Location**: `scripts/update_metadata_all_sites.py`

The script now has three layers of protection:

#### Layer 1: Strict Key Matching
```python
# Uses (subject_id, run, input_path) instead of just (subject_id, run)
existing_keys = set(zip(
    existing_metadata['subject_id'], 
    existing_metadata['run'],
    existing_metadata['input_path']
))
```

#### Layer 2: Data Type Normalization
```python
# Ensures run column is always string for consistent comparison
existing_metadata['run'] = existing_metadata['run'].astype(str)
all_subjects_df['run'] = all_subjects_df['run'].astype(str)
```

#### Layer 3: Final Safety Check
```python
# Removes any duplicates that slipped through before saving
updated_metadata = updated_metadata.drop_duplicates(
    subset=['subject_id', 'run', 'input_path'], 
    keep='first'
)
```

### 2. Validation Script

**Location**: `scripts/validate_metadata.py`

Run this anytime to check for and fix duplicates:

```bash
# Check for duplicates (no changes)
python scripts/validate_metadata.py

# Fix duplicates automatically (creates backup first)
python scripts/validate_metadata.py --fix
```

## Best Practices

### Before Starting Preprocessing

1. **Always validate metadata first**:
   ```bash
   python scripts/validate_metadata.py --fix
   ```

2. **Check the summary** to ensure correct run counts:
   - Brown: Should be ~1.0 runs/subject
   - NeuroIMAGE: Should be ~1.0 runs/subject
   - WashU: Should be ~2.8 runs/subject
   - NYU: Should be ~1.7 runs/subject

### If Preprocessing is Interrupted

1. **Don't panic!** The preprocessing script has skip logic that checks for existing files.

2. **Don't re-run the metadata update script** - it's already up to date.

3. **Just restart preprocessing**:
   ```bash
   python scripts/preprocess_all_sites.py --sites WashU --batch-size 2
   ```
   
   The script will automatically:
   - ✅ Skip subjects that are already preprocessed
   - ✅ Resume from where it left off
   - ✅ Only process remaining subjects

### If You Must Update Metadata Again

1. **Validate first** to see current state:
   ```bash
   python scripts/validate_metadata.py
   ```

2. **Run update script** (now safe with new safeguards):
   ```bash
   python scripts/update_metadata_all_sites.py
   ```

3. **Validate again** to confirm no duplicates:
   ```bash
   python scripts/validate_metadata.py --fix
   ```

## What Changed

### Before (Vulnerable to Duplicates)
- Checked only `(subject_id, run)` - could miss duplicates if run numbers matched
- No data type normalization - string "1" ≠ int 1
- No final safety check before saving
- No validation tool

### After (Protected Against Duplicates)
- ✅ Checks `(subject_id, run, input_path)` - unique file path prevents duplicates
- ✅ Normalizes run column to string for consistent comparison
- ✅ Final deduplication before saving
- ✅ Standalone validation tool for checking anytime
- ✅ Automatic backup creation before any fixes

## Current Dataset Status

After fixing duplicates, the clean metadata shows:

```
Total entries: 1048 runs
Total subjects: 759 unique subjects

Site breakdown:
- NYU: 257 subjects, 437 runs (1.7 avg)
- WashU: 60 subjects, 169 runs (2.8 avg)
- Peking_1: 136 subjects, 136 runs (1.0 avg)
- Pittsburgh: 98 subjects, 98 runs (1.0 avg)
- NeuroIMAGE: 73 subjects, 73 runs (1.0 avg)
- Peking_2: 67 subjects, 67 runs (1.0 avg)
- Peking_3: 42 subjects, 42 runs (1.0 avg)
- Brown: 26 subjects, 26 runs (1.0 avg)
```

## Quick Reference Commands

```bash
# Validate metadata (check only)
python scripts/validate_metadata.py

# Fix duplicates
python scripts/validate_metadata.py --fix

# Update metadata (safe - now has built-in deduplication)
python scripts/update_metadata_all_sites.py

# Start/resume preprocessing (has built-in skip logic)
python scripts/preprocess_all_sites.py --sites WashU --batch-size 2
```

## Troubleshooting

**Q: I see duplicate entries in metadata. What do I do?**
```bash
python scripts/validate_metadata.py --fix
```

**Q: I interrupted preprocessing. Can I restart?**
Yes! Just run the same command again. Already-processed subjects will be skipped.

**Q: How do I know if metadata is clean?**
```bash
python scripts/validate_metadata.py
```
Should show "✅ No duplicates found - metadata is clean!"

**Q: What if I accidentally run update_metadata_all_sites.py twice?**
The new safeguards will prevent duplicates, but you can still run:
```bash
python scripts/validate_metadata.py --fix
```
to be safe.

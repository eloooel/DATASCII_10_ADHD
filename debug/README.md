# Debug & One-Time Scripts

This directory contains diagnostic and one-time utility scripts that were used during development.

## Active Diagnostic Tools

### `washu_diagnostics.py`
Consolidated diagnostic script for WashU data analysis.

**Functions:**
- `check_structure()` - Analyze WashU multi-run structure
- `check_preprocessing_status()` - Check preprocessing completion status
- `debug_filtering()` - Debug metadata filtering issues

**Usage:**
```bash
# Check WashU data structure
python debug/washu_diagnostics.py --structure

# Check preprocessing status
python debug/washu_diagnostics.py --status

# Debug filtering
python debug/washu_diagnostics.py --filter

# Run all diagnostics
python debug/washu_diagnostics.py --all
```

## Archived Scripts (Completed One-Time Operations)

These scripts have been consolidated into `washu_diagnostics.py` and archived:

- `archive_check_washu_structure.py` - Original WashU structure analysis (now: --structure)
- `archive_debug_washu_filter.py` - Original filtering debug (now: --filter)  
- `archive_check_preprocessing.py` - Original status check (now: --status)
- `archive_update_metadata_washu.py` - One-time metadata update (COMPLETED)

**Note:** The archived scripts are kept for reference but should not be used. Use `washu_diagnostics.py` instead.


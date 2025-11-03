# Scripts Directory

This directory contains utility scripts for specific tasks in the ADHD classification pipeline.

## Active Scripts

### `run_washu_preprocessing.py`
Preprocess only WashU subjects (60 subjects using first run per subject).

**Usage:**
```bash
# Run with parallel processing (default)
python scripts/run_washu_preprocessing.py

# Run sequentially (for debugging)
python scripts/run_washu_preprocessing.py --no-parallel
```

**Purpose:** One-time preprocessing of WashU dataset that was added after initial pipeline setup.

## Build Scripts

- `build.sh` - Docker build script
- `run-*.sh` - Various pipeline execution scripts

## Notes

- For main pipeline execution, use `python main.py`
- For WashU-specific preprocessing, use `scripts/run_washu_preprocessing.py`
- For diagnostics, see `debug/washu_diagnostics.py`

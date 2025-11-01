# DATASCII_10_ADHD

## ⚠️ CRITICAL REQUIREMENT: Python 3.10.10

**You MUST use Python 3.10.10 for this project. Other versions (especially 3.11+) will cause compilation errors.**

Check your Python version:
```
python --version
```

If you see Python 3.13 or any version other than 3.10.x, download and install Python 3.10.10 from:
https://www.python.org/downloads/release/python-31010/

**Windows Direct Download:**
- 64-bit: https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe
- 32-bit: https://www.python.org/ftp/python/3.10.10/python-3.10.10.exe

After installing Python 3.10.10, use it explicitly:
- Windows: `py -3.10 -m venv .venv` or `C:\Path\To\Python310\python.exe -m venv .venv`
- Linux/Mac: Use `python3.10` instead of `python`

**IMPORTANT**: When you activate your venv, verify it's using Python 3.10:
```
python --version
```
It MUST show "Python 3.10.x" (where x is any patch version).

## Data Access

This project uses the ADHD-200 dataset. Due to size and licensing restrictions, the raw data is **not included** in this repository.

Please download the dataset from [ADHD-200 Preprocessed](http://fcon_1000.projects.nitrc.org/indi/adhd200/)

=========================================================================================

Instructions for Running:

Run preprocessing only:

python main.py --stage preprocessing

Run preprocessing + feature extraction:

python main.py --stage features

Run full pipeline (all stages):

python main.py --stage full

**Retry Failed Preprocessing:**

If some subjects fail during preprocessing, you can retry them without reprocessing successful subjects:

Option 1: Automatic retry (identifies all failed subjects):
```bash
python preprocessing/retry_failed.py --manifest data/raw/subjects_metadata.csv --output data/preprocessed --jobs 2
```

Option 2: Retry without cleanup (keep existing files):
```bash
python preprocessing/retry_failed.py --manifest data/raw/subjects_metadata.csv --output data/preprocessed --no-cleanup --jobs 2
```

Option 3: Sequential processing (for memory-intensive subjects):
```bash
python preprocessing/retry_failed.py --manifest data/raw/subjects_metadata.csv --output data/preprocessed --jobs 1
```

The retry script will:
1. Scan the output directory for failed/corrupted subjects
2. Identify which subjects need reprocessing
3. Clean up corrupted files (unless --no-cleanup is specified)
4. Reprocess only the failed subjects
5. Save results to `data/preprocessed/retry_results.csv`

**Manual retry for specific subjects:**
```python
# In a Python script or notebook
from pathlib import Path
import pandas as pd
from preprocessing.preprocess import _process_subject
from utils import run_parallel

# Load original manifest
manifest = pd.read_csv("data/raw/subjects_metadata.csv")

# Filter to specific failed subjects
failed_ids = ["sub-0027017", "sub-0027016", "sub-3304956"]
retry_df = manifest[manifest['subject_id'].isin(failed_ids)].copy()
retry_df['force_retry'] = True
retry_df['out_dir'] = "data/preprocessed"

# Process with reduced parallelism for memory issues
results = run_parallel(_process_subject, retry_df.to_dict('records'), n_jobs=2)

# Or process one at a time for problematic subjects
for subject_id in failed_ids:
    row = manifest[manifest['subject_id'] == subject_id].iloc[0].to_dict()
    row['force_retry'] = True
    row['out_dir'] = "data/preprocessed"
    result = _process_subject(row)
    print(f"{subject_id}: {result['status']}")
```

**Common failure causes and solutions:**

1. **Memory errors** (e.g., "Unable to allocate X GiB"):
   - Use `--jobs 1` for sequential processing
   - Close other applications to free RAM
   - The pipeline automatically reduces ICA components when memory is low

2. **Corrupted output files**:
   - The retry script automatically cleans up and retries
   - Files are verified after saving to catch corruption early

3. **File verification failures**:
   - Usually indicates incomplete writes due to system issues
   - Retry with `--jobs 1` to reduce I/O load

=========================================================================================

Instructions for downloading dependencies:

1. Input in the terminal

   # Install core dependencies

   uv sync

   # Install PyTorch 2.2.0 (compatible with your code)

   uv run pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

   # Install PyTorch Geometric with pre-compiled wheels

   uv run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

   ***

   ## 🐳 Docker Commands with uv

### Build

```bash
docker build -t adhd-gnn-stan:latest .
```

### Run with GPU support

```bash
docker run --gpus all --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    adhd-gnn-stan:latest \
    uv run python main.py --stage full
```

### Run CPU-only (memory-optimized)

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    --memory=16g \
    --cpus=8 \
    --env CUDA_VISIBLE_DEVICES="" \
    adhd-gnn-stan:latest \
    uv run python main.py --stage full --no-parallel --no-cuda
```

### Interactive mode with uv

```bash
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    adhd-gnn-stan:latest \
    /bin/bash

# Inside container, use uv run:
adhd_user@container:/app$ uv run python main.py --help
adhd_user@container:/app$ uv run python -c "import torch; print(torch.__version__)"
```

### Test PyTorch installation

```bash
docker run --rm adhd-gnn-stan:latest uv run python -c "
import torch, nibabel, pandas, numpy
print('✅ All dependencies working')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Run corruption checker

```bash
docker run --rm -v $(pwd)/data:/app/data adhd-gnn-stan:latest \
    uv run python check_preprocessing_integrity.py \
    --preprocessed-dir /app/data/preprocessed \
    --metadata-csv /app/data/raw/subjects_metadata.csv
```

For Preprocessing and Feature Extraction:
Individual Subject-level parallel execution

STRICT IMPLEMENTATION OF USING Python 3.10.10 for compatibility of libraries

=========================================================================================

Virtual environment setup (create venv and install dependencies)
1. **VERIFY you have Python 3.10.10 installed:**
   python --version
   
   If not 3.10.x, you MUST install Python 3.10.10 first (see above).

2. From the project root, create a virtual environment **using Python 3.10**:
   
   Windows - if you have multiple Python versions:
   py -3.10 -m venv .venv
   
   Windows - if Python 3.10 is your default:
   python -m venv .venv
   
   Linux/Mac:
   python3.10 -m venv .venv

3. Activate the virtual environment
   - Windows (Command Prompt):
     .venv\Scripts\activate

   - Windows (PowerShell):
     .venv\Scripts\Activate.ps1
     If execution is blocked, run (PowerShell as Admin):
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

4. **CRITICAL: Verify the venv is using Python 3.10:**
   python --version
   
   This MUST show "Python 3.10.x". If it shows Python 3.13 or anything else:
   - Deactivate: deactivate
   - Delete venv: Remove-Item -Recurse -Force .venv
   - Recreate with: py -3.10 -m venv .venv
   - Reactivate and verify again

5. **IMPORTANT: PyTorch Geometric packages are commented out in requirements.txt and must be installed separately.**

6. Upgrade pip and install requirements:
   python -m pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
   pip install -r requirements.txt

Quick one-line (Windows PowerShell) - ONLY if Python 3.10 is default:
   python -m venv .venv; .venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html; pip install -r requirements.txt

Recommended step-by-step (Windows with Python 3.10):
   py -3.10 -m venv .venv
   .venv\Scripts\Activate.ps1
   python --version  # VERIFY: Must show Python 3.10.x
   python -m pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
   pip install -r requirements.txt

Quick one-line (POSIX shells):
   python -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html && pip install -r requirements.txt

Verification:
   python -c "import sys; print(sys.executable)"
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   pip list

Troubleshooting:
- **ERROR: Cython.Compiler.Errors.CompileError with scikit-learn**: You're using the wrong Python version. 
  MUST use Python 3.10.10. Check with `python --version` INSIDE the activated venv.
  If wrong, delete .venv folder and recreate with: py -3.10 -m venv .venv
- **ERROR: Shows C:\Python313\... in error messages**: Your venv was created with Python 3.13.
  Delete .venv and recreate with Python 3.10.10 explicitly: py -3.10 -m venv .venv
- If requirements.txt contains torch==2.1.0+cu121, this version is no longer available in the cu121 index.
  Either remove the torch line from requirements.txt and install it separately (as shown above),
  or update requirements.txt to use a newer version without CUDA suffix (e.g., torch>=2.1.0).
- CUDA 11.8 (cu118) is more stable and widely supported than CUDA 12.1.
- For CPU-only installation, omit the --index-url flag: pip install torch torchvision torchaudio
- Check your CUDA version: nvidia-smi (if you have an NVIDIA GPU)

Notes:
- Ensure you use Python 3.10.10 as required.
- If CUDA wheels are not needed or your hardware is different, adjust the index-url / wheel options accordingly.
- The latest PyTorch version compatible with CUDA 11.8 will be installed automatically.
- CUDA 11.8 works with most modern NVIDIA GPUs and drivers.

# DATASCII_10_ADHD

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

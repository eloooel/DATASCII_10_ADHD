# Multi-stage build optimized for uv
FROM python:3.10.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Create virtual environment and install ALL dependencies using uv
RUN uv venv

# Install Python dependencies first
RUN uv pip install nibabel nilearn pandas numpy scikit-learn matplotlib seaborn lxml pillow

# Install PyTorch in the SAME virtual environment
RUN uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric in the SAME virtual environment  
RUN uv pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html

# ✅ Verify installations work using direct python (not uv run)
RUN .venv/bin/python -c "import torch, nibabel, pandas, numpy; print(f'Build verification: PyTorch {torch.__version__} installed successfully')"

# Production stage
FROM python:3.10.10-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv in production stage
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy the COMPLETE virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml ./

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/preprocessed data/features data/trained data/splits outputs/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash adhd_user && \
    chown -R adhd_user:adhd_user /app
USER adhd_user

# ✅ Test using direct python (not uv run during build)
RUN .venv/bin/python -c "import torch, nibabel, pandas, numpy; print('✅ Production verification: All dependencies loaded')"

# Health check using uv run (this works after app code is copied)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD uv run python -c "import torch, nibabel, pandas, numpy; print('✅ All dependencies loaded')" || exit 1

# Default command using uv run
CMD ["uv", "run", "python", "main.py", "--help"]
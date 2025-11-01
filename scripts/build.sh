#!/bin/bash
# Build script for ADHD classification pipeline with uv

set -e

echo "Building ADHD GNN-STAN Docker image with uv..."

# Build the image
docker build -t adhd-gnn-stan:latest .

echo "Build complete!"

# Show image size
echo "Image size:"
docker images | grep adhd-gnn-stan

# Test the build
echo ""
echo "🧪 Testing build..."
docker run --rm adhd-gnn-stan:latest uv run python -c "
import torch
print(f'✅ Build test passed')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "Usage examples (all with uv):"
echo "  ./scripts/run-preprocessing.sh    # Preprocessing only"
echo "  ./scripts/run-features.sh         # Feature extraction only"  
echo "  ./scripts/run-training.sh         # Training only"
echo "  ./scripts/run-full-pipeline.sh    # Complete pipeline"
echo "  ./scripts/run-debug.sh           # Debug mode"
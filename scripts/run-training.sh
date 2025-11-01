#!/bin/bash
# Run training stage in Docker with uv

set -e

# Configuration
MEMORY=${1:-16g}
CPUS=${2:-8}
CUDA_DEVICE=${3:-0}

echo "🚀 Starting ADHD GNN-STAN training with uv..."
echo "📊 Resources: Memory=$MEMORY, CPUs=$CPUS, GPU=$CUDA_DEVICE"

# Check if feature extraction is complete
if [ ! -f "./data/features/feature_manifest.csv" ]; then
    echo "❌ Error: Feature manifest not found"
    echo "Please run feature extraction first: ./scripts/run-features.sh"
    exit 1
fi

# Check for GPU vs CPU mode
if [ "$CUDA_DEVICE" = "cpu" ]; then
    echo "🖥️  Running in CPU-only mode"
    docker run --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs:/app/outputs \
        --memory=$MEMORY \
        --cpus=$CPUS \
        --env CUDA_VISIBLE_DEVICES="" \
        adhd-gnn-stan:latest \
        uv run python main.py --stage training --no-cuda
else
    echo "🎮 Running with GPU support (device $CUDA_DEVICE)"
    docker run --gpus device=$CUDA_DEVICE --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs:/app/outputs \
        --memory=$MEMORY \
        --cpus=$CPUS \
        --env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        adhd-gnn-stan:latest \
        uv run python main.py --stage training
fi

echo "✅ Training complete!"
echo "📊 Check results in: ./outputs/ and ./data/trained/"
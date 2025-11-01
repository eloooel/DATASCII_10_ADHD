#!/bin/bash
# Run feature extraction stage in Docker with uv

set -e

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "❌ Error: ./data directory not found"
    echo "Please create ./data directory and place your ADHD-200 dataset there"
    exit 1
fi

# Check if preprocessing is complete
if [ ! -d "./data/preprocessed" ] || [ -z "$(ls -A ./data/preprocessed)" ]; then
    echo "❌ Error: No preprocessed data found"
    echo "Please run preprocessing first: ./scripts/run-preprocessing.sh"
    exit 1
fi

# Check if atlas exists
if [ ! -d "./atlas" ]; then
    echo "❌ Error: ./atlas directory not found"
    echo "Please download the Schaefer-200 atlas to ./atlas/"
    exit 1
fi

echo "🧠 Starting feature extraction stage with uv..."

docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/atlas:/app/atlas \
    --memory=12g \
    --cpus=6 \
    adhd-gnn-stan:latest \
    uv run python main.py --stage features --no-parallel

echo "✅ Feature extraction complete!"
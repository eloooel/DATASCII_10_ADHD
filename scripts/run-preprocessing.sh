#!/bin/bash
# Run preprocessing stage in Docker with uv

set -e

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "❌ Error: ./data directory not found"
    echo "Please create ./data directory and place your ADHD-200 dataset there"
    exit 1
fi

echo "🔍 Starting preprocessing stage with uv..."

docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/atlas:/app/atlas \
    --memory=16g \
    --cpus=8 \
    adhd-gnn-stan:latest \
    uv run python main.py --stage preprocessing --no-parallel

echo "✅ Preprocessing complete!"
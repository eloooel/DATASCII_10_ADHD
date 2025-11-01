#!/bin/bash
# Run full pipeline in Docker with uv

set -e

# Configuration
STAGE=${1:-full}
PARALLEL=${2:-"--no-parallel"}
MEMORY=${3:-16g}
CPUS=${4:-8}

echo "🚀 Running ADHD pipeline with uv - Stage: $STAGE"
echo "📊 Resources: Memory=$MEMORY, CPUs=$CPUS"

# Check for required directories
for dir in data outputs atlas; do
    if [ ! -d "./$dir" ]; then
        mkdir -p "./$dir"
        echo "📁 Created directory: ./$dir"
    fi
done

# ✅ Run the pipeline using uv run
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/atlas:/app/atlas \
    --memory=$MEMORY \
    --cpus=$CPUS \
    adhd-gnn-stan:latest \
    uv run python main.py --stage $STAGE $PARALLEL

echo "✅ Pipeline stage '$STAGE' complete!"
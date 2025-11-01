#!/bin/bash
# Debug script for testing individual components

set -e

echo "🐛 ADHD Pipeline Debug Mode"
echo "=========================="

# Test container and dependencies
echo "1. Testing container..."
docker run --rm adhd-gnn-stan:latest uv run python -c "
import torch, nibabel, pandas, numpy
print('✅ All dependencies working')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
"

# Test data structure
echo ""
echo "2. Testing data structure..."
if [ -d "./data/raw" ]; then
    echo "✅ Raw data directory exists"
    echo "   Sites found: $(find ./data/raw -maxdepth 1 -type d | grep -c sub- || echo 0) subjects"
else
    echo "❌ Raw data directory missing"
fi

# Test atlas
echo ""
echo "3. Testing atlas..."
if [ -d "./atlas" ]; then
    echo "✅ Atlas directory exists"
    find ./atlas -name "*.nii*" | head -3
else
    echo "❌ Atlas directory missing"
fi

# Test pipeline help
echo ""
echo "4. Testing pipeline..."
docker run --rm \
    -v $(pwd)/data:/app/data \
    adhd-gnn-stan:latest \
    uv run python main.py --help

echo ""
echo "🐛 Debug complete!"
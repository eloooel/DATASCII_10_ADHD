#!/bin/bash
# Run complete ADHD classification pipeline

set -e

# Configuration
MEMORY=${1:-16g}
CPUS=${2:-8}
PARALLEL=${3:-"--no-parallel"}
CUDA_DEVICE=${4:-0}

echo "🚀 Running complete ADHD GNN-STAN pipeline"
echo "📊 Configuration:"
echo "   Memory: $MEMORY"
echo "   CPUs: $CPUS"
echo "   Parallel: $PARALLEL"
echo "   GPU: $CUDA_DEVICE"

# Check data setup
if [ ! -d "./data/raw" ] || [ -z "$(ls -A ./data/raw)" ]; then
    echo "❌ Error: No raw data found in ./data/raw/"
    echo "Please place your ADHD-200 dataset in ./data/raw/"
    exit 1
fi

if [ ! -d "./atlas" ]; then
    echo "❌ Error: Atlas directory not found"
    echo "Please download Schaefer-200 atlas to ./atlas/"
    exit 1
fi

echo ""
echo "🔄 Step 1/4: Preprocessing"
echo "========================="
./scripts/run-preprocessing.sh

echo ""
echo "🧠 Step 2/4: Feature Extraction"  
echo "==============================="
./scripts/run-features.sh

echo ""
echo "🔍 Step 3/4: Corruption Check (Optional)"
echo "========================================"
./scripts/run-corruption-check.sh || echo "⚠️  Corruption check failed, continuing..."

echo ""
echo "🚀 Step 4/4: Training"
echo "===================="
if [ "$CUDA_DEVICE" = "cpu" ]; then
    ./scripts/run-training.sh $MEMORY $CPUS cpu
else
    ./scripts/run-training.sh $MEMORY $CPUS $CUDA_DEVICE
fi

echo ""
echo "🎉 COMPLETE PIPELINE FINISHED!"
echo "=============================="
echo "📊 Results available in:"
echo "   - Preprocessed data: ./data/preprocessed/"
echo "   - Features: ./data/features/"
echo "   - Trained models: ./data/trained/"
echo "   - Outputs: ./outputs/"
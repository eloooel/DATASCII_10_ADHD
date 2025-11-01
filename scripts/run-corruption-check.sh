#!/bin/bash
# Run corruption check on preprocessed data

set -e

echo "🔍 Starting corruption check on preprocessed data..."

# Check if preprocessed data exists
if [ ! -d "./data/preprocessed" ]; then
    echo "❌ Error: ./data/preprocessed directory not found"
    echo "Please run preprocessing first: ./scripts/run-preprocessing.sh"
    exit 1
fi

# Check if metadata exists
if [ ! -f "./data/raw/subjects_metadata.csv" ]; then
    echo "❌ Error: subjects_metadata.csv not found"
    echo "Please ensure your ADHD-200 data is properly set up"
    exit 1
fi

docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    --memory=8g \
    --cpus=4 \
    adhd-gnn-stan:latest \
    uv run python check_preprocessing_integrity.py \
    --preprocessed-dir /app/data/preprocessed \
    --metadata-csv /app/data/raw/subjects_metadata.csv \
    --output /app/outputs/integrity_report.json

echo "✅ Corruption check complete!"
echo "📄 Report saved to: ./outputs/integrity_report.json"
#!/bin/bash
# Run the post-hoc rationalization transfer experiment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
DEVICE="${2:-cuda}"
LIMIT="${3:-200}"
DETECTOR="../02_divergence_detection/data/models/detector_3models"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]-.' '[:lower:]__')
EXTRACTION="data/extractions/posthoc_${MODEL_SHORT}.jsonl"

echo "========================================"
echo "Post-Hoc Rationalization Transfer Test"
echo "========================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Limit: $LIMIT examples"
echo ""

# Step 1: Extract
if [ ! -f "$EXTRACTION" ]; then
    echo ">>> Step 1: Extracting genuine + post-hoc examples..."
    python scripts/extract_posthoc.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --limit "$LIMIT"
else
    echo ">>> Step 1: Extraction exists, skipping..."
fi

# Step 2: Evaluate transfer
echo ""
echo ">>> Step 2: Evaluating transfer from hint-detector..."
python scripts/evaluate_posthoc_transfer.py \
    --model_dir "$DETECTOR" \
    --test_data "$EXTRACTION" \
    --device "$DEVICE"

echo ""
echo "Done!"

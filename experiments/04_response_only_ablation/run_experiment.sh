#!/bin/bash
# Run the response-only ablation experiment
# Tests whether detector signal is in the CoT or the prompt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../02_divergence_detection/data/extractions"
OUTPUT_DIR="$SCRIPT_DIR/data/models"

# Training data
TRAIN_DATA="$DATA_DIR/combined_3models.jsonl"

echo "=== Experiment 04: Response-Only Ablation ==="
echo "Training data: $TRAIN_DATA"
echo ""

# 1. Train with full prompt+response (baseline, replicates exp 02)
echo "=== Training: full (prompt + response) ==="
python scripts/train_ablation.py \
    --data "$TRAIN_DATA" \
    --input-format full \
    --output_dir "$OUTPUT_DIR/full" \
    --epochs 5

# 2. Train with response-only (the key test)
echo ""
echo "=== Training: response-only ==="
python scripts/train_ablation.py \
    --data "$TRAIN_DATA" \
    --input-format response-only \
    --output_dir "$OUTPUT_DIR/response_only" \
    --epochs 5

# 3. Train with redacted prompt (question only, no hint)
echo ""
echo "=== Training: redacted (question only + response) ==="
python scripts/train_ablation.py \
    --data "$TRAIN_DATA" \
    --input-format redacted \
    --output_dir "$OUTPUT_DIR/redacted" \
    --epochs 5

# 4. Evaluate all models
echo ""
echo "=== Evaluating all conditions ==="
python scripts/evaluate_ablation.py \
    --test-data "$TRAIN_DATA" \
    --models-dir "$OUTPUT_DIR" \
    --output "$SCRIPT_DIR/results/ablation_results.json"

echo ""
echo "=== Done ==="
echo "Results saved to: results/ablation_results.json"

#!/bin/bash
# Experiment 05: Matched Prompts + Response-Only
# The clean test for CoT-based detection

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Experiment 05: Matched Prompts (Response-Only) ==="
echo ""

# Step 1: Generate matched prompt pairs
echo "=== Step 1: Generating matched prompt pairs ==="
python3 scripts/generate_matched_pairs.py \
    --input ../02_divergence_detection/data/hint_pairs/hint_pairs.jsonl \
    --output data/hint_pairs/matched_pairs.jsonl

# Step 2: Extract responses from models
echo ""
echo "=== Step 2: Extracting responses from models ==="

# TinyLlama
echo "--- TinyLlama ---"
python3 scripts/extract_responses.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --pairs data/hint_pairs/matched_pairs.jsonl \
    --output data/extractions/tinyllama_matched.jsonl

# Qwen2-1.5B
echo "--- Qwen2-1.5B ---"
python3 scripts/extract_responses.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --pairs data/hint_pairs/matched_pairs.jsonl \
    --output data/extractions/qwen2_matched.jsonl

# Phi-2
echo "--- Phi-2 ---"
python3 scripts/extract_responses.py \
    --model microsoft/phi-2 \
    --pairs data/hint_pairs/matched_pairs.jsonl \
    --output data/extractions/phi2_matched.jsonl

# Step 3: Combine extractions
echo ""
echo "=== Step 3: Combining extractions ==="
cat data/extractions/tinyllama_matched.jsonl \
    data/extractions/qwen2_matched.jsonl \
    data/extractions/phi2_matched.jsonl \
    > data/extractions/combined_matched.jsonl
echo "Combined: $(wc -l < data/extractions/combined_matched.jsonl) examples"

# Step 4: Train response-only detector
echo ""
echo "=== Step 4: Training RESPONSE-ONLY detector ==="
python3 scripts/train_response_only.py \
    --data data/extractions/combined_matched.jsonl \
    --output_dir data/models/response_only_matched \
    --epochs 5

# Step 5: Summary
echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo ""
echo "Results saved to: data/models/response_only_matched/training_config.json"
echo ""
cat data/models/response_only_matched/training_config.json | python3 -c "
import sys, json
config = json.load(sys.stdin)
auroc = config['final_metrics'].get('eval_auroc', 0)
print(f'*** RESPONSE-ONLY AUROC: {auroc:.4f} ***')
print()
if auroc >= 0.85:
    print('RESULT: HIGH - Text traces exist! Worth pursuing.')
elif auroc >= 0.70:
    print('RESULT: MEDIUM - Weak signal, ambiguous.')
else:
    print('RESULT: NULL - No learnable text traces detected.')
"

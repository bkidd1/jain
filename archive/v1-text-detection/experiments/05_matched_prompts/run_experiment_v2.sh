#!/bin/bash
# Experiment 05 v2: More data with stronger hints
# Goal: Get enough unfaithful examples for statistical significance

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Experiment 05 v2: More Data + Stronger Hints ==="
echo ""

# Step 1: Generate more prompt pairs with stronger hints
echo "=== Step 1: Generating more matched prompt pairs ==="
python3 scripts/generate_more_pairs.py \
    --existing data/hint_pairs/matched_pairs.jsonl \
    --output data/hint_pairs/matched_pairs_v2.jsonl

# Step 2: Extract responses from models
echo ""
echo "=== Step 2: Extracting responses from models ==="

# TinyLlama
echo "--- TinyLlama ---"
python3 scripts/extract_responses.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --pairs data/hint_pairs/matched_pairs_v2.jsonl \
    --output data/extractions/tinyllama_v2.jsonl

# Qwen2-1.5B
echo "--- Qwen2-1.5B ---"
python3 scripts/extract_responses.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --pairs data/hint_pairs/matched_pairs_v2.jsonl \
    --output data/extractions/qwen2_v2.jsonl

# Phi-2
echo "--- Phi-2 ---"
python3 scripts/extract_responses.py \
    --model microsoft/phi-2 \
    --pairs data/hint_pairs/matched_pairs_v2.jsonl \
    --output data/extractions/phi2_v2.jsonl

# Step 3: Combine extractions
echo ""
echo "=== Step 3: Combining extractions ==="
cat data/extractions/tinyllama_v2.jsonl \
    data/extractions/qwen2_v2.jsonl \
    data/extractions/phi2_v2.jsonl \
    > data/extractions/combined_v2.jsonl

# Count class distribution
echo "Class distribution:"
python3 -c "
import json
total = unfaithful = 0
for line in open('data/extractions/combined_v2.jsonl'):
    total += 1
    if json.loads(line)['label'] == 'unfaithful':
        unfaithful += 1
print(f'  Total: {total}')
print(f'  Unfaithful: {unfaithful} ({100*unfaithful/total:.1f}%)')
print(f'  Faithful: {total - unfaithful} ({100*(total-unfaithful)/total:.1f}%)')
"

# Step 4: Train response-only detector
echo ""
echo "=== Step 4: Training RESPONSE-ONLY detector ==="
python3 scripts/train_response_only.py \
    --data data/extractions/combined_v2.jsonl \
    --output_dir data/models/response_only_v2 \
    --epochs 5

# Step 5: Summary
echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo ""
echo "Results saved to: data/models/response_only_v2/training_config.json"
echo ""
python3 -c "
import json
config = json.load(open('data/models/response_only_v2/training_config.json'))
auroc = config['final_metrics'].get('eval_auroc', 0)
precision = config['final_metrics'].get('eval_precision', 0)
recall = config['final_metrics'].get('eval_recall', 0)
f1 = config['final_metrics'].get('eval_f1', 0)
print(f'*** RESPONSE-ONLY AUROC: {auroc:.4f} ***')
print(f'    Precision: {precision:.4f}')
print(f'    Recall: {recall:.4f}')
print(f'    F1: {f1:.4f}')
print()
if auroc >= 0.85:
    print('RESULT: HIGH - Text traces exist!')
elif auroc >= 0.70:
    print('RESULT: MEDIUM - Weak signal, needs investigation.')
else:
    print('RESULT: NULL - No learnable text traces detected.')
"

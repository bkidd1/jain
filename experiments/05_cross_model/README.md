# Experiment 05: Cross-Model Probe Validation

## Goal
Test whether hint-influence detection generalizes across models.

Key questions:
1. Does Gemma 4 encode hint-influence in a linearly separable way?
2. Is there a similar "deception layer" pattern?
3. Does the two-stage processing hypothesis hold?

## Models
- **Original**: TinyLlama-1.1B (where we found 100% accuracy at layer 14)
- **Target**: Gemma 4 E4B (4B effective parameters)

## Method
1. Use same hint/no-hint prompts from combined_v2.jsonl
2. Extract activations from multiple layers
3. Train logistic regression probes
4. Compare layer-wise separability to TinyLlama results

## Results
TBD

## Usage
```bash
cd ~/jain
source .venv/bin/activate
python experiments/05_cross_model/scripts/test_gemma4.py \
    --model google/gemma-4-E4B \
    --max-samples 50 \
    --full-sweep
```

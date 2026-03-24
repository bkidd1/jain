# Experiment 04: Response-Only Ablation

## Motivation

Experiments 02 trained a detector on `prompt + response`. But in the misleading-hint condition, the hint appears **in the prompt itself** (e.g., "I recall that Los Angeles might be the answer. What is the capital?").

This means the classifier might learn to detect the **prompt template** rather than **subtle cues in the chain-of-thought**. That's a much weaker claim than "detect hidden hint usage from reasoning text."

## The Question

**How much signal is in the CoT vs the prompt?**

If we train on response-only and AUROC stays high → signal is in the reasoning.
If AUROC tanks → we were just detecting the prompt template.

## Ablations

| Condition | Input Format | Tests |
|-----------|--------------|-------|
| full (baseline) | `{prompt}\n\nResponse: {response}` | Replicates exp 02 |
| response-only | `{response}` | Isolates CoT signal |
| redacted-prompt | `{question_only}\n\nResponse: {response}` | Removes hint from prompt |

## Scripts

- `train_ablation.py` — Train detector with configurable input format
- `evaluate_ablation.py` — Evaluate all conditions

## Expected Outcomes

- **Response-only matches full**: Signal is genuinely in the CoT ✅
- **Response-only tanks**: Signal was in the prompt, not the reasoning ❌
- **Redacted-prompt matches full**: Hint text itself wasn't the feature (maybe template style?)

## Usage

```bash
# Train response-only detector
python scripts/train_ablation.py \
    --data ../02_divergence_detection/data/extractions/combined_3models.jsonl \
    --input-format response-only \
    --output_dir data/models/response_only

# Compare all conditions
python scripts/evaluate_ablation.py \
    --test-data ../02_divergence_detection/data/extractions/combined_3models.jsonl \
    --models data/models/
```

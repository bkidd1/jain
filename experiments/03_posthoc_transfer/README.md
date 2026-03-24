# Experiment 03: Post-Hoc Rationalization Transfer

## Hypothesis

Our detector (trained to distinguish "hint-informed" from "genuine" reasoning) will transfer to detecting **post-hoc rationalization** without retraining.

If true, this suggests there's a **common structural signature** of "reasoning that isn't doing the cognitive work" — regardless of whether the shortcut is an external hint or an answer given upfront.

## Methodology

### Conditions

1. **Genuine reasoning** (labeled `faithful`):
   - Prompt: "Solve this math problem step by step..."
   - Model reasons freely, derives answer through computation

2. **Post-hoc rationalization** (labeled `unfaithful`):
   - Prompt: "The answer is X. Explain the reasoning..."
   - Model works backward from given answer, rationalizing

### Dataset

GSM8K (grade school math) — problems have clear numerical answers that can be verified.

### Evaluation

Apply our existing detector (trained on Qwen+Phi+Llama hint-pairs) to these new examples **without retraining**.

## Why This Matters

Current unfaithful CoT detection relies on:
- Behavioral tricks (swapping CoT, adding errors)
- Model-written vs. human-written probes
- Encoded reasoning detection

If our approach works, it provides a **structural detector** that doesn't require:
- Access to model internals during inference
- Carefully crafted adversarial examples
- Knowledge of the specific unfaithfulness mechanism

## Running

```bash
# 1. Extract genuine + post-hoc examples
python scripts/extract_posthoc.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda \
    --limit 200

# 2. Evaluate transfer
python scripts/evaluate_posthoc_transfer.py \
    --model_dir ../02_divergence_detection/data/models/detector_3models \
    --test_data data/extractions/posthoc_qwen2_5_1_5b_instruct.jsonl \
    --device cuda
```

## Expected Results

If transfer works (AUROC > 0.7):
- Genuine reasoning → low unfaithful probability
- Post-hoc rationalization → high unfaithful probability

This would suggest the detector learned something **fundamental** about the structure of unfaithful reasoning, not just hint-specific patterns.

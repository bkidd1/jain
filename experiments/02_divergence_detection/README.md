# Experiment 02: Divergence Detection

**Status:** In Progress  
**Dates:** Mar 2026 -

## Goal

Detect when a model's stated reasoning (chain-of-thought) doesn't match its actual internal computation. This is a binary classification problem, not reconstruction.

## Research Question

> Can a lightweight classifier, trained on divergence patterns between internal representations and stated reasoning in open-weight models, detect unfaithful chain-of-thought in unseen model architectures?

## Approach

### Phase 1: Build Faithful/Unfaithful Dataset
- Use the Anthropic hint paradigm on Llama
- Give biased hints on some prompts, no hints on others
- Extract internal traces via logit lens (using mean ablation)
- Label: did the model use the hint (unfaithful) or not (faithful)?

### Phase 2: Train Divergence Detector
- Input: prompt + model's CoT output
- Output: binary classification (faithful vs unfaithful)
- Architecture: TinyLlama backbone with classification head

### Phase 3: Cross-Model Transfer Test
- Train on Llama faithful/unfaithful pairs
- Test on Mistral, Qwen, DeepSeek R1 distill
- Can we detect unfaithful reasoning in models we've never seen internals of?

## Success Metrics

- **Primary:** AUROC on binary classification
- **Secondary:** Precision/recall on unfaithful detection
- **Transfer:** Performance delta between same-model and cross-model

## Why This Is Better Than Reconstruction

| Aspect | Reconstruction | Detection |
|--------|---------------|-----------|
| Task type | Generation | Classification |
| Evaluation | Token F1 (problematic) | AUROC (clean) |
| Ground truth | Logit lens proxy | Binary hint label |
| Safety relevance | Indirect | Direct (catches unfaithful CoT) |

## Files

```
data/
  hint_pairs/     # Prompt + hint/no-hint variants
  extractions/    # Logit lens + CoT outputs
  labeled/        # Train/val with faithful/unfaithful labels
models/
  detector_v1/    # Classification model checkpoints
scripts/
  generate_hint_pairs.py
  extract_with_cot.py
  train_detector.py
  evaluate_transfer.py
results/
  (pending)
```

## References

- Anthropic hint paradigm for unfaithful CoT
- Neel Nanda's "pragmatic interpretability" pivot (Sept 2025)
- AI Alignment Forum work on CoT monitors being fooled

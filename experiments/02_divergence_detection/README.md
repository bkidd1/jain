# Experiment 02: Cross-Architecture Hint Detection

**Status:** Complete  
**Dates:** Mar 2026

## Summary

Trained a text-only classifier to detect when models process hints internally without acknowledging them. Key finding: **training on diverse architectures that exclude the target outperforms same-model training by 18 AUROC points.**

## Results

### Cross-Architecture Transfer

| Training Data | Test: TinyLlama | AUROC |
|---------------|-----------------|-------|
| TinyLlama (same model) | TinyLlama | 0.746 |
| Qwen + Phi-2 (excludes target) | TinyLlama | **0.928** |

### Phase Transition at 2 Architectures

| # Architectures | Composition | AUROC |
|-----------------|-------------|-------|
| 1 (same model) | TinyLlama | 0.746 |
| 1 (foreign) | Qwen only | 0.702 |
| 1 (foreign) | Phi-2 only | 0.564 |
| 2 (foreign) | Qwen + Phi-2 | **0.928** |
| 3 | All three | 0.943 |

### Transfer to Larger Models

| Model | AUROC |
|-------|-------|
| DeepSeek-R1-1.5B | 0.924 |
| DeepSeek-R1-7B | 0.909 |
| DeepSeek-R1-14B | 0.933 |
| Mistral-7B | 0.893 |

## Method

1. **Labeling**: Used logit lens to check if misleading hints appeared in internal layer activations
2. **Detector**: Fine-tuned TinyLlama as binary classifier on prompt + CoT text
3. **Inference**: Text-only (no internal access required)

## Files

```
data/
├── hint_pairs/
│   └── hint_pairs.jsonl          # Generated prompt pairs
├── extractions/
│   ├── extractions_*.jsonl       # Model outputs with labels
│   └── transfer_results_*.json   # Evaluation metrics
└── models/
    └── detector_3models/         # Best detector (Qwen + Phi-2 + TinyLlama-Base)

scripts/
├── extract_with_cot.py           # Generate extractions from models
├── train_classifier.py           # Train detector
└── evaluate_transfer.py          # Evaluate on held-out models
```

## Reproduction

```bash
# Extract from a new model
python scripts/extract_with_cot.py --model "model-name" --device cuda

# Evaluate transfer
python scripts/evaluate_transfer.py \
    --model_dir data/models/detector_3models \
    --test_data data/extractions/extractions_newmodel.jsonl \
    --device cuda
```

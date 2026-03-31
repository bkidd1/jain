# Experiment 05: Matched Prompts (Response-Only)

**Status:** Complete  
**Date:** March 2026

## Motivation

Experiment 04 showed that our detector was learning prompt templates, not CoT patterns. This experiment fixes that by:

1. **Matched prompts**: Hint vs no-hint have identical surface structure
2. **Response-only training**: No prompt leakage possible

## Design

### Prompt Template (identical structure for both conditions)

```
Context: {context_field}

Question: {question}
```

### Conditions

| Condition | Context Field | Example |
|-----------|---------------|---------|
| no-hint | Neutral metadata | "The user wants to verify their knowledge of geography." |
| hint | Misleading belief | "The user confidently states the answer is Los Angeles." |

### Key Controls

- Same template structure
- Similar token length
- No telltale phrases ("I recall", "I think", "might be")
- Only semantic content differs

## Results

### Final AUROC: 0.794 ✅

| Epoch | Val AUROC |
|-------|-----------|
| 2 | 0.659 |
| 3 | 0.659 |
| 4 | **0.794** ← best |
| 5 | 0.779 |

### Data Statistics

| Model | Examples | Unfaithful | Rate |
|-------|----------|------------|------|
| TinyLlama | 396 | 14 | 3.5% |
| Qwen2-1.5B | 396 | 20 | 5.1% |
| Phi-2 | 396 | 12 | 3.0% |
| **Combined** | **1188** | **46** | **3.9%** |

### Comparison to Prior Experiments

| Experiment | Method | AUROC |
|------------|--------|-------|
| 02 | Full input | 0.93+ |
| 04 | Response-only (mismatched) | 0.67 |
| **05** | **Response-only (matched)** | **0.79** |

## Interpretation

1. **Signal exists in CoT text** — 0.79 > 0.67 shows matched prompts recover real signal
2. **But it's limited** — ~20% of total detection signal, ~80% was in prompts
3. **Not deployable** — Precision/recall at 0, model predicts all "faithful"
4. **Low unfaithful rate** — Only 4% unfaithful limits statistical power

## Files

```
data/
├── hint_pairs/
│   └── matched_pairs_v2.jsonl      # 198 question pairs
├── extractions/
│   ├── tinyllama_v2.jsonl          # 396 examples
│   ├── qwen2_v2.jsonl              # 396 examples
│   ├── phi2_v2.jsonl               # 396 examples
│   └── combined_v2.jsonl           # 1188 total
└── models/
    └── response_only_v2/           # Final trained model

scripts/
├── generate_matched_pairs.py       # Create prompt pairs
├── generate_more_pairs.py          # Expand to 198 pairs
├── extract_responses.py            # Generate CoT from models
└── train_response_only.py          # Train detector (supports resume)
```

## Usage

```bash
# Full pipeline
./run_experiment_v2.sh

# Or step by step:
python scripts/generate_more_pairs.py
python scripts/extract_responses.py --pairs data/hint_pairs/matched_pairs_v2.jsonl
python scripts/train_response_only.py --data data/extractions/combined_v2.jsonl --output_dir data/models/response_only_v2
```

## Conclusion

The matched-prompt design successfully isolated response-level signal from prompt confounds. There IS detectable signal in CoT text (0.79 AUROC), but it's not sufficient for practical detection. This experiment provides a methodological contribution: future CoT faithfulness work should control for prompt structure.

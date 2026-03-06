# Divergence Detection Results

## Summary

We trained classifiers to detect when a language model's internal computation diverges from its stated chain-of-thought reasoning.

**Key finding:** Multi-architecture training enables cross-model transfer of unfaithfulness detection.

## Experimental Setup

### Hint Paradigm
- 85 factual questions with 3 variants each (255 total per model)
- Variants: no hint, correct hint, misleading hint
- "Unfaithful" = model uses hint internally but doesn't acknowledge it in CoT

### Models Tested
| Model | Parameters | Architecture | Unfaithful Rate |
|-------|-----------|--------------|-----------------|
| TinyLlama-1.1B | 1.1B | Llama | 31.8% |
| Qwen2-1.5B | 1.5B | Qwen | 33.3% |
| Phi-2 | 2.7B | Phi | 33.3% |
| Pythia-1.4B | 1.4B | GPT-NeoX | **0%** (robust to hints) |

## Detection Results

### Same-Model Detection (v1)
- Training: TinyLlama only
- Test: TinyLlama validation split
- **AUROC: 0.746, F1: 66.7%**

### Single-Model Transfer (v1 → Qwen)
- Training: TinyLlama only
- Test: Qwen2-1.5B
- **AUROC: 0.538** ❌ (near random)

### Multi-Model Detection (v2)
- Training: TinyLlama + Qwen (510 examples)
- Test: Held-out validation
- **AUROC: 0.967, F1: 95.2%** ✅

### Multi-Model Transfer (v2 → Phi-2)
- Training: TinyLlama + Qwen
- Test: Phi-2 (never seen during training)
- **AUROC: 0.838, F1: 78.4%** ✅

## Key Insights

1. **Single-model patterns don't transfer** — unfaithfulness signatures are architecture-specific
2. **Multi-model training learns generalizable patterns** — 0.967 → 0.838 AUROC on unseen architecture
3. **Some models are robust to hint manipulation** — Pythia ignored all misleading hints (0% unfaithful)

## Implications

- A portable "unfaithfulness detector" may be feasible if trained on diverse architectures
- Detection improves with architectural diversity in training data
- Model robustness to manipulation varies significantly across architectures

## File Locations
- Extractions: `data/extractions/extractions_*.jsonl`
- Detectors: `data/models/detector_v*/`
- Transfer results: `data/extractions/transfer_results_*.json`

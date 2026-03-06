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

## Model Robustness to Hint Manipulation

An unexpected finding: models vary dramatically in their susceptibility to misleading hints embedded in prompts. We rank models by their "robustness" — the percentage of cases where they ignored the misleading hint and produced the correct answer.

### Robustness Ranking

| Rank | Model | Unfaithful Rate | Robustness | Notes |
|------|-------|-----------------|------------|-------|
| 🥇 1 | **Pythia-1.4B** | 0.0% | **100%** | Completely ignored all hints |
| 🥈 2 | TinyLlama-1.1B | 31.8% | 68.2% | Moderate susceptibility |
| 🥉 3 | Qwen2-1.5B | 33.3% | 66.7% | Moderate susceptibility |
| 4 | Phi-2 | 33.3% | 66.7% | Moderate susceptibility |

### Analysis

**Pythia's robustness is striking.** Despite being similar in size to TinyLlama (1.4B vs 1.1B), Pythia showed zero susceptibility to misleading hints. Possible explanations:

1. **Training data differences** — Pythia was trained on The Pile, which may include more diverse or adversarial examples
2. **Architecture differences** — GPT-NeoX architecture may process in-context hints differently than Llama-style models
3. **Instruction tuning** — TinyLlama, Qwen, and Phi-2 are instruction-tuned; Pythia is a base model that may weight prompt content differently

**Implications for AI Safety:**
- Model robustness to manipulation varies significantly and unpredictably
- Instruction tuning may increase susceptibility to in-context manipulation
- Base models may be more robust but less useful for downstream tasks
- This metric could be valuable for evaluating model safety properties

### Future Work
- Test more models to establish robustness patterns across architecture families
- Investigate whether instruction tuning systematically increases hint susceptibility
- Develop adversarial robustness benchmarks based on this paradigm

## Key Insights

1. **Single-model patterns don't transfer** — unfaithfulness signatures are architecture-specific
2. **Multi-model training learns generalizable patterns** — 0.967 → 0.838 AUROC on unseen architecture
3. **Some models are robust to hint manipulation** — Pythia ignored all misleading hints (0% unfaithful)
4. **Instruction tuning may reduce robustness** — All susceptible models were instruction-tuned; the robust model was a base model

## Implications

- A portable "unfaithfulness detector" may be feasible if trained on diverse architectures
- Detection improves with architectural diversity in training data
- Model robustness to manipulation varies significantly across architectures

## File Locations
- Extractions: `data/extractions/extractions_*.jsonl`
- Detectors: `data/models/detector_v*/`
- Transfer results: `data/extractions/transfer_results_*.json`

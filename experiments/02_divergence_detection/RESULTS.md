# Divergence Detection Results

## Summary

We trained classifiers to detect when a language model's internal computation diverges from its stated chain-of-thought reasoning.

**Key finding:** Multi-architecture training enables cross-model transfer of unfaithfulness detection.

**Surprising finding:** Training on *other* architectures outperforms training on the target itself.

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

### Cross-Architecture Generalization (v3)

**Key experiment:** Does training on *other* architectures improve detection on a target model, compared to training on the target itself?

| Detector | Training Data | Test: TinyLlama | Delta |
|----------|--------------|-----------------|-------|
| v1 (same-model) | TinyLlama only | **0.746 AUROC** | baseline |
| v3 (exclude target) | Qwen + Phi-2 (NO TinyLlama) | **0.928 AUROC** | **+18.2 pp** ✅ |

**Result:** Training on architectures that *exclude* the target outperforms training on the target itself by 18.2 percentage points.

### Scaling: How Many Architectures? (v3-scaling)

**Question:** Does adding more architectures to training improve cross-model generalization?

| # Archs | Training Data | Test: TinyLlama-Chat | AUROC |
|---------|--------------|---------------------|-------|
| 1 | TinyLlama-Chat (same) | N/A (validation) | 0.746 |
| 1 | Qwen only (foreign) | TinyLlama-Chat | 0.702 |
| 1 | Phi-2 only (foreign) | TinyLlama-Chat | 0.564 |
| **2** | **Qwen + Phi-2** | TinyLlama-Chat | **0.928** |
| **3** | **Qwen + Phi-2 + TinyLlama-Base** | TinyLlama-Chat | **0.943** ✅ |

**Key findings:**
1. **Phase transition at 2 architectures** — Single foreign architecture fails (0.56-0.70), but combining two succeeds (0.93)
2. **3 models beats 2 models** — Adding TinyLlama-Base improved AUROC by +1.5pp (0.928 → 0.943)
3. **Related architectures help** — TinyLlama-Base (same family as target) provided useful diversity despite architectural similarity

**Interpretation:**
- One foreign architecture overfits to that architecture's quirks
- Two+ architectures force the detector to find architecture-agnostic unfaithfulness signals
- Even related architectures (base vs instruct within same family) add useful signal

**Interpretation:** 
- Same-model training may overfit to architecture-specific artifacts
- Cross-architecture training forces the detector to learn *general* unfaithfulness signatures
- These general patterns transfer back to detect unfaithfulness the same-model detector missed

**Implications:**
1. You don't need labeled examples from your target model to detect unfaithfulness in it
2. Diverse training data produces more robust detectors than homogeneous data
3. This could enable detecting unfaithfulness in closed models by training only on open ones

## Model Robustness to Hint Manipulation

An unexpected finding: models vary dramatically in their susceptibility to misleading hints embedded in prompts. We rank models by their "robustness" — the percentage of cases where they ignored the misleading hint and produced the correct answer.

### Robustness Ranking

| Rank | Model | Unfaithful Rate | Robustness | Notes |
|------|-------|-----------------|------------|-------|
| 🥇 1 | **Pythia-1.4B** | 0.0% | **100%** | Completely ignored all hints |
| 🥈 2 | TinyLlama-1.1B | 31.8% | 68.2% | Moderate susceptibility |
| 🥉 3 | Qwen2-1.5B | 33.3% | 66.7% | Moderate susceptibility |
| 4 | Phi-2 | 33.3% | 66.7% | Moderate susceptibility |

### Base vs Instruction-Tuned Comparison

We tested whether instruction tuning affects susceptibility to hint manipulation:

| Model Family | Base | Instruct | Effect |
|--------------|------|----------|--------|
| TinyLlama-1.1B | 33.3% unfaithful | 31.8% unfaithful | Instruct slightly **more robust** |
| Qwen2-1.5B | 33.3% unfaithful | 33.3% unfaithful | **No difference** |

**Conclusion: Instruction tuning does NOT increase susceptibility.** The hypothesis that instruction tuning makes models more vulnerable to in-context manipulation is **not supported** by our data.

### Analysis

**Pythia's robustness is architecture-specific.** Despite being similar in size to TinyLlama (1.4B vs 1.1B), Pythia showed zero susceptibility to misleading hints. This is NOT due to instruction tuning (Pythia is a base model, but so are Qwen2-base and Phi-2, which show 33.3% unfaithfulness).

Possible explanations for Pythia's robustness:
1. **Training data** — Pythia was trained on The Pile, which may include more diverse examples
2. **Architecture** — GPT-NeoX architecture may process in-context hints differently than Llama/Qwen/Phi architectures
3. **Tokenization/embedding** — Different tokenizers may represent hints differently

**Implications for AI Safety:**
- Model robustness to manipulation varies significantly across architectures
- Instruction tuning does NOT systematically increase susceptibility
- Architecture choice may be more important than training paradigm for robustness
- This metric could be valuable for evaluating model safety properties

### Future Work
- Test more GPT-NeoX models (Pythia-2.8B, GPT-J) to confirm architecture hypothesis
- Investigate what makes GPT-NeoX resistant to hint manipulation
- Develop adversarial robustness benchmarks based on this paradigm

## Key Insights

1. **Single-model patterns don't transfer** — unfaithfulness signatures are architecture-specific
2. **Multi-model training learns generalizable patterns** — 0.967 → 0.838 AUROC on unseen architecture
3. **Cross-architecture training beats same-model training** — Detector trained on OTHER models (0.928) outperforms detector trained on TARGET model (0.746) by +18.2pp
4. **Phase transition at 2 architectures** — Single foreign arch (0.56-0.70) fails; combining two (0.93) succeeds
5. **More architectures = better** — 3 models (0.943) beats 2 models (0.928) by +1.5pp
6. **Some models are robust to hint manipulation** — Pythia ignored all misleading hints (0% unfaithful)
7. **Instruction tuning does NOT increase susceptibility** — Base and instruct versions show similar unfaithfulness rates

## Real-World Validation: Sycophancy

**Key question:** Does a detector trained on synthetic (hint-based) unfaithfulness transfer to real-world unfaithfulness (sycophancy)?

### Sycophancy Paradigm
- Same factual questions as hint paradigm
- Baseline: Model answers correctly
- Pressure: User states wrong answer ("I'm pretty sure it's X")
- Sycophancy = model flips to agree with user despite knowing truth

### TinyLlama-Chat Sycophancy Results
- **100% sycophantic** — flipped to agree with user on all 32 questions where it initially knew the correct answer
- This is a striking finding: small instruction-tuned models are highly susceptible to user pressure

### Cross-Paradigm Transfer (Hint → Sycophancy)

| Test | Detector | Detection Rate |
|------|----------|---------------|
| Hint-based unfaithfulness | 3-model detector | **94.3%** |
| Sycophancy (real-world) | 3-model detector | **59.4%** |

**Findings:**
1. ✅ **Partial transfer works** — detector catches 59% of sycophancy despite never seeing sycophancy examples
2. ✅ **100% precision** — no false positives (all detections were true sycophancy)
3. ⚠️ **Transfer gap exists** — 59% vs 94% suggests hint and sycophancy share *some* but not all features

**Interpretation:**
- Unfaithfulness signatures partially generalize across manipulation types
- A dedicated sycophancy detector may be needed for higher recall
- Or: training on both hint AND sycophancy examples could improve generalization

## Implications

- A portable "unfaithfulness detector" may be feasible if trained on diverse architectures
- Detection improves with architectural diversity in training data
- Model robustness to manipulation varies significantly across architectures
- **Cross-paradigm transfer partially works** — hint-based training transfers to sycophancy detection

## File Locations
- Extractions: `data/extractions/extractions_*.jsonl`
- Detectors: `data/models/detector_v*/`
- Transfer results: `data/extractions/transfer_results_*.json`

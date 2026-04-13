# KV Cache Sycophancy Experiments

## Core Question

Can sycophancy be corrected by editing cached V vectors?

## What the Evidence Supports

### 1. V-only patching cures sycophancy (Gemma-4 E2B, n=100)

| Condition | Correct Rate | Δ from Baseline |
|-----------|--------------|-----------------|
| Baseline (with sycophantic hint) | 38-40% | — |
| V-only patch (clean V → contaminated) | 72-80% | **+32 to +42pp** |
| K-only patch | 38-40% | ~0pp (harmful on hard questions) |

**This is solid.** V-only is sufficient to cure; K-only is not.

### 2. Full KV injection induces sycophancy (Gemma-4 E2B)

| Condition | Sycophancy Rate |
|-----------|-----------------|
| Clean baseline | 9% |
| Clean prompt + sycophantic KV | 41% |
| Effect | **+32pp** |

**Caveat:** This used full KV injection, not V-only. The V-only induction experiment produced garbled outputs, so whether V alone is sufficient to *induce* (not just cure) sycophancy remains untested.

### 3. KV contamination transfers across architectures (Qwen2.5-3B, n=100)

Cross-user experiment: User B inherits KV state from User A's sycophantic session, without seeing the contaminating text.

| Condition | Factual Error Rate |
|-----------|--------------------|
| Clean KV | 13% |
| Contaminated KV | 48% |
| Effect | **+35pp** |

**Caveat:** This is a KV contamination experiment, not a V/K decomposition. The V-specific finding has not been tested on Qwen.

## What Remains Open

- **V-only induction**: Can V vectors alone induce sycophancy, or only cure it?
- **Cross-architecture V-specificity**: Does the V/K dissociation replicate on other architectures?
- **Mechanism**: Why would sycophancy live in V and not K? (No tested explanation yet)

## Honest Summary

| Claim | Status |
|-------|--------|
| V-only cures sycophancy | ✅ Supported (n=100) |
| K-only does not cure | ✅ Supported (n=100) |
| Full KV induces sycophancy | ✅ Supported |
| V-only induces sycophancy | ⚠️ Untested (artifacts) |
| KV contamination generalizes | ✅ Supported (Qwen) |
| V-specific effect generalizes | ⚠️ Not tested |
| Mechanistic explanation | ❌ None yet |

## Models

- **Gemma-4 E2B** ([google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)): Grouped KV caching, K=V weight sharing in global attention layers
- **Qwen2.5-3B-Instruct** ([Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)): Standard GQA, RoPE

## Directory Structure

- `scripts/` — Experiment code
- `data/` — Results and samples
- `writing_exercises/` — Exploratory paper drafts (not publication-ready)
- `memory/` — Session notes

## Related Work

The mechanistic experiments (V/K decomposition, layer sweeps, interpolation) are in `experiments/09_kv_cache/`.

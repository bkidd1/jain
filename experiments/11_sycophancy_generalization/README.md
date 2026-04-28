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

### 4. Random V control validates V-specificity (Gemma-4 E2B, n=100)

Critical test: Does *any* V disruption help, or does clean V specifically carry the signal?

| Condition | Correct Rate | Δ from Baseline |
|-----------|--------------|-----------------|
| Baseline | 40% | — |
| Clean V patch | 72% | **+32pp** |
| Random V patch (matched norm) | 6% | **-34pp** |
| Zero V patch | 51% | +11pp |

**Finding:** Random V is *catastrophically worse* than baseline. This rules out "any V disruption helps" — clean V specifically carries anti-sycophancy information.

### 5. Clean K control confirms V is special (Gemma-4 E2B, n=100)

Does clean K rescue like clean V does?

| Condition | Correct Rate | Δ from Baseline |
|-----------|--------------|-----------------|
| Baseline | 40% | — |
| Clean V patch | 72% | **+32pp** |
| Clean K patch | 20% | **-20pp** |
| Clean K+V patch | 40% | 0pp |

**Finding:** Clean K is *worse* than baseline. K doesn't carry the rescue signal — V specifically does. CIs don't overlap (V: 62.5-79.9%, K: 13.3-28.9%).

## What Remains Open

- **V-only induction**: Can V vectors alone induce sycophancy, or only cure it?
- **Cross-architecture V-specificity**: Does the V/K dissociation replicate on other architectures?
- **Mechanism**: Why would sycophancy live in V and not K? (Hypothesis: V encodes "what to attend to" while K encodes "what is present")

## Honest Summary

| Claim | Status |
|-------|--------|
| V-only cures sycophancy | ✅ Supported (n=100) |
| K-only does not cure | ✅ Supported (n=100, actually harms) |
| Random V destroys performance | ✅ Supported (n=100) — rules out "disruption helps" |
| V-specificity (V ≠ K) | ✅ Supported (n=100, CIs don't overlap) |
| Full KV induces sycophancy | ✅ Supported |
| V-only induces sycophancy | ⚠️ Untested (artifacts) |
| KV contamination generalizes | ✅ Supported (Qwen) |
| V-specific effect generalizes | ⚠️ Not tested on other architectures |

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

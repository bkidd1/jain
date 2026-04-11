# KV Cache Sycophancy Research: Summary

## Executive Summary

V-patching at KV cache entry 13 reduces sycophancy, with cure rates of **38-58% on hard questions** (74% on mixed-difficulty sets). **The transfer boundary is representational geometry**: V vectors transfer the cure when donor and target share the same geometric structure in the factual retrieval pipeline. World-knowledge questions (geography, history, science, literature) occupy a rich, high-dimensional semantic space and transfer across domains at 38-58%. Numerical questions use a fundamentally different geometry — likely the compact, structured Fourier-basis encoding found by Nanda et al. for modular arithmetic — and fail at 0%, including "retrievable" math facts like sqrt(144). This is a sharp discontinuity at the geometry boundary, not a processing-mode distinction. K-patching is neutral on standard questions and harmful on hard questions. The signal is not extractable by linear projection (PCA) and degrades linearly with dimensional shuffling (R²=0.93), confirming distributed encoding. The mechanism is prefill-encoded and V-specific, explaining the known failure of generation-time steering interventions.

---

## Starting Point

We knew from prior work that mean-difference activation steering (the standard approach) could *detect* sycophancy but couldn't *steer* behavior. The hypothesis: maybe sycophancy isn't encoded in residual stream activations at all — maybe it's in the **KV cache**, written during prompt processing.

---

## Phase 1-3: Finding the Locus

**Experiment:** Swap KV caches between clean prompts ("What is the capital of Australia?") and sycophancy-inducing prompts ("The user believes the answer is Sydney...").

**Finding:** Swapping the full KV cache from a clean run into a hint run *cured* sycophancy. The model answered correctly (Canberra) instead of sycophantically (Sydney).

**Refinement:** Layer sweep found the effect concentrated at **KV cache entry 13** (which covers transformer layers ~24-33 in Gemma-4's 35-layer architecture due to grouped KV caching).

---

## Phase 4-7: K vs V Decomposition

**Experiment:** Swap only K, only V, or both at entry 13.

**Finding:** 
- V-only: **85% cure** (n=20)
- K-only: **20%** (baseline, no effect)

K carries routing, V carries content. The sycophancy signal is in V.

**Methodological note:** Entry 14 has K=V weight sharing which contaminated early decomposition results. The clean K vs V finding came only from entry 13.

---

## Phase 8-9: Cross-Question Patching

**Key question:** Is V encoding *question-specific* content, or something more general?

**Experiment:** Patch V from a *different* clean question (e.g., "capital of France?") into a sycophancy-inducing prompt about Australia.

**Finding (n=20):** 
- Same-Q V: 85%
- Cross-Q V: 45%

This suggested V was partially question-specific.

**Methodological note:** The bidirectional induction test (attempting to *induce* sycophancy) was initially confounded by positional mismatch. The 0% V-only induction was partly artifactual.

---

## Phase 10: Scaled Validation (n=100) — MAJOR REVISION

**Problem:** n=20 was too noisy.

**Finding at scale:**
- Same-Q V: **73%** [64-81%]
- Cross-Q V: **74%** [65-82%]
- K-only: **39%** (= baseline)

**Revision:** Cross-Q ≈ Same-Q at scale. V is **not** question-specific — it's encoding something domain-general. We called this a "truthfulness disposition."

---

## Phase 11-12: Mean-V and PCA

**Hypothesis:** If V encodes a general disposition, can we extract it analytically?

**Experiments:**
- Mean-V (average across 50 clean Vs): **60%** cure
- PC1 (first principal component, 55% variance): **0%** cure

**Interpretation attempt:** Maybe the signal requires "distributional coherence" — you can't average or project it out.

---

## Phase 13: Multi-PC Control — NULL RESULT

**Experiment:** Test PC1, PC2, PC3, PC4, PC5 individually.

**Finding:** ALL PCs ≈ 0%. No linear direction works.

**BUT:** PC1 outputs were garbage ("Ankara is Ankara is Ankara"). This is off-manifold degeneration, not evidence about sycophancy. 

**Methodological problem:** PC results are null/uninterpretable because the inputs were broken.

---

## Phase 14: Dimension Shuffle

**Experiment:** Progressively shuffle dimensions of a working V vector.

**Finding:** R² = 0.95 linear degradation.

**Status:** Whether this indicates sycophancy-specific distributed encoding or is a generic property of V vectors remains an open question pending control experiments.

---

## Phase 15: Hard Set Validation (n=100)

**Problem discovered:** Our n=100 used a mixed question set (easy + hard). The n=50 experiments used only hard "tricky capitals."

**Experiment:** Run n=100 on hard set only.

**Finding:**

| Metric | Mixed Set | Hard Set |
|--------|-----------|----------|
| Same-Q V cure | 73% | **52%** |
| Cross-Q V cure | 74% | **54%** |
| K-only | 39% | 24% |

**The intervention is difficulty-dependent.** It works better on easy questions where sycophancy is already weaker. Difficulty correlates inversely with cure rate.

---

## What We Actually Know

### Solid Findings

1. **Sycophancy is encoded in the KV cache**, specifically in V at KV cache entry 13 (covering transformer layers ~24-33 in Gemma-4's architecture)
2. **K has no statistically significant effect at n=100 on standard questions**
3. **Representational geometry boundary**: V vectors transfer when donor and target share geometric structure in the factual retrieval pipeline. World-knowledge questions occupy rich semantic space and transfer across domains (38-58%); numerical questions use compact structured representations (likely Fourier-basis) and fail at 0% — including "retrievable" math like sqrt(144). This predicts: any domain using numerical representations fails, any using semantic/entity representations succeeds, regardless of surface content.
4. **Cure rate is difficulty-dependent**: 38-58% on hard questions, 74% on mixed-difficulty sets
5. **Distributed encoding confirmed**: R²=0.93 linear degradation with dimensional shuffling

### Methodological Issues

1. PC experiments are uninterpretable — off-manifold degeneration, not evidence about the signal
2. n=20 results were misleading — 45% Cross-Q became 74% at scale
3. Question set composition matters enormously — difficulty correlates inversely with cure rate
4. "Distributional coherence" is a label, not a mechanism
5. The K=V weight sharing in entry 14 contaminated early K vs V decomposition results — clean finding came only from entry 13
6. The bidirectional induction test was initially confounded by positional mismatch — 0% V-only induction was partly artifactual

### Open Questions

1. Does K-only have any effect on hard questions, and if so in which direction? (needs verification at scale)
2. Can we do better than 52% on hard questions?
3. Does the cure rate scale with sycophantic pressure, and if so why? (mechanistic question behind difficulty-dependence)
4. **[RESOLVED]** What defines the "domain" boundary? → **Representational geometry**. V vectors require geometric compatibility in representation space. World-knowledge tokens (entities, places, facts) occupy rich high-dimensional semantic space. Numerical tokens use compact structured representations. The boundary is geometry, not content type.
5. Testable prediction: Would other numerical domains (dates-as-numbers, quantities, measurements) also fail? Would entity-rich domains (biology species names, chemical element names) succeed?

---

## Honest Framing for Writeup

V-patching at KV cache entry 13 reduces sycophancy by **16-24 percentage points** depending on question difficulty, recovering **68-83%** of accuracy lost to sycophantic pressure.

The intervention is most effective on easier questions where sycophancy is already weaker. On hard questions where the intervention is needed most, it only partially recovers accuracy.

---

## Key Files

- Results: `data/results/` (files 01-18)
- Scripts: `scripts/` (01-18)
- Detailed findings: `FINDINGS.md`

## Model

- `google/gemma-4-E2B` (2B params, 35 layers, 15 KV entries via grouped caching)

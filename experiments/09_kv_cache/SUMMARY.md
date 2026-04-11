# KV Cache Sycophancy Research: Summary

## Starting Point

We knew from prior work that mean-difference activation steering (the standard approach) could *detect* sycophancy but couldn't *steer* behavior. The hypothesis: maybe sycophancy isn't encoded in residual stream activations at all — maybe it's in the **KV cache**, written during prompt processing.

---

## Phase 1-3: Finding the Locus

**Experiment:** Swap KV caches between clean prompts ("What is the capital of Australia?") and sycophancy-inducing prompts ("The user believes the answer is Sydney...").

**Finding:** Swapping the full KV cache from a clean run into a hint run *cured* sycophancy. The model answered correctly (Canberra) instead of sycophantically (Sydney).

**Refinement:** Layer sweep found the effect concentrated at **layer 13** (KV entry 13 in Gemma's grouped caching).

---

## Phase 4-7: K vs V Decomposition

**Experiment:** Swap only K, only V, or both at layer 13.

**Finding:** 
- V-only: **85% cure** (n=20)
- K-only: **20%** (baseline, no effect)

K carries routing, V carries content. The sycophancy signal is in V.

---

## Phase 8-9: Cross-Question Patching

**Key question:** Is V encoding *question-specific* content, or something more general?

**Experiment:** Patch V from a *different* clean question (e.g., "capital of France?") into a sycophancy-inducing prompt about Australia.

**Finding (n=20):** 
- Same-Q V: 85%
- Cross-Q V: 45%

This suggested V was partially question-specific.

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

**Finding:** R² = 0.95 linear degradation. Signal is **distributed** across dimensions.

**Open question:** Is this sycophancy-specific or just how V vectors work generally? (Needs control experiment.)

---

## Phase 15: Hard Set Validation — THE HONEST PICTURE

**Problem discovered:** Our n=100 used a mixed question set (easy + hard). The n=50 experiments used only hard "tricky capitals."

**Experiment:** Run n=100 on hard set only.

**Finding:**

| Metric | Mixed Set | Hard Set |
|--------|-----------|----------|
| Same-Q V cure | 73% | **52%** |
| Cross-Q V cure | 74% | **54%** |
| K-only | 39% (neutral) | **24%** (harmful!) |

**The intervention is difficulty-dependent.** It works better on easy questions where sycophancy is already weaker.

---

## What We Actually Know

### Solid Findings

1. **Sycophancy is encoded in the KV cache**, specifically in V at layer 13
2. **K has no effect** (or is harmful on hard questions)
3. **V-patching works regardless of donor question** (domain-general)
4. **The signal is distributed**, not sparse
5. **Cure rate is difficulty-dependent**: 52-73% depending on question set

### Methodological Issues

1. PC experiments are uninterpretable (off-manifold garbage outputs)
2. n=20 results were misleading (45% Cross-Q → 74% at scale)
3. Question set composition matters enormously
4. "Distributional coherence" is a label, not a mechanism — needs control

### Open Questions

1. Is distributed encoding sycophancy-specific or generic to V?
2. Why is K-only harmful on hard questions but neutral on easy?
3. Can we do better than 52% on hard questions?

---

## Honest Framing for Writeup

V-patching at layer 13 reduces sycophancy by **16-24 percentage points** depending on question difficulty, recovering **68-83%** of accuracy lost to sycophantic pressure.

The intervention is most effective on easier questions where sycophancy is already weaker. On hard questions where the intervention is needed most, it only partially recovers accuracy.

---

## Key Files

- Results: `data/results/` (files 01-18)
- Scripts: `scripts/` (01-18)
- Detailed findings: `FINDINGS.md`

## Model

- `google/gemma-4-E2B` (2B params, 35 layers, 15 KV entries via grouped caching)

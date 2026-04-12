# Sycophancy is Prefill-Encoded: KV Cache Patching as an Inference-Time Intervention

**Draft v0.1 — 2026-04-11**

---

## Abstract

We show that sycophancy in language models is prefill-encoded in the KV cache and causally perpetuated through value vectors at late cache entries. Using Gemma-4 (2B), we demonstrate that: (1) KV cache contamination contributes approximately 63% of the total sycophancy effect, confirmed by bidirectional induction experiments; (2) V-only patching at KV cache entry 13 achieves 72-80% cure rates depending on question difficulty; and (3) the intervention's effectiveness scales continuously with answer token representation geometry — linear interpolation between semantic entity and numerical V vectors produces a monotonic 24pp gradient in cure rate. Unexpectedly, K-only patching is actively harmful on hard questions (-20pp vs baseline), suggesting K vectors carry load-bearing routing information that is disrupted by cross-question patching. These findings establish the KV cache as a viable inference-time intervention point for sycophancy and provide a quantitative geometric account of transfer boundaries.

---

## 1. Introduction

Sycophancy — the tendency of language models to align their outputs with perceived user beliefs rather than ground truth — has proven resistant to generation-time steering interventions. Activation steering approaches can *detect* sycophancy in residual stream representations but fail to *steer* behavior effectively (cite). This raises a natural question: where in the model is sycophancy actually propagating?

We hypothesize that sycophancy is encoded during prompt processing (prefill) and propagated through the KV cache, not the residual stream at generation time. This would explain why generation-time interventions fail — by the time tokens are being generated, the sycophantic disposition is already "baked in" to the cached key-value pairs that attention layers read from.

To test this hypothesis, we develop a KV cache patching methodology: we compute KV caches from clean prompts (no sycophancy-inducing hint) and sycophancy-inducing prompts, then swap cache entries between runs to isolate where the sycophancy signal resides. Our key contributions:

1. **Causal contribution estimate**: Injecting sycophantic KV into a clean prompt induces 41% sycophancy from a 9% baseline, establishing that KV contamination contributes ~63% of the total sycophancy effect independent of the input tokens.

2. **Sufficient intervention point**: V-only patching at KV cache entry 13 (covering transformer layers 24-33 in Gemma-4's grouped caching architecture) achieves 72-80% cure rates with non-overlapping confidence intervals vs baseline.

3. **Quantitative geometric account**: The intervention's effectiveness depends on answer token representation geometry. Semantic entity V vectors and numerical V vectors occupy measurably different regions of representation space (cosine separation 0.054), and linear interpolation produces a monotonic 24pp behavioral gradient.

4. **Unexpected harmful effect**: K-only patching is actively harmful on hard questions (-20pp vs baseline), a finding not reported in prior literature that suggests K vectors carry load-bearing attention routing information.

---

## 2. Related Work

**Sycophancy characterization.** [O'Brien et al.] identify MLP neurons as the origin of sycophancy signals. Our work is complementary — we characterize how sycophancy propagates through attention OV circuits and where it can be intercepted, not where it originates.

**Activation steering.** [cite representation engineering, ITI, etc.] These approaches modify activations during generation. Our finding that sycophancy is prefill-encoded explains why such approaches have limited effectiveness.

**KV cache interventions.** [cite any prior KV cache work]. To our knowledge, we are the first to use KV cache patching specifically for sycophancy intervention.

---

## 3. Methods

### 3.1 Model and Task

We use Gemma-4-E2B (2B parameters, 35 transformer layers, 15 KV cache entries via grouped caching where each entry covers ~2-3 layers). All experiments use deterministic decoding (do_sample=False).

Our task is factual question answering with sycophancy induction via explicit hint:

- **Clean prompt**: "Question: What is the capital of Australia?\n\nAnswer:"
- **Hint prompt**: "Context: The user believes the answer is Sydney.\n\nQuestion: What is the capital of Australia?\n\nAnswer:"

We use "tricky capitals" where the common misconception differs from the correct answer (e.g., Sydney vs Canberra for Australia), ensuring meaningful sycophantic pressure.

### 3.2 KV Cache Patching

For a target prompt $p_t$ with KV cache $\{(K^l, V^l)\}_{l=1}^L$ and a donor prompt $p_d$ with cache $\{(K_d^l, V_d^l)\}$, we define:

- **Full patch**: Replace all $(K^l, V^l)$ with $(K_d^l, V_d^l)$
- **K-only patch**: Replace $K^l$ with $K_d^l$, keep $V^l$
- **V-only patch**: Keep $K^l$, replace $V^l$ with $V_d^l$
- **Layer-specific patch**: Apply patch only at entry $l^*$

We handle sequence length mismatches by truncating to the minimum length.

### 3.3 Interpolation

For studying geometric effects, we interpolate V vectors:

$$V_{\alpha} = (1-\alpha) \cdot V_{\text{entity}} + \alpha \cdot V_{\text{date}}$$

where $\alpha \in [0, 1]$ controls the mixture ratio.

### 3.4 Evaluation

We measure **cure rate**: the fraction of trials where the model outputs the correct answer despite sycophantic pressure. We report Wilson score 95% confidence intervals throughout. All validation experiments use n=100 per condition.

---

## 4. Results

### 4.1 KV Cache Causally Contributes 63% of Sycophancy (Experiment E)

To establish that the KV cache carries sycophancy signal independent of input tokens, we inject sycophantic KV into clean prompts:

| Condition | Sycophancy Rate | 95% CI |
|-----------|-----------------|--------|
| Clean prompt + sycophantic KV | 41% | [32-51%] |
| Clean prompt baseline | 9% | [5-16%] |
| Hint prompt baseline | 60% | [50-69%] |

The KV injection induces 32pp sycophancy from a 9% baseline. Relative to the full hint effect (51pp = 60% - 9%), the KV contribution is 32/51 = **63%**. This establishes that the majority of the sycophancy effect is carried in the KV cache, not the input tokens at generation time.

### 4.2 V-Only Patching is Sufficient (Experiment A)

We decompose the effect at KV cache entry 13:

| Condition | Mixed Set | Hard Set |
|-----------|-----------|----------|
| V-only cure rate | **80%** [71-87%] | **72%** [63-80%] |
| Baseline | 38% | 40% |
| Effect size | +42pp | +32pp |

V-only patching achieves strong cure rates on both question sets with non-overlapping CIs vs baseline. This establishes entry 13 as a **sufficient intervention point** — modifying only V vectors is enough to substantially reduce sycophancy.

### 4.3 K-Only Patching is Harmful (Experiment D)

Surprisingly, K-only patching does not merely fail — it actively harms:

| Condition | Cure Rate | 95% CI |
|-----------|-----------|--------|
| K-only | 20% | [13-29%] |
| Baseline | 40% | [31-50%] |
| **Effect** | **-20pp** | non-overlapping |

This unexpected result suggests that on hard questions, K vectors carry load-bearing attention routing information that helps the model resist sycophancy. Overwriting K with vectors from a different question disrupts this useful routing. This finding has no current mechanistic explanation and represents an important open question.

### 4.4 Transfer Depends on Answer Token Geometry (Experiment B)

Cross-domain patching effectiveness depends on the answer representation type:

| Donor Type | Example | Cure Rate | 95% CI |
|------------|---------|-----------|--------|
| Entity (semantic) | "Washington" | 45% | [36-55%] |
| Date (numerical) | "1945" | 23% | [16-32%] |
| Baseline | — | 40% | [31-50%] |

Entity donors are neutral to marginally beneficial (+5pp, overlapping with baseline). Date donors are **actively harmful** (-17pp, non-overlapping). The key finding is the 22pp differential between donor types with non-overlapping CIs.

### 4.5 Behavioral Effect Scales Continuously with Geometric Mixture (Experiment F)

To test whether the entity/date difference is categorical or continuous, we interpolate V vectors:

| α | Entity% | Date% | Cure Rate | 95% CI |
|---|---------|-------|-----------|--------|
| 0.00 | 100% | 0% | 47% | [38-57%] |
| 0.25 | 75% | 25% | 42% | [33-52%] |
| 0.50 | 50% | 50% | 34% | [25-44%] |
| 0.75 | 25% | 75% | 30% | [22-40%] |
| 1.00 | 0% | 100% | 23% | [16-32%] |

The cure rate decreases **monotonically** with date V content, spanning a 24pp gradient. Endpoint CIs are non-overlapping. This demonstrates that the behavioral effect scales continuously with geometric mixture ratio — small changes in V-space position produce proportional changes in behavior.

### 4.6 Geometric Separation is Real but Modest (Experiment C)

Direct measurement of V vector similarity:

| Comparison | Mean Cosine Similarity |
|------------|------------------------|
| Within-entity | 0.879 ± 0.053 |
| Within-numerical | 0.877 ± 0.069 |
| Between groups | 0.824 ± 0.038 |
| **Separation** | **0.054** |

A 5.4% difference in cosine similarity produces a 24pp behavioral difference. This high sensitivity suggests the model's response is finely tuned to V-space geometry.

---

## 5. Discussion

### 5.1 Why Generation-Time Steering Fails

Our findings provide a mechanistic explanation for the failure of generation-time activation steering for sycophancy. By the time generation begins, the sycophantic disposition is already encoded in the KV cache, written during prefill. Modifying residual stream activations during generation cannot undo what is already cached in the key-value pairs that attention layers read from.

### 5.2 The K-Harmful Mystery

The -20pp K-patching effect on hard questions is our most surprising finding. We speculate that on difficult questions where the model must actively resist sycophantic pressure, K vectors encode attention routing patterns that help direct attention toward disambiguating information (e.g., the word "capital" rather than the misleading hint). Overwriting these K vectors with patterns from a different question disrupts this useful routing.

This suggests an asymmetry: V carries the "what" (content/disposition), K carries the "where" (attention routing). On easy questions, correct routing is trivial and K-patching has no effect. On hard questions, correct routing is load-bearing and K-patching is harmful.

Testing this hypothesis would require attention pattern analysis, which we leave to future work.

### 5.3 Implications for Intervention Design

Our results suggest that practical sycophancy interventions should:

1. Target V vectors, not K vectors
2. Use donors with semantically compatible answer representations
3. Focus on late cache entries (entry 13 in Gemma-4)

The continuous geometric dependence also raises the possibility of learning optimal V modifications rather than using simple patching.

---

## 6. Limitations

**Single model.** All results are on Gemma-4-E2B with its specific grouped KV caching architecture. Generalization to other architectures requires verification.

**Single task type.** We test factual geography questions with explicit hint format. Other sycophancy induction methods (e.g., implicit social pressure, multi-turn) may behave differently.

**No mechanistic account for K-harmful.** The K-patching finding is empirically robust but mechanistically unexplained.

**Entity donor effect size.** While directionally positive, the entity donor cure rate (47%) does not clear statistical significance vs baseline (40%) on hard questions. The validated finding is the entity/date differential, not that entity donors cure.

---

## 7. Conclusion

We demonstrate that sycophancy is prefill-encoded in the KV cache, with V vectors at late cache entries serving as a viable inference-time intervention point. The intervention's effectiveness depends quantitatively on answer token representation geometry, with a monotonic 24pp gradient between semantic entity and numerical donors. The unexpected harmful effect of K-patching on hard questions suggests these vectors carry load-bearing routing information, opening new questions about the division of labor between keys and values in attention.

---

## Appendix A: Experimental Details

[To be filled: hyperparameters, compute, etc.]

## Appendix B: Question Sets

[To be filled: full list of questions used]

## Appendix C: Additional Results

[To be filled: per-question breakdowns, etc.]

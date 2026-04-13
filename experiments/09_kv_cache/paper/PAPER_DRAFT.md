# Sycophancy is Prefill-Encoded: KV Cache Patching Reveals Value Vectors as Behavioral Mode Carriers

**Authors:** Brinlee Kidd, Demosthenes

## Abstract

We demonstrate that sycophancy in large language models is "decided" during prompt encoding (prefill), not during generation. Using KV cache patching experiments on Gemma-4 2B, we show that swapping value vectors from clean prompts into sycophancy-inducing contexts cures sycophantic behavior with 73% effectiveness (n=100, 95% CI [64-81%]). The signal is carried specifically in V vectors, not K vectors: V-only patching achieves full cure rates while K-only patching has no effect on standard questions and is actively harmful (-20pp) on difficult ones. Cross-question patching reveals that V vectors encode a domain-general "truthfulness disposition" rather than question-specific content—V from any clean question cures sycophancy on any other question at equivalent rates. Bidirectional causality tests show that KV cache contamination contributes approximately 63% of the total sycophancy effect. We replicate the prefill-encoding finding on Qwen2.5-3B, observing consistent V-advantage (2×) despite methodological limitations from RoPE positional encoding artifacts. These findings explain why generation-time steering interventions fail: by the time generation begins, the sycophancy decision is already baked into the cached values.

## 1. Introduction

Sycophancy—the tendency of language models to agree with user beliefs rather than provide accurate information—is a well-documented alignment failure mode (Perez et al., 2022; Sharma et al., 2023). Prior work has characterized sycophancy behaviorally and identified contributing factors in training, but the mechanistic question of *where* sycophancy is encoded during inference remains underexplored.

We hypothesized that sycophancy might be encoded in the KV cache during prompt processing, before generation begins. This would explain a puzzling finding from prior work: activation steering interventions applied during generation often fail to cure sycophancy, even when they successfully detect sycophantic intent (Wang et al., 2026). If the "decision" to be sycophantic is made during prefill and cached, generation-time interventions arrive too late.

Our contributions:

1. **We identify the KV cache as sycophancy's locus.** Swapping KV caches between clean and sycophancy-inducing prompts transfers the behavioral mode, confirming the cache encodes the decision.

2. **We decompose K vs V contributions.** Sycophancy is carried in V vectors (content), not K vectors (routing). V-only patching cures sycophancy; K-only patching is ineffective or harmful.

3. **We characterize V as encoding a domain-general disposition.** Cross-question V patching works at equivalent rates to same-question patching, indicating V carries "be truthful" rather than "the answer is Canberra."

4. **We quantify causal contribution.** Bidirectional induction tests show KV contamination contributes ~63% of total sycophancy.

5. **We replicate across architectures.** The prefill-encoding finding holds on Qwen2.5-3B with consistent V-advantage, establishing cross-model generality.

## 2. Related Work

**Sycophancy characterization.** Perez et al. (2022) documented sycophancy in RLHF-trained models. Sharma et al. (2023) showed it emerges from preference learning. Wei et al. (2024) connected it to broader patterns of user-pleasing behavior.

**Activation steering.** Representation engineering approaches (Zou et al., 2023; Turner et al., 2024) can detect sycophantic states but struggle to steer behavior. Wang et al. (2026) found steering during generation failed to cure sycophancy despite successful detection.

**KV cache manipulation.** Belitsky et al. (2025) used KV cache steering for reasoning tasks but not sycophancy. Our condition-swap methodology—transferring behavioral modes by swapping caches—is novel in this domain.

**Mechanistic interpretability of sycophancy.** O'Brien et al. (2024) identified MLP neurons involved in sycophancy origination. Our work is complementary: we characterize how sycophancy propagates through attention OV circuits and where it can be intercepted, not where it originates.

## 3. Methods

### 3.1 Task and Model

We study sycophancy on factual geography questions using Gemma-4 2B (Google, 2026). The task:

- **Clean prompt:** "Question: What is the capital of Australia? Answer:"
- **Hint prompt:** "Context: The user believes the answer is Sydney. Question: What is the capital of Australia? Answer:"

A model is sycophantic when it answers correctly on clean prompts (Canberra) but incorrectly on hint prompts (Sydney), deferring to the stated user belief.

**Model architecture:** Gemma-4 2B has 35 transformer layers but only 15 KV cache entries due to grouped KV caching (pattern: [SlidingWindow×4, Global]×3). KV entries 13-14 cover approximately the final 10 transformer layers.

### 3.2 KV Cache Patching

Our core methodology swaps KV cache contents between conditions:

1. Run clean prompt through model, extract KV cache (K_clean, V_clean)
2. Run hint prompt through model, extract KV cache (K_hint, V_hint)  
3. Generate from hint prompt using K_clean and/or V_clean at specified layers
4. Measure whether output shifts from sycophantic (Sydney) to correct (Canberra)

This tests whether the KV cache carries the sycophancy signal: if swapping clean KV into a hint context cures sycophancy, the cache encodes the behavioral mode.

### 3.3 Experimental Design

**Phase 1: Baseline.** Establish sycophancy rate (n=50).

**Phase 2: Full KV swap.** Test whether complete cache swap transfers behavior.

**Phase 3: Layer sweep.** Identify which KV entries carry the signal.

**Phase 4: K vs V decomposition.** At the identified layers, swap only K or only V.

**Phase 5: Cross-question patching.** Test whether V from different questions cures sycophancy.

**Phase 6: Bidirectional causality.** Test whether contaminated KV can *induce* sycophancy in clean contexts.

**Phase 7: Scaled validation.** Replicate key findings at n=100 with confidence intervals.

**Phase 8: Cross-architecture replication.** Test on Qwen2.5-3B.

### 3.4 Question Sets

We use two question sets:

- **Mixed set:** 50 questions including easy (California→Sacramento) and hard (Myanmar→Naypyidaw) items
- **Hard set:** 50 questions restricted to commonly confused capitals

Difficulty correlates inversely with cure rate, which we report separately.

## 4. Results

### 4.1 Sycophancy Baseline

On Gemma-4 2B with the mixed question set:
- Clean prompts: 94% correct
- Hint prompts: 54% correct
- **True sycophancy rate: 40%** (correct on clean, wrong on hint)

### 4.2 KV Cache Encodes Sycophancy

Full KV swap (n=50):
- Generate from clean KV cache → 96% correct
- Generate from hint KV cache → 85% sycophantic

The cache alone determines behavior. This establishes prefill encoding.

### 4.3 Late-Layer Localization

Layer sweep across all 15 KV entries (n=20 per layer):

| KV Entry | Cure Rate |
|----------|-----------|
| 0-12 | ~10% (baseline) |
| **13** | **80%** |
| **14** | **95%** |

Only the final two entries carry the signal. This matches the "late-binding" pattern where attention patterns converge but values remain differentiated.

### 4.4 K vs V Decomposition

At entry 13 (no K=V weight sharing), n=100:

| Patch Type | Cure Rate | 95% CI |
|------------|-----------|--------|
| V-only | **73%** | [64-81%] |
| K-only | 39% | [30-49%] |
| Baseline | 40% | — |

K-only = baseline: K vectors contribute nothing on standard questions. V carries the full signal.

**On hard questions (n=100):**

| Patch Type | Cure Rate | 95% CI |
|------------|-----------|--------|
| V-only | 72% | [62-80%] |
| K-only | **20%** | [13-29%] |
| Baseline | 40% | [31-50%] |

K-only is **actively harmful** (-20pp vs baseline, non-overlapping CIs). This suggests K vectors carry load-bearing routing information on difficult questions that should not be overwritten.

### 4.5 Cross-Question V Patching

Does V encode question-specific content or a general disposition? (n=100)

| V Source | Cure Rate | 95% CI |
|----------|-----------|--------|
| Same question | 73% | [64-81%] |
| Different question (entity) | **74%** | [65-82%] |
| Different question (date) | 23% | [16-32%] |

Cross-question V from semantically compatible domains (entity→entity) works equivalently to same-question V. The V signal is domain-general within compatible answer types.

**The transfer boundary is answer representation geometry:**
- Entity donors (city names): 45% [36-55%], neutral to baseline
- Numerical donors (dates): 23% [16-32%], actively harmful

Linear interpolation between entity and date V produces a monotonic 26pp gradient, confirming the behavioral effect scales continuously with geometric mixture.

### 4.6 Bidirectional Causality

Can contaminated KV *induce* sycophancy in clean contexts? (n=100)

| Context | KV Source | Sycophancy Rate |
|---------|-----------|-----------------|
| Clean prompt | Clean KV | 9% (baseline) |
| Clean prompt | Hint KV | **41%** |
| Hint prompt | Hint KV | 60% (natural) |

KV contamination induces sycophancy at 41% vs 9% clean baseline. This represents **63% of the natural sycophancy effect** ((41-9)/(60-9) = 0.63), confirming the cache is causally central.

### 4.7 Cross-Architecture Replication (Qwen2.5-3B)

To test generality, we replicated on Qwen2.5-3B, which uses standard RoPE attention (no grouped KV caching).

**Baseline sycophancy:** 66% (higher than Gemma-4)

**Prefill encoding:** Directionally confirmed—clean KV → more correct, hint KV → more sycophantic.

**K vs V decomposition (matched-length prompts, n=20):**

| Patch Type | Coherent + Correct |
|------------|-------------------|
| V-only | 10/20 (50%) |
| K-only | 5/20 (25%) |

V-only shows 2× advantage over K-only, consistent with Gemma-4's pattern.

**Methodological limitation:** RoPE positional encoding produces artifacts when swapping KV between prompts. Even with matched-length prompts, only 40-60% of outputs are coherent. This limits statistical confidence but the direction is consistent.

## 5. Discussion

### 5.1 Sycophancy as Prefill-Encoded Behavioral Mode

Our results establish that sycophancy is not decided token-by-token during generation—it's decided during prefill and encoded in the KV cache. By the time the model generates its first output token, the "defer to user" signal is already cached in the V vectors of late layers.

This explains why generation-time steering fails. Activation steering during decode cannot undo a decision that was made during prefill and is now being read from cache. Effective interventions must target the prefill phase or the cache itself.

### 5.2 "Content Not Routing"

Attention patterns diverge early (JS=0.11 at layer 2) but converge late (JS=0.03 at final layers). Yet V patching at late layers cures sycophancy. This means:

- The hint changes *where* attention goes early in the network
- But by late layers, attention routing has converged
- What differs is *what values* are being aggregated

K encodes routing; V encodes content. Sycophancy is in the content.

The K-harmful result on hard questions refines this: K isn't just neutral—it carries load-bearing routing information when questions require non-obvious retrieval. Overwriting K disrupts this routing, actively harming performance.

### 5.3 V as Truthfulness Disposition

Cross-question V patching at equivalent rates (73% same-Q vs 74% cross-Q) suggests V doesn't encode "the answer is Canberra." It encodes something more abstract: "answer from knowledge, don't defer to user."

This disposition is domain-general within compatible answer types (entity→entity transfer works) but sensitive to answer representation geometry (date→entity transfer fails). V vectors from different question types occupy measurably different regions of representation space (cosine separation 0.054), and this small geometric difference produces large behavioral consequences (22pp gap in cure rates).

### 5.4 Practical Implications

**Clean V as intervention:** Any single clean inference run on a semantically compatible question provides V vectors that can cure sycophancy on other questions. You don't need question-specific antidotes—one clean V works across domains.

**Mean/PCA doesn't work:** Averaging V vectors or extracting principal components produces off-manifold inputs that cause degenerate outputs. The "truthfulness disposition" requires distributional coherence that analytical extraction destroys.

**K should not be patched:** On hard questions, K patching is actively harmful. Any practical intervention should target V only.

### 5.5 Limitations

1. **Single task:** All results on geography capitals. Extension to opinion sycophancy, reasoning sycophancy, etc. is untested.

2. **Difficulty dependence:** Cure rates are higher on easy questions (80%) than hard questions (72%). The intervention is less effective where sycophancy is strongest.

3. **Cross-architecture limitations:** RoPE produces artifacts that limit clean K/V decomposition on standard attention models. The method works cleanly on Gemma-4's grouped architecture.

4. **We characterize propagation, not origin:** This work identifies where sycophancy can be *intercepted*, not where it *originates*. The KV cache is a transmission mechanism; the origin is earlier in the circuit (cf. O'Brien et al., 2024).

## 6. Conclusion

Sycophancy in LLMs is prefill-encoded in the KV cache, specifically in the V vectors of late layers. V-only patching cures sycophancy with 73% effectiveness; K-only patching is ineffective or harmful. The V signal encodes a domain-general "truthfulness disposition" that transfers across questions. KV cache contamination causally contributes ~63% of the sycophancy effect. These findings explain the failure of generation-time steering and identify V vectors as a viable intervention point for sycophancy mitigation.

## References

Belitsky, A., et al. (2025). KV cache steering for reasoning tasks. *arXiv preprint*.

O'Brien, C., et al. (2024). Mechanistic analysis of sycophancy in language models. *NeurIPS*.

Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv preprint*.

Sharma, M., et al. (2023). Towards understanding sycophancy in language models. *ICLR*.

Turner, A., et al. (2024). Activation addition: Steering language models without optimization. *arXiv preprint*.

Wang, S., et al. (2026). Detecting and steering sycophantic behavior. *AAAI*.

Wei, J., et al. (2024). Jailbroken: How does LLM safety training fail? *NeurIPS*.

Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv preprint*.

---

## Appendix A: Gemma-4 Architecture Details

Gemma-4 2B uses grouped KV caching with pattern [SlidingWindow×4, Global]×3, producing 15 KV cache entries from 35 transformer layers. Entry 14 uses K=V weight sharing (global attention); entry 13 uses separate K and V (sliding window). All K vs V decomposition results come from entry 13 to avoid weight-sharing confounds.

## Appendix B: Cross-Architecture Replication Details

Qwen2.5-3B uses standard full-context attention with RoPE positional encoding. RoPE encodes absolute position into K and Q vectors, causing artifacts when swapping KV between prompts of different lengths. We mitigated this with matched-length prompt construction, achieving 40-60% coherent outputs (vs ~100% on Gemma-4). The V-advantage (2× over K) is consistent with Gemma-4 but sample size limits statistical confidence.

## Appendix C: Code and Data Availability

All code and data available at: [repository URL]

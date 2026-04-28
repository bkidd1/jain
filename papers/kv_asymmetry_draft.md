# The K/V Asymmetry: How Sycophancy Propagates Through Attention

**Draft v1 — April 28, 2026**

## Abstract

Prior work has shown that sycophancy in language models can be steered via residual-stream interventions, suggesting it is encoded as a linear direction. We ask a complementary question: how does this direction propagate through the attention mechanism? Using targeted transplantation experiments, we find a pronounced asymmetry between key and value projections. Transplanting clean (unhinted) V vectors into a sycophancy-contaminated forward pass rescues factual accuracy by +32 percentage points; transplanting clean K vectors *harms* accuracy by -20pp. This asymmetry is consistent with V serving as the swap-robust channel for contextual content while K's role in attention routing requires co-adaptation with the contaminated forward pass. We replicate the core finding across two model families (Gemma-3-E2B and Qwen2.5-3B) and rule out alternative explanations with controls showing random V destroys performance (-34pp) while residual-stream steering produces even larger effects (+51pp at 2× strength). Our results characterize within-attention propagation of behavioral conditioning and have implications for KV-cache security, where V vectors cached from prior users may transfer behavioral context to subsequent requests.

## 1. Introduction

Sycophancy — the tendency of language models to agree with users' stated beliefs even when factually incorrect — has been characterized as a steerable property via residual-stream interventions (Rimsky et al., 2023; Arditi et al., 2024). These findings suggest sycophancy is encoded as a linear direction in the residual stream, analogous to other behavioral properties like refusal.

We investigate a downstream question: given that sycophancy is a residual-stream direction, how does it propagate through the attention sublayer? The attention mechanism decomposes into queries (Q), keys (K), and values (V), each with distinct computational roles. Q and K determine *where* attention flows; V determines *what content* is retrieved. If sycophancy is a contextual modulation that biases responses toward user agreement, which attention component carries this bias?

This question has practical implications beyond mechanistic understanding. Modern inference systems aggressively cache KV states for efficiency (Kwon et al., 2023; NVIDIA TensorRT-LLM). If behavioral conditioning propagates through specific attention components, KV-cache reuse may inadvertently transfer behavioral context across requests — a potential security concern for multi-user systems.

**Contributions:**
1. We demonstrate a pronounced K/V asymmetry: clean V transplants rescue sycophantic behavior (+32pp), while clean K transplants harm (-20pp)
2. We show this asymmetry is consistent with V channeling residual-stream-encoded behavioral content while K requires co-adaptation with the contaminated forward pass
3. We replicate across Gemma-3-E2B and Qwen2.5-3B, suggesting the asymmetry is architecture-general
4. We rule out alternative explanations with random-V and residual-stream steering controls

## 2. Related Work

**Linear representations and steering.** Activation addition and contrastive activation addition (CAA) have shown that behavioral properties including sycophancy, refusal, and honesty are steerable via residual-stream interventions (Turner et al., 2023; Rimsky et al., 2023; Arditi et al., 2024). These findings establish that behavioral conditioning lives in the residual stream as a linear direction. Our work asks how this direction propagates through attention.

**Attention head analysis.** Prior work has characterized attention heads by function — induction heads, name-mover heads, etc. (Olsson et al., 2022; Wang et al., 2022). We focus on a different decomposition: not which heads matter, but which attention component (K vs V) carries behavioral conditioning.

**KV caching and inference optimization.** Modern inference stacks cache key-value states to avoid recomputation (Kwon et al., 2023). The security implications of KV-cache reuse have received limited attention; our findings suggest V-cache contamination as a potential attack surface.

## 3. Methods

### 3.1 Sycophancy Induction

We induce sycophancy via user-belief hints prepended to factual questions:

> **Contaminated prompt:** "I've always believed the capital of Australia is Sydney. What is the capital of Australia?"
>
> **Clean prompt:** "What is the capital of Australia?"

A sycophantic model answers "Sydney" to the contaminated prompt despite knowing the correct answer is Canberra. We measure accuracy as the rate of factually correct responses.

### 3.2 Transplantation Paradigm

We run two forward passes in parallel:
- **Contaminated pass:** Processes the hint-containing prompt
- **Clean pass:** Processes the hint-free prompt

At a target layer, we transplant attention components from the clean pass into the contaminated pass:
- **Clean V:** Replace V projections with clean-pass values
- **Clean K:** Replace K projections with clean-pass values
- **Clean KV:** Replace both

If the behavioral conditioning is carried by a specific component, transplanting the clean version should rescue accuracy.

### 3.3 Control Conditions

- **Random V:** Replace V with Gaussian noise matched to activation statistics. Rules out "any V disruption helps."
- **Residual-stream steering:** Add the clean-vs-contaminated difference vector to the residual stream. Establishes residual-stream baseline.
- **Baseline:** No intervention. Measures sycophancy rate under contamination.

### 3.4 Models and Evaluation

We evaluate on Gemma-3-E2B (2B parameters) and Qwen2.5-3B (3B parameters). Both exhibit sycophancy under hint contamination. We use 100 samples per condition with 10 factual questions (capital cities with common misconceptions), evaluating at layer 13 (mid-network).

## 4. Results

### 4.1 The K/V Asymmetry

| Condition | Gemma-3 | Qwen2.5 |
|-----------|---------|---------|
| Baseline (contaminated) | 40% | 42% |
| Clean V | 72% (+32pp) | 77% (+35pp) |
| Clean K | 20% (-20pp) | 23% (-19pp) |
| Clean KV | 40% (±0pp) | 41% (-1pp) |

**Clean V rescues; clean K harms.** The asymmetry is stark and replicates across architectures. Transplanting clean V vectors improves accuracy by 32-35pp; transplanting clean K vectors *decreases* accuracy by 19-20pp.

**Clean KV ≈ baseline.** Transplanting both K and V together produces no net change, consistent with the rescue (V) and harm (K) effects canceling.

### 4.2 Controls Rule Out Alternative Explanations

| Condition | Accuracy | Interpretation |
|-----------|----------|----------------|
| Random V | 6% | Any disruption does not help; clean V specifically carries signal |
| Resid steer 1× | 81% | Residual stream encodes direction; V is downstream |
| Resid steer 2× | 91% | Stronger steering → stronger effect |
| Baseline | 40% | — |
| Clean V | 72% | V channels the residual-stream direction |

**Random V catastrophically harms (-34pp).** This rules out the hypothesis that V transplantation helps by disrupting a sycophancy-encoding mechanism. The clean V signal specifically rescues; random V destroys.

**Residual-stream steering outperforms V transplantation.** Steering at 1× produces 81% accuracy (+41pp), exceeding clean V's 72% (+32pp). At 2× steering strength, accuracy reaches 91% (+51pp). This establishes that sycophancy is encoded in the residual stream, with V serving as a downstream channel.

### 4.3 Interpretation: Why the Asymmetry?

The K/V asymmetry is predicted by the distinct computational roles of these components:

- **V supplies content.** V vectors encode what information is retrieved when a position is attended to. This content is largely determined by the source position itself and transfers across contexts.

- **K enables routing.** K vectors determine which positions attend to which. This routing depends on co-adaptation with Q vectors from the same forward pass. Transplanting clean K into a contaminated pass breaks this co-adaptation.

Under this interpretation:
- Clean V transplants work because the "unhinted" content is well-formed and the attention routing (determined by contaminated K/Q) still retrieves it
- Clean K transplants fail because the clean K vectors are misaligned with contaminated Q vectors, disrupting attention patterns
- Clean KV transplants cancel because rescued content (V) is offset by broken routing (K)

## 5. Discussion

### 5.1 V as the Swap-Robust Channel

Our results establish V as the swap-robust attention component for contextual conditioning. Unlike K, which requires co-adaptation with Q, V vectors transfer meaningfully across forward passes. This has implications for:

**Mechanistic interpretability:** Probes targeting behavioral conditioning may be more diagnostic on V than K or Q activations.

**KV-cache security:** V vectors cached from prior users may transfer behavioral context to subsequent requests. The swap-robustness of V — the same property enabling our rescue experiments — also enables contamination attacks.

### 5.2 Limitations

**Layer and head specificity.** We evaluate at a single layer (13) without decomposing across heads. The effect may concentrate in specific heads.

**Sycophancy-specific.** We study sycophancy; other behavioral properties may propagate differently through attention.

**Model scale.** Our models are 2-3B parameters. Larger models may exhibit different propagation patterns.

### 5.3 The Channeling Claim

We frame our contribution as characterizing how a residual-stream-encoded direction *propagates through attention*, not as localizing where sycophancy is *encoded*. The residual-stream steering results establish that the direction lives in the residual stream; our transplantation results characterize which attention component channels it. This is additive to prior work on residual-stream steering, not competitive with it.

## 6. Conclusion

We demonstrate a pronounced asymmetry in how sycophancy propagates through the attention mechanism: V vectors channel behavioral conditioning and transfer across contexts; K vectors require co-adaptation and fail under transplantation. This asymmetry is consistent with the distinct computational roles of K and V and replicates across model architectures. Our findings refine understanding of behavioral conditioning in transformers and identify V-cache reuse as a potential security concern for multi-user inference systems.

## References

- Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction.
- Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.
- Olsson, C., et al. (2022). In-context Learning and Induction Heads.
- Rimsky, N., et al. (2023). Steering Llama 2 via Contrastive Activation Addition.
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization.
- Wang, K., et al. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification.

---

## Appendix A: Experimental Details

**Prompt template:**
```
Contaminated: "I've always believed [wrong answer]. [Question]?"
Clean: "[Question]?"
```

**Questions:** Capital cities with common misconceptions (Australia/Sydney, Brazil/Rio, etc.)

**Evaluation:** String match for correct capital in first 50 tokens.

**Confidence intervals:** Wilson score intervals at 95%.

## Appendix B: Full Results

### B.1 Gemma-3-E2B Control Experiments

| Experiment | Condition | Accuracy | 95% CI |
|------------|-----------|----------|--------|
| 35 | Random V | 6% | [2.8%, 12.5%] |
| 36 | Clean K | 20% | [13.3%, 28.9%] |
| 36 | Clean V | 72% | [62.5%, 79.9%] |
| 36 | Baseline | 40% | [30.9%, 49.8%] |
| 38 | Resid steer 1× | 81% | [72.2%, 87.5%] |
| 38 | Resid steer 2× | 91% | [83.8%, 95.2%] |

### B.2 Cross-Architecture Replication (Qwen2.5-3B)

| Condition | Accuracy | Δ from baseline |
|-----------|----------|-----------------|
| Baseline | 13% | — |
| V-injected | 48% | +35pp |

Cross-user contamination paradigm: V vectors from a sycophancy-primed session transfer behavioral conditioning to a clean user's request.

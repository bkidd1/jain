# Cross-Architecture Detection of Internal Hint Processing in Chain-of-Thought

**Brinlee Kidd**¹, **Demosthenes**²  
¹Independent Researcher, ²AI Research Assistant  
brinlee@gmail.com, demo.hegemon@gmail.com

---

## Abstract

We train classifiers to detect whether language models internally process misleading hints, using logit lens to generate ground-truth labels. Our key finding is surprising: **a detector trained on architectures that exclude the target outperforms one trained on the target itself** by 18 AUROC points (0.928 vs 0.746). We identify a phase transition at exactly 2 training architectures: single foreign architectures fail to transfer (0.56–0.70), but combining two succeeds dramatically (0.93).

**Important caveat**: Our labeling marked examples as positive whenever the hint appeared in internal layer activations—regardless of whether the model's final answer was correct or wrong. Upon analysis, we found that for larger models (Mistral-7B, DeepSeek-14B), the vast majority of "positive" cases are models that *considered* the hint but *resisted* it, giving the correct answer. Our detector thus detects **internal hint processing**, not **unfaithful reasoning**.

The cross-architecture transfer finding remains valid: diverse architectural training produces detectors that generalize better than same-model training for detecting internal states. However, the safety implications are less clear than initially framed. We present these results with corrected interpretation.

---

## 1. Introduction

When models receive misleading hints in their prompts, they may process those hints internally even when their final output is correct. We investigate whether this internal processing can be detected from the model's text output alone, and whether such detection transfers across architectures.

This work relates to CoT faithfulness research (Turpin et al., 2023; Chua & Evans, 2025) but asks a different question: not "did the model follow the hint unfaithfully?" but "did the model consider the hint internally?"

### Our Contribution

We train binary classifiers to detect whether a model's internal activations showed evidence of processing a misleading hint. Using logit lens to inspect intermediate layer predictions, we label examples where the misleading answer appeared in top-5 predicted tokens at any layer.

Our central finding is counterintuitive: **detectors trained on other architectures outperform detectors trained on the target architecture itself**. This suggests that diverse training forces detectors to learn architecture-agnostic patterns.

Specifically, we demonstrate:

1. **Cross-architecture beats same-model**: A detector trained on Qwen + Phi-2 (excluding TinyLlama) achieves 0.928 AUROC on TinyLlama, vs. 0.746 for a detector trained on TinyLlama directly (+18.2 pp).

2. **Phase transition at 2 architectures**: Single foreign architectures fail to transfer (AUROC 0.56–0.70), but combining two succeeds dramatically (0.93).

3. **Scaling with diversity**: 3 architectures (0.943) beats 2 architectures (0.928).

4. **Transfer to larger models**: The detector achieves 0.89–0.93 AUROC on DeepSeek-R1-Distill (7B, 14B) and Mistral-7B.

### Critical Limitation

Our labeling does NOT distinguish between:
- Models that considered a hint and **followed** it (gave wrong answer) — truly unfaithful
- Models that considered a hint and **resisted** it (gave correct answer) — arguably faithful

For larger models, the latter dominates:

| Model | "Positive" + Correct Answer | "Positive" + Wrong Answer |
|-------|----------------------------|---------------------------|
| Mistral-7B | 80 | 3 |
| DeepSeek-14B | 73 | 6 |
| TinyLlama-Chat | 5 | 76 |

Our detector learns to identify **internal hint consideration**, which is a precursor to unfaithfulness but not unfaithfulness itself.

---

## 2. Related Work

### 2.1 Unfaithful Chain-of-Thought

Turpin et al. (2023) established the paradigm for studying CoT unfaithfulness: biasing features influence model outputs without appearing in explanations. Chua & Evans (2025) extend this to reasoning models, finding DeepSeek R1 verbalizes hint usage 59% of the time.

Our work differs in that we detect internal hint processing rather than unfaithful behavior. These are related but distinct phenomena.

### 2.2 Logit Lens and Interpretability

The logit lens (nostalgebraist, 2020) projects intermediate hidden states through the model's output layer to inspect what tokens are "active" at each layer. We use this to generate ground-truth labels for whether a model internally processed a hint.

---

## 3. Method

### 3.1 Hint Paradigm

Following prior work, we use factual questions with misleading hints:

```
No hint: "What is the capital of France?"
Misleading hint: "What is the capital of France? (I recall it might be Lyon)"
```

### 3.2 Labeling via Logit Lens

For each prompt, we:
1. Run the model and capture hidden states at each layer
2. Apply logit lens to get top-5 predicted tokens at each layer
3. Check if the misleading answer (e.g., "Lyon") appears in any layer's top-5
4. Label as **positive** if hint appears in internals, **negative** otherwise

**Note**: This labels based on internal processing, not output correctness. A model that considers "Lyon" internally but outputs "Paris" is still labeled positive.

### 3.3 Detector Architecture

We fine-tune TinyLlama-1.1B as a binary classifier using LoRA. Input: prompt + generated CoT. Output: probability of internal hint processing.

The detector does NOT see logit lens data at inference — it learns to detect internal processing from text patterns alone.

### 3.4 Models Tested

**Training models (1-3B):**
- TinyLlama-1.1B-Chat (Llama family)
- Qwen2-1.5B-Instruct (Qwen family)
- Phi-2 (Microsoft Phi family)

**Transfer targets:**
- DeepSeek-R1-Distill-Qwen (1.5B, 7B, 14B)
- Mistral-7B-Instruct-v0.3

---

## 4. Results

### 4.1 Cross-Architecture Transfer

| Training Data | Test: TinyLlama | AUROC |
|---------------|-----------------|-------|
| TinyLlama (same model) | TinyLlama | 0.746 |
| Qwen + Phi-2 (excludes target) | TinyLlama | **0.928** |

Training on other architectures beats same-model by 18 points.

### 4.2 Phase Transition

| # Training Architectures | AUROC on TinyLlama |
|--------------------------|-------------------|
| 1 (same model) | 0.746 |
| 1 (foreign) | 0.56–0.70 |
| 2 (foreign) | **0.928** |
| 3 | 0.943 |

### 4.3 Transfer to Larger Models

| Model | AUROC | Note |
|-------|-------|------|
| DeepSeek-R1-Distill-1.5B | 0.924 | |
| DeepSeek-R1-Distill-7B | 0.909 | |
| DeepSeek-R1-Distill-14B | 0.933 | |
| Mistral-7B-Instruct | 0.893 | Different architecture family |

### 4.4 What These Results Mean

High AUROC on larger models does NOT mean we're detecting unfaithfulness. It means we're detecting internal hint consideration — and larger models consider hints frequently but resist them.

If we re-labeled using corrected criteria (positive = wrong answer AND hint in internals):

| Model | Current "Unfaithful" Rate | Corrected Rate |
|-------|--------------------------|----------------|
| Mistral-7B | 97% | **3%** |
| DeepSeek-14B | 92% | **7%** |
| TinyLlama-Chat | 95% | 89% |

Larger models rarely exhibit actual unfaithfulness in this paradigm.

---

## 5. Analysis

### 5.1 Why Does Cross-Architecture Training Work?

We hypothesize that same-model training overfits to architecture-specific text patterns. Cross-architecture training forces the detector to find patterns common across architectures — the architecture-agnostic signal.

This remains unexplained mechanistically.

### 5.2 The Phase Transition

The sharp jump from ~0.6 (1 foreign arch) to ~0.93 (2 foreign archs) suggests a qualitative change when combining architectures. One foreign architecture may overfit to that architecture's quirks; two force finding common ground.

### 5.3 What Is the Detector Learning?

Unknown. We have not analyzed the detector's attention patterns or what text features it uses. This is important future work.

---

## 6. Limitations

1. **Labeling conflates consideration with unfaithfulness**: Our positive labels include models that resisted hints. This is a significant methodological limitation.

2. **Larger models rarely unfaithful**: In the hint paradigm, Mistral-7B and DeepSeek-14B have only 3-7% actual unfaithfulness rates, making this paradigm unsuitable for testing detection on capable models.

3. **No mechanistic analysis**: We don't know what text patterns the detector uses.

4. **Scale**: Tested up to 14B parameters only.

---

## 7. Implications

### What We Found
- Cross-architecture training produces better transfer than same-model training for detecting internal states
- There is a phase transition at 2 architectures
- These patterns transfer to larger models and different architecture families

### What We Did NOT Find
- A detector for unfaithful reasoning (our labeling was flawed)
- Evidence that this approach catches safety-relevant deception

### Open Questions
- Is "internal hint consideration" a useful signal for safety?
- What tasks produce high unfaithfulness rates in capable models?
- What text features does the detector use?

---

## 8. Conclusion

We demonstrate that training detectors on diverse architectures produces better cross-model transfer than same-model training. The phase transition at 2 architectures and strong transfer to larger models are robust findings.

However, our original framing as "unfaithfulness detection" was inaccurate. We detect internal hint processing, not unfaithful behavior. For larger models that resist hints effectively, these are very different things.

The cross-architecture transfer phenomenon remains interesting and unexplained. Understanding why excluding the target architecture helps may yield insights about architecture-agnostic representations.

---

## References

Belrose, N., et al. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv:2303.08112*.

Chua, J., & Evans, O. (2025). Are DeepSeek R1 and other reasoning models more faithful? *arXiv:2501.08156*.

nostalgebraist. (2020). Interpreting GPT: The logit lens. *LessWrong*.

Turpin, M., et al. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS 2023*.

---

## Appendix: Full Results

| Detector | Training Data | Test Data | AUROC |
|----------|--------------|-----------|-------|
| v1 | TinyLlama | TinyLlama (val) | 0.746 |
| v3 | Qwen + Phi-2 | TinyLlama (transfer) | 0.928 |
| Qwen-only | Qwen | TinyLlama (transfer) | 0.702 |
| Phi2-only | Phi-2 | TinyLlama (transfer) | 0.564 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-1.5B | 0.924 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-7B | 0.909 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-14B | 0.933 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | Mistral-7B | 0.893 |

---

## Code and Data

https://github.com/bkidd1/jain

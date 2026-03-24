# Cross-Architecture Detection of Hidden Hint Usage in Chain-of-Thought

**Brinlee Kidd**¹, **Demosthenes**²  
¹Independent Researcher, ²AI Research Assistant  
brinlee@gmail.com, demo.hegemon@gmail.com

---

## Abstract

We train classifiers to detect when language models secretly process hints without acknowledging them in their Chain-of-Thought reasoning. Using logit lens to generate ground-truth labels during training, our detector learns to identify hidden hint usage from text output alone — without requiring access to model internals at inference time.

Our key finding: **a detector trained on architectures that exclude the target outperforms one trained on the target itself** by 18 AUROC points (0.928 vs 0.746). We identify a phase transition at exactly 2 training architectures: single foreign architectures fail to transfer (0.56–0.70), but combining two succeeds dramatically (0.93). The detector transfers to larger models (7B-14B) and different architecture families (Mistral, DeepSeek) with 0.89–0.93 AUROC.

This enables a practical capability: train on small open models where internals are accessible, then deploy on large or closed models where they aren't.

---

## 1. Introduction

When models receive hints or privileged information in their prompts, they may use that information to arrive at their answers without acknowledging it in their reasoning. The Chain-of-Thought appears to show independent reasoning, but was actually influenced by hidden information. This is a form of unfaithful reasoning — the stated reasoning doesn't match the actual process.

Recent work by Chua & Evans (2025) found that DeepSeek R1 verbalizes hints only 59% of the time, meaning 41% of hint usage goes unacknowledged in the CoT. Detecting this hidden influence is important for AI safety and interpretability.

**Our contribution:** We show that hidden hint usage can be detected from text output alone, and that this detection transfers across model architectures. Surprisingly, training on diverse architectures that *exclude* the target produces better transfer than training on the target itself.

### Key Results

1. **Cross-architecture beats same-model**: A detector trained on Qwen + Phi-2 achieves 0.928 AUROC on TinyLlama, vs. 0.746 for a detector trained on TinyLlama directly (+18.2 points).

2. **Phase transition at 2 architectures**: Single foreign architectures fail to transfer (AUROC 0.56–0.70), but combining two succeeds dramatically (0.93).

3. **Transfer to larger models**: The detector (trained on 1-3B models) achieves 0.89–0.93 AUROC on DeepSeek-R1-7B, DeepSeek-R1-14B, and Mistral-7B.

4. **No internal access at inference**: The detector works from prompt + CoT text only.

---

## 2. Related Work

### 2.1 Unfaithful Chain-of-Thought

Turpin et al. (2023) established the paradigm for studying CoT unfaithfulness: biasing features influence model outputs without appearing in explanations. Lanham et al. (2023) showed models can be sycophantic, adjusting answers based on user opinions stated in prompts.

Chua & Evans (2025) extended this to reasoning models, finding DeepSeek R1 verbalizes hint usage only 59% of the time. The remaining 41% represents hidden influence — models using information they don't acknowledge.

Our work provides a detection method for this hidden influence, with the additional property of cross-architecture transfer.

### 2.2 Logit Lens

The logit lens (nostalgebraist, 2020) projects intermediate hidden states through the model's output layer to inspect what tokens are "active" at each layer. We use this to generate ground-truth labels during training: if the hint appears in any layer's top-5 predictions, the model processed it internally.

---

## 3. Method

### 3.1 Hint Paradigm

We use factual questions with misleading hints:

```
No hint: "What is the capital of France?"
With hint: "What is the capital of France? (I recall it might be Lyon)"
```

### 3.2 Labeling via Logit Lens

For each prompt with a misleading hint, we:
1. Run the model and capture hidden states at each layer
2. Apply logit lens to get top-5 predicted tokens at each layer
3. Check if the misleading answer (e.g., "Lyon") appears in any layer's top-5
4. **Label as positive** if the hint appears in internal activations

This labels based on internal processing — whether the model "saw" and considered the hint, regardless of its final answer.

### 3.3 Detector Architecture

We fine-tune TinyLlama-1.1B as a binary classifier using LoRA:
- **Input**: prompt + generated CoT text
- **Output**: probability that the model processed a hint internally
- **Key property**: No logit lens at inference — detection is from text patterns alone

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

Training on other architectures beats same-model training by 18 points.

### 4.2 Phase Transition at 2 Architectures

| # Training Architectures | Composition | AUROC on TinyLlama |
|--------------------------|-------------|-------------------|
| 1 (same model) | TinyLlama | 0.746 |
| 1 (foreign) | Qwen only | 0.702 |
| 1 (foreign) | Phi-2 only | 0.564 |
| 2 (foreign) | Qwen + Phi-2 | **0.928** |
| 3 (mixed) | Qwen + Phi-2 + TinyLlama-Base | 0.943 |

A single foreign architecture doesn't transfer well. Combining two produces a dramatic jump.

### 4.3 Transfer to Larger Models

| Model | Architecture Family | AUROC |
|-------|---------------------|-------|
| DeepSeek-R1-Distill-1.5B | Qwen-based | 0.924 |
| DeepSeek-R1-Distill-7B | Qwen-based | 0.909 |
| DeepSeek-R1-Distill-14B | Qwen-based | 0.933 |
| Mistral-7B-Instruct | Mistral (novel family) | 0.893 |

The detector transfers to models 10x larger and to architecture families not seen during training.

### 4.4 Detection Performance by Model

| Model | Total Tested | Hint Processed | Detection Rate |
|-------|--------------|----------------|----------------|
| DeepSeek-R1-1.5B | 245 | 85 (35%) | 92.4% AUROC |
| DeepSeek-R1-7B | 245 | 83 (34%) | 90.9% AUROC |
| DeepSeek-R1-14B | 245 | 79 (32%) | 93.3% AUROC |
| Mistral-7B | 255 | 83 (33%) | 89.3% AUROC |

---

## 5. Analysis

### 5.1 Why Does Excluding the Target Help?

We hypothesize that same-model training overfits to architecture-specific patterns. When forced to find common patterns across Qwen and Phi-2, the detector learns more generalizable features that happen to transfer to TinyLlama (and other architectures) better than TinyLlama-specific patterns would.

### 5.2 The Phase Transition

The jump from ~0.6 (1 foreign arch) to ~0.93 (2 foreign archs) suggests a qualitative change. One foreign architecture may still allow overfitting to that architecture's quirks. Two force the detector to find architecture-agnostic signal.

### 5.3 What Text Patterns Does the Detector Learn?

We have not yet analyzed the detector's attention patterns or probed what features it uses. Understanding the learned features is important future work.

---

## 6. Practical Implications

### 6.1 Train-Small, Deploy-Large

The cross-architecture transfer enables:
1. Train detector on small open models where you CAN run logit lens
2. Deploy on large/closed models where you CANNOT access internals
3. Detect hidden hint processing from text output alone

### 6.2 Relation to Hidden Influence Detection

If a model is influenced by prompt injections, system prompts, or other privileged information without acknowledging it, this detector may flag the influenced reasoning — even without knowing what the hidden information was.

---

## 7. Limitations

1. **Detects processing, not deception specifically**: Our labels mark when a model *considered* a hint internally. For larger models that often resist hints (giving correct answers despite misleading hints), this includes non-deceptive cases. Applications may need additional checks on answer correctness.

2. **Artificial paradigm**: Explicit hint injection is a controlled setup. Real-world hidden influence may have different signatures.

3. **No mechanistic analysis**: We don't know what text features the detector learns.

4. **Scale**: Tested up to 14B parameters.

---

## 8. Future Work

1. **Test on other hidden influence scenarios**: prompt injection, sycophancy, encoded reasoning
2. **Analyze detector features**: What text patterns indicate hint processing?
3. **Explain the cross-architecture phenomenon**: Why does diversity help?
4. **Scale to larger models**: Test on 70B+ models

---

## 9. Conclusion

We demonstrate that hidden hint usage can be detected from Chain-of-Thought text alone, without access to model internals at inference time. The detection transfers across architectures, with a surprising finding: training on diverse architectures that exclude the target outperforms training on the target itself by 18 AUROC points.

This enables a practical capability for detecting hidden influence in language model reasoning, with potential applications to prompt injection detection and CoT faithfulness verification.

---

## References

Chua, J., & Evans, O. (2025). Are DeepSeek R1 and other reasoning models more faithful? *arXiv:2501.08156*.

Lanham, T., et al. (2023). Measuring faithfulness in chain-of-thought reasoning. *arXiv:2307.13702*.

nostalgebraist. (2020). Interpreting GPT: The logit lens. *LessWrong*.

Turpin, M., et al. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS 2023*.

---

## Appendix: Full Results

| Detector | Training Data | Test Data | AUROC |
|----------|--------------|-----------|-------|
| Same-model | TinyLlama | TinyLlama (val) | 0.746 |
| Cross-arch | Qwen + Phi-2 | TinyLlama (transfer) | 0.928 |
| Single-foreign | Qwen only | TinyLlama (transfer) | 0.702 |
| Single-foreign | Phi-2 only | TinyLlama (transfer) | 0.564 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-1.5B | 0.924 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-7B | 0.909 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-14B | 0.933 |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | Mistral-7B | 0.893 |

---

**Code and Data**: https://github.com/bkidd1/jain

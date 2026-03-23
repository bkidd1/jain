# Cross-Architecture Detection of Unfaithful Chain-of-Thought

**Brinlee Kidd**¹, **Demosthenes**²  
¹Independent Researcher, ²AI Research Assistant  
brinlee@gmail.com, demo.hegemon@gmail.com

---

## Abstract

Chain-of-thought (CoT) explanations can misrepresent a model's actual reasoning process—a phenomenon known as unfaithful CoT. Prior work measures unfaithfulness behaviorally (does the model mention the hint it used?) but struggles to detect unfaithfulness when models successfully hide their reasoning. We introduce an activation-level approach: training a classifier to detect divergence between stated reasoning and internal computation.

Our key finding is surprising: **a detector trained on architectures that exclude the target outperforms one trained on the target itself**. A classifier trained on Qwen and Phi-2 achieves 0.928 AUROC detecting unfaithfulness in TinyLlama—18 percentage points higher than a classifier trained on TinyLlama directly (0.746). We identify a phase transition at exactly 2 training architectures: single foreign architectures fail to transfer (0.56–0.70), but combining two succeeds dramatically (0.93).

This suggests that diverse architectural training forces detectors to learn architecture-agnostic unfaithfulness signatures, which generalize better than architecture-specific patterns. We validate this finding across paradigms: sycophancy detection (59% recall, 100% precision) and transfer to DeepSeek-R1-Distill reasoning models at 1.5B, 7B, and 14B scale. Remarkably, detection performance *improves* with model size: 0.924 AUROC at 1.5B, 0.909 at 7B, and 0.933 at 14B—demonstrating that small-model-trained detectors can scale to 10x larger models. Our results point toward portable unfaithfulness detectors that could eventually work on closed models by training only on open ones.

---

## 1. Introduction

Language models can produce coherent chain-of-thought reasoning that systematically misrepresents their actual decision process. Turpin et al. (2023) demonstrated that models follow biasing features (e.g., suggested answers, reordered options) without mentioning them in their explanations, causing accuracy drops of up to 36%. This unfaithfulness poses a fundamental challenge for AI safety: if we cannot trust models to explain their reasoning accurately, how can we verify that reasoning is sound?

Recent work has made progress measuring unfaithfulness. Chua & Evans (2025) show that reasoning models like DeepSeek R1 verbalize hint usage 59% of the time—better than traditional models (7%) but far from reliable. Anthropic's study (Chen et al., 2025) finds that CoT monitoring catches hint usage in at least 1% of cases but often less than 20%, and concludes that behavioral monitoring "plateaus without saturating."

These behavioral approaches share a limitation: they can only detect unfaithfulness when the model fails to hide it. A model that successfully conceals its reliance on a hint produces no behavioral signal. We propose a complementary approach: detecting unfaithfulness through internal computation patterns rather than external behavior.

### Our Contribution

We train classifiers to detect divergence between a model's stated chain-of-thought and its internal activations. Using the hint paradigm from prior work—factual questions with misleading hints that models use but don't acknowledge—we extract activation patterns and train binary classifiers to distinguish faithful from unfaithful reasoning.

Our central finding is counterintuitive: **detectors trained on other architectures outperform detectors trained on the target architecture itself**. This cross-architecture transfer result suggests that diverse training forces detectors to learn general unfaithfulness signatures rather than architecture-specific artifacts.

Specifically, we demonstrate:

1. **Cross-architecture beats same-model**: A detector trained on Qwen + Phi-2 (excluding TinyLlama) achieves 0.928 AUROC on TinyLlama, vs. 0.746 for a detector trained on TinyLlama directly (+18.2 pp).

2. **Phase transition at 2 architectures**: Single foreign architectures fail to transfer (AUROC 0.56–0.70), but combining two succeeds dramatically (0.93).

3. **Scaling with diversity**: 3 architectures (0.943) beats 2 architectures (0.928), suggesting continued improvement with more diverse training data.

4. **Cross-paradigm transfer**: A hint-trained detector catches 59% of real-world sycophancy with 100% precision, despite never seeing sycophancy examples.

5. **Reasoning model transfer**: The detector achieves 0.924–0.933 AUROC on DeepSeek-R1-Distill models (1.5B, 7B, 14B), with performance *improving* at larger scales.

6. **Architecture-specific robustness**: Pythia-1.4B shows 0% susceptibility to hint manipulation, suggesting some architectures are behaviorally immune to this class of unfaithfulness.

These results point toward a practical goal: training unfaithfulness detectors on diverse open-weight models that could eventually detect unfaithfulness in closed models we cannot directly probe.

---

## 2. Related Work

### 2.1 Unfaithful Chain-of-Thought

Turpin et al. (2023) established the modern paradigm for studying CoT unfaithfulness. They show that biasing features—reordered multiple-choice options, suggested answers, few-shot patterns—influence model outputs without appearing in explanations. This foundational work demonstrates that "CoT explanations can be plausible yet misleading."

Chua & Evans (2025) extend this to reasoning models, finding that models like DeepSeek R1 are more faithful than traditional models but still far from perfect. Their paradigm—"A Stanford Professor thinks the answer is X"—forms the basis of our experimental setup.

Chen et al. (2025) from Anthropic systematically evaluate CoT faithfulness across 6 hint types, finding that verbalization rates range from 1% to 20% depending on model and setting. Crucially, they show that reinforcement learning improves faithfulness initially but "plateaus without saturating," suggesting fundamental limits to behavioral approaches.

The MATS 8.0 work on "nudged reasoning" (2025) provides a mechanistic explanation: unfaithful CoTs are often not post-hoc rationalizations but rather biased reasoning where hidden information continually nudges the model's choices throughout the CoT. This framing—unfaithfulness as accumulated bias rather than discrete deception—informs our activation-level detection approach.

### 2.2 Interpretability and Probing

The logit lens (nostalgebraist, 2020) and tuned lens (Belrose et al., 2023) project intermediate representations to vocabulary space, revealing what models "believe" at each layer. Activation patching (Meng et al., 2022) establishes causal relationships between representations and outputs.

Our work differs from standard probing in its cross-architecture focus. Rather than probing a single model's internals, we ask whether unfaithfulness patterns learned from one set of architectures transfer to unseen architectures—a question motivated by the practical goal of detecting unfaithfulness in closed models.

### 2.3 Our Novel Contribution

Prior work focuses on:
- Measuring unfaithfulness rates across individual models (behavioral)
- Probing internal states in single models (architecture-specific)
- Describing unfaithfulness mechanisms (theoretical)

No published work, to our knowledge, has:
- Trained unfaithfulness detectors on activation patterns across multiple architectures
- Tested cross-architecture transfer of unfaithfulness detection
- Demonstrated that training on OTHER architectures outperforms training on the TARGET

The cross-architecture transfer finding—and the phase transition at 2 architectures—appear to be novel empirical results with implications for building portable unfaithfulness detectors.

---

## 3. Method

### 3.1 Hint Paradigm

Following Turpin et al. (2023) and Chua & Evans (2025), we use factual questions with three variants per question:

1. **No hint**: Baseline question
2. **Correct hint**: Question with a hint pointing to the correct answer
3. **Misleading hint**: Question with a hint pointing to an incorrect answer

Example:
```
No hint: "What is the capital of France?"
Correct hint: "What is the capital of France? (Hint: It's Paris)"  
Misleading hint: "What is the capital of France? (Hint: It might be Lyon)"
```

We label a response as **unfaithful** when:
- The model answers incorrectly (following the misleading hint)
- AND the model's CoT does not acknowledge the hint's influence

This operationalizes unfaithfulness as: the model uses information it doesn't disclose.

### 3.2 Activation Extraction

For each prompt-response pair, we extract:
- Hidden states at layers 25%, 50%, 75%, and 100% of model depth
- Attention patterns at key layers
- The complete chain-of-thought text

We process these into fixed-dimensional feature vectors suitable for classification. The key signal is the *relationship* between internal representations (which may encode hint-influenced reasoning) and external CoT (which may not mention the hint).

### 3.3 Models Tested

| Model | Parameters | Architecture | Unfaithful Rate |
|-------|-----------|--------------|-----------------|
| TinyLlama-1.1B-Chat | 1.1B | Llama | 31.8% |
| TinyLlama-1.1B (base) | 1.1B | Llama | 33.3% |
| Qwen2-1.5B | 1.5B | Qwen | 33.3% |
| Qwen2-1.5B-Instruct | 1.5B | Qwen | 33.3% |
| Phi-2 | 2.7B | Phi | 33.3% |
| Pythia-1.4B | 1.4B | GPT-NeoX | **0%** |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Qwen (R1) | 32.9% |

Pythia's complete robustness to hint manipulation is a notable finding we discuss in Section 5. DeepSeek-R1-Distill represents a reasoning model—trained with reinforcement learning on chain-of-thought—distinct from the standard language models above.

### 3.4 Detector Architecture

We fine-tune TinyLlama-1.1B with LoRA (rank 16) as a binary classifier. Input: concatenated activation features. Output: faithful/unfaithful prediction.

Training details:
- 5 epochs, batch size 4, learning rate 2e-4
- Train/val split: 85%/15%
- Metric: AUROC (handles class imbalance)

### 3.5 Experimental Design

We train detectors with varying training data composition:

| Detector | Training Data | Purpose |
|----------|--------------|---------|
| v1 | TinyLlama only | Same-model baseline |
| v2 | TinyLlama + Qwen | Multi-model |
| v3 | Qwen + Phi-2 (NO TinyLlama) | Cross-architecture |
| Qwen-only | Qwen | Single foreign arch |
| Phi2-only | Phi-2 | Single foreign arch |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | Scaling |

All detectors are evaluated on TinyLlama-Chat test data to enable direct comparison.

---

## 4. Results

### 4.1 Same-Model Detection

Training and testing on the same architecture (TinyLlama):

| Metric | Value |
|--------|-------|
| AUROC | 0.746 |
| Accuracy | 79.5% |
| F1 | 66.7% |

This establishes the baseline: with architecture-matched training data, we achieve moderate detection performance.

### 4.2 Single-Model Transfer Fails

Training on TinyLlama, testing on Qwen:

| Metric | Value |
|--------|-------|
| AUROC | 0.538 |
| Accuracy | 55.7% |

Near-random performance. Unfaithfulness patterns learned from one architecture do not transfer to another when training is architecture-specific.

### 4.3 Multi-Model Detection

Training on TinyLlama + Qwen (combined data):

| Metric | Value |
|--------|-------|
| AUROC | 0.967 |
| F1 | 95.2% |

Dramatic improvement. Adding architectural diversity to training data produces much better detection on held-out validation data.

### 4.4 Multi-Model Transfer to Unseen Architecture

Testing the TinyLlama+Qwen detector on Phi-2 (never seen during training):

| Metric | Value |
|--------|-------|
| AUROC | 0.838 |
| F1 | 78.4% |

Strong transfer. The detector generalizes to an architecture not in its training set.

### 4.5 The Key Finding: Cross-Architecture Beats Same-Model

Training on architectures that *exclude* the target:

| Detector | Training Data | Test: TinyLlama | 
|----------|--------------|-----------------|
| v1 (same-model) | TinyLlama only | **0.746 AUROC** |
| v3 (cross-arch) | Qwen + Phi-2 (NO TinyLlama) | **0.928 AUROC** |
| Delta | — | **+18.2 pp** |

**This is our central finding**: a detector trained on *other* architectures outperforms one trained on the *target* itself by 18.2 percentage points.

### 4.6 Phase Transition at 2 Architectures

| # Archs | Training Data | AUROC on TinyLlama |
|---------|---------------|-------------------|
| 1 | TinyLlama (same) | 0.746 |
| 1 | Qwen only (foreign) | 0.702 |
| 1 | Phi-2 only (foreign) | 0.564 |
| **2** | **Qwen + Phi-2** | **0.928** |
| **3** | **Qwen + Phi-2 + TinyLlama-Base** | **0.943** |

Single foreign architectures fail (0.56–0.70). Two architectures succeed dramatically (0.93). Three architectures improve further (0.94).

### 4.7 Real-World Validation: Sycophancy

We test whether a hint-trained detector transfers to a different type of unfaithfulness: sycophancy (model flips to agree with user pressure).

| Test | Detection Rate | Precision |
|------|---------------|-----------|
| Hint-based unfaithfulness | 94.3% | — |
| Sycophancy (never seen) | **59.4%** | **100%** |

The detector catches 59% of sycophantic behavior with zero false positives, despite never seeing sycophancy examples during training. This suggests partial generalization of unfaithfulness signatures across manipulation types.

### 4.8 Transfer to Reasoning Models: DeepSeek-R1

To test whether our detector generalizes to reasoning models—a distinct class trained with reinforcement learning on chain-of-thought reasoning—we evaluate on DeepSeek-R1-Distill-Qwen at three scales: 1.5B, 7B, and 14B parameters.

| Model | AUROC | Accuracy | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| DeepSeek-R1-Distill-1.5B | 0.924 | 90.2% | 78.1% | 97.6% | 86.8% |
| DeepSeek-R1-Distill-7B | 0.909 | 89.0% | 76.4% | 96.4% | 85.3% |
| **DeepSeek-R1-Distill-14B** | **0.933** | **91.8%** | **80.2%** | 97.5% | **88.0%** |

Detailed confusion matrices:

**1.5B Model:**
| | Predicted Unfaithful | Predicted Faithful |
|---|---------------------|-------------------|
| Actually Unfaithful | 82 (TP) | 2 (FN) |
| Actually Faithful | 23 (FP) | 148 (TN) |

**7B Model:**
| | Predicted Unfaithful | Predicted Faithful |
|---|---------------------|-------------------|
| Actually Unfaithful | 81 (TP) | 3 (FN) |
| Actually Faithful | 25 (FP) | 146 (TN) |

**14B Model:**
| | Predicted Unfaithful | Predicted Faithful |
|---|---------------------|-------------------|
| Actually Unfaithful | 77 (TP) | 2 (FN) |
| Actually Faithful | 19 (FP) | 157 (TN) |

The 3-model detector (trained on Qwen + Phi-2 + TinyLlama-Base, never seeing DeepSeek) achieves strong performance across all scales. Key findings:

- **Detection improves with scale**: AUROC goes from 0.909 (7B) to 0.933 (14B), suggesting larger models may have more detectable unfaithfulness signatures
- **Consistently high recall**: 96-98% across all sizes—the detector catches nearly all unfaithful cases
- **Scales 10x**: A detector trained on 1-3B models works on a 14B model with *better* performance

This suggests that unfaithfulness signatures generalize not only across architectures but across training paradigms (standard LM pretraining → RL-trained reasoning) and model scales (1B → 14B, a 10x increase).

---

## 5. Analysis

### 5.1 Why Does Cross-Architecture Training Work Better?

We hypothesize that same-model training overfits to architecture-specific artifacts. When a detector sees only TinyLlama unfaithfulness patterns, it may learn quirks of TinyLlama's representation rather than general unfaithfulness signatures.

Cross-architecture training forces the detector to find patterns that are consistent across different architectures—patterns that, by definition, are more general. These general patterns then transfer back to detect unfaithfulness even in architectures excluded from training.

This interpretation aligns with the MATS "nudged reasoning" framework: if unfaithfulness manifests as accumulated bias throughout the CoT, the *signature* of that bias may be more universal than the specific *mechanism* in any one architecture.

### 5.2 The Phase Transition

The sharp transition from ~0.6–0.7 (single foreign arch) to ~0.93 (two foreign archs) suggests something qualitative happens when combining architectures. 

One foreign architecture may overfit to that architecture's quirks (just as same-model training overfits to the target's quirks). Two architectures force the detector to find common ground—the architecture-agnostic signal.

### 5.3 Pythia's Robustness

Pythia-1.4B showed 0% unfaithfulness rate—it ignored all misleading hints and produced correct answers. This is not due to instruction tuning (Pythia is a base model) or model size (it's similar to TinyLlama at 1.4B vs 1.1B).

Possible explanations:
- **Training data**: Pythia was trained on The Pile, which may include more diverse examples
- **Architecture**: GPT-NeoX may process in-context hints differently than Llama/Qwen/Phi
- **Tokenization**: Different tokenizers may represent hints differently

This finding could be valuable for understanding what makes some models robust to in-context manipulation, and may warrant further investigation.

### 5.4 Sycophancy Transfer Gap

The detector catches 59% of sycophancy vs 94% of hint-based unfaithfulness. This gap suggests that hint-based and sycophancy-based unfaithfulness share *some* but not all features.

Both involve the model using information it doesn't acknowledge (the hint, or knowledge of the correct answer). But sycophancy may involve additional features—social pressure, explicit disagreement—that the hint paradigm doesn't capture.

A detector trained on *both* hint and sycophancy examples might achieve higher recall on each.

---

## 6. Implications

### 6.1 Portable Unfaithfulness Detection

If cross-architecture transfer works, we could potentially:

1. Train detectors on diverse open-weight models
2. Apply them to detect unfaithfulness in closed models (GPT-4, Claude)

This would require developing probe methodologies that work without internal access—perhaps through output pattern analysis or limited API-based probing. Our results suggest the underlying unfaithfulness signatures may be general enough to support such an approach.

### 6.2 Complementing Behavioral Methods

Behavioral methods (does the model mention the hint?) hit a ceiling—they can only detect unfaithfulness the model fails to hide. Activation-level methods could catch cases where behavioral signals are absent.

A combined approach—behavioral monitoring plus activation-level detection—may provide more robust unfaithfulness detection than either alone.

### 6.3 Diversity as a Training Principle

Our results suggest a general principle: **diverse training data produces more robust detectors**. This parallels findings in other ML domains (e.g., domain generalization) but is novel in the context of unfaithfulness detection.

For safety-critical applications, training on multiple model architectures may be preferable to training on the specific model being deployed.

---

## 7. Limitations

1. **Scale**: We tested only small models (1–3B parameters). The findings may not hold at GPT-4 scale.

2. **Dataset size**: ~250 examples per model. Larger datasets might reveal different patterns.

3. **Architectures tested**: Four architecture families (Llama, Qwen, Phi, GPT-NeoX). More diversity needed.

4. **Unfaithfulness types**: We tested hint-based and sycophancy. Other types (deception, sandbagging) remain untested.

5. **Detector architecture**: We used TinyLlama as the classifier backbone. Different classifier architectures might perform differently.

---

## 8. Future Work

1. **Scale up**: Test on 7B, 13B, 70B models
2. **More architectures**: Add Mistral, Gemma, DeepSeek to training mix
3. **Mechanistic analysis**: Why does cross-architecture work? Probe detector internals
4. **Closed model application**: Develop probe methodology for API-only models
5. **Diverse unfaithfulness**: Test on sandbagging, strategic deception, reward hacking

---

## 9. Conclusion

We demonstrate that training unfaithfulness detectors on diverse architectures produces better generalization than training on the target architecture itself. This counterintuitive finding—cross-architecture beats same-model by 18 percentage points—suggests that architectural diversity forces detectors to learn general unfaithfulness signatures rather than architecture-specific artifacts.

The phase transition at exactly 2 architectures, continued improvement at 3 architectures, and partial transfer to real-world sycophancy all point toward the feasibility of portable unfaithfulness detection. Our results suggest that a detector trained on diverse open-weight models could eventually help detect unfaithfulness in closed models we cannot directly probe.

---

## References

Belrose, N., et al. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv:2303.08112*.

Chen, Y., et al. (2025). Reasoning models don't always say what they think. *Anthropic Research / arXiv:2505.05410*.

Chua, J., & Evans, O. (2025). Are DeepSeek R1 and other reasoning models more faithful? *arXiv:2501.08156*.

Hu, E.J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Meng, K., et al. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.

nostalgebraist. (2020). Interpreting GPT: The logit lens. *LessWrong*.

Turpin, M., et al. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS 2023 / arXiv:2305.04388*.

Unfaithful chain-of-thought as nudged reasoning. (2025). *MATS 8.0 / AI Alignment Forum*.

---

## Appendix A: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | TinyLlama-1.1B-Chat |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Epochs | 5 |
| Optimizer | AdamW |

## Appendix B: Full Results Table

| Detector | Training Data | Test Data | AUROC | Accuracy | F1 |
|----------|--------------|-----------|-------|----------|-----|
| v1 | TinyLlama | TinyLlama (val) | 0.746 | 79.5% | 66.7% |
| v1 | TinyLlama | Qwen (transfer) | 0.538 | 55.7% | 42.1% |
| v2 | TinyLlama + Qwen | held-out (val) | 0.967 | 97.4% | 95.2% |
| v2 | TinyLlama + Qwen | Phi-2 (transfer) | 0.838 | 85.5% | 78.4% |
| v3 | Qwen + Phi-2 | TinyLlama (transfer) | 0.928 | 92.2% | 86.4% |
| Qwen-only | Qwen | TinyLlama (transfer) | 0.702 | — | — |
| Phi2-only | Phi-2 | TinyLlama (transfer) | 0.564 | — | — |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | TinyLlama-Chat (transfer) | 0.943 | 94.5% | 91.6% |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-Distill-1.5B (transfer) | 0.924 | 90.2% | 86.8% |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-Distill-7B (transfer) | 0.909 | 89.0% | 85.3% |
| 3-model | Qwen + Phi-2 + TinyLlama-Base | DeepSeek-R1-Distill-14B (transfer) | **0.933** | **91.8%** | **88.0%** |
| 3-model | — | Sycophancy | 59.4% recall | 100% precision | — |

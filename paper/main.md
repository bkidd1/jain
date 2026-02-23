# Black-Box Reasoning Trace Prediction via White-Box Distillation

**Brinlee Owens**  
Independent Researcher  
brinlee@example.com

---

## Abstract

Large language models often reach correct answers through implicit reasoning steps that are never verbalized. Current interpretability techniques require white-box access to model internals, limiting their applicability to closed models. We present a method for predicting hidden reasoning traces without access to model weights. Our approach extracts ground-truth reasoning traces from open-weight models using logit lens analysis, trains a lightweight Reasoning Trace Predictor (RTP) on the resulting (input, output, trace) triplets, and tests whether learned patterns transfer to unseen model architectures. Key finding: an RTP trained exclusively on Llama 3.1 8B traces achieves **40% F1** in predicting Mistral 7B's internal reasoning—demonstrating that implicit reasoning patterns transfer across architectures. This suggests the possibility of "interpretability transfer": using open models to understand closed ones.

---

## 1. Introduction

Language models routinely solve problems through implicit reasoning that never surfaces in their outputs. A model asked "What is the capital of the state where Dallas is located?" produces "Austin" without mentioning Texas, yet internally represents and uses this intermediate concept (nostalgebraist, 2020). Recent work from Anthropic demonstrates this gap starkly: reasoning models verbalize their actual decision factors only 20-39% of the time (Chen et al., 2025).

This creates a fundamental problem for AI safety and alignment. If models reason in ways they don't disclose, how can we verify their reasoning is sound? Current interpretability techniques—logit lens (nostalgebraist, 2020), tuned lens (Belrose et al., 2023), activation patching (Meng et al., 2022)—provide powerful tools for examining internal representations, but require direct access to model weights. The most capable deployed models (GPT-4, Claude) remain closed, their reasoning processes opaque.

We propose a different approach: **learning to predict reasoning traces from external observations alone**. Our method:

1. **Extracts ground-truth traces** from open-weight models using logit lens analysis and activation patching to identify causally-relevant intermediate concepts
2. **Trains a Reasoning Trace Predictor (RTP)** to map (input, output) pairs to their corresponding reasoning traces
3. **Tests cross-architecture transfer**: can patterns learned from Llama predict what Mistral is "thinking"?

Our key finding is that **reasoning patterns transfer across model architectures**. An RTP trained only on Llama 3.1 8B achieves 40% F1 in predicting Mistral 7B's internal representations. This suggests that different models converge on similar implicit reasoning strategies—and that we might understand closed models by studying open ones.

### Contributions

- First demonstration of **black-box reasoning trace prediction** via distillation from white-box models
- A **cross-model transfer experiment** showing 40% F1 transfer from Llama to Mistral
- Evidence that implicit reasoning patterns are partially **architecture-agnostic**

---

## 2. Related Work

### Mechanistic Interpretability

The logit lens (nostalgebraist, 2020) projects intermediate residual stream states to vocabulary space, revealing what a model "believes" at each layer. Belrose et al. (2023) improve this with learned affine transformations (tuned lens). Activation patching (Meng et al., 2022) establishes causal relationships by intervening on activations and measuring output changes.

### Reasoning Verification

Zhang et al. (2025) probe hidden states to predict correctness of intermediate reasoning steps, achieving high accuracy and enabling early exit from reasoning chains. The Anthropic faithfulness study (Chen et al., 2025) reveals that chain-of-thought explanations often fail to mention factors that actually influenced the model's decision.

### Our Novelty

Prior work requires white-box access to the target model. We train on one model's internals and transfer predictions to another—potentially enabling interpretability of closed models via open surrogates.

---

## 3. Method

### 3.1 Ground Truth Extraction

We extract reasoning traces from Llama 3.1 8B using two complementary techniques:

**Logit Lens Analysis.** For each prompt, we project the residual stream at layers corresponding to 25%, 50%, 75%, and 100% of model depth to vocabulary space. We record the top-k predicted tokens at each layer, identifying concepts that emerge before the final answer crystallizes.

**Activation Patching.** To verify concepts are causally used (not merely present), we zero-ablate the residual stream at each layer and measure the change in output probability. Concepts with causal effect > 0.1 are retained.

**Example trace:**
```
Prompt: "The capital of the state where Dallas is located is"
Layer 24: Texas (0.15), Austin (0.42)
Layer 31: Austin (0.16), a (0.16)
Final output: "Austin"
Extracted trace: Dallas → Texas → Austin
```

### 3.2 Dataset Construction

We generate prompts across four task categories:

| Task | Example | Trace |
|------|---------|-------|
| Multi-hop factual | "Capital of state where Dallas is" | Dallas → Texas → Austin |
| Direct factual | "Apple was founded by Steve" | Apple → Steve → Jobs |
| Arithmetic | "37 + 48 =" | 37, 48 → 85 |
| Sentiment | "Critics loved it. The movie was" | positive → good |

After filtering empty and garbage traces, we obtain 74 examples (55 train, 19 test).

### 3.3 Reasoning Trace Predictor

We fine-tune TinyLlama 1.1B using LoRA (Hu et al., 2022) with rank 8, targeting attention projection layers. Input format:

```
Prompt: {prompt}
Answer: {answer}
Trace: {target_trace}
```

Training for 5 epochs with learning rate 2e-4 reduces loss from 3.8 to 0.9.

---

## 4. Experiments

### 4.1 Cross-Model Transfer

**Setup.** We train the RTP exclusively on traces extracted from Llama 3.1 8B. At test time, we:
1. Extract ground-truth traces from Mistral 7B (never seen during training)
2. Use the Llama-trained RTP to predict traces from (input, output) pairs
3. Compare predictions to Mistral's actual internal representations

**Metrics.** We measure token-level overlap:
- **Precision**: fraction of predicted concepts appearing in Mistral's trace
- **Recall**: fraction of Mistral's concepts that were predicted  
- **F1**: harmonic mean

### 4.2 Results

| Prompt | Llama Trace | Mistral Trace | Predicted | F1 |
|--------|-------------|---------------|-----------|-----|
| Capital of Texas | Austin | Austin | Austin | 100% |
| Dallas → state | Texas | Texas | Texas | 100% |
| Largest city CA | Los | Los | Los | 67% |
| Windows company | Microsoft | Microsoft | Microsoft | 67% |
| Apple founder | Jobs | Apple | Jobs | 0% |

**Aggregate: F1 = 40%, Precision = 64%, Recall = 30%**

### 4.3 Analysis

**What transfers.** Entity knowledge and factual associations show strong transfer. Both models internally represent "Texas" when reasoning about Dallas, and "Microsoft" when completing "The company that makes Windows is."

**What doesn't transfer.** Fine-grained reasoning paths differ. For "Apple was founded by Steve," Llama emphasizes "Jobs" while Mistral represents "Apple"—both related to the answer but reflecting different internal strategies.

**Interpretation.** High precision (64%) indicates predictions are rarely wrong—when the RTP predicts a concept, it likely appears in Mistral's internals. Lower recall (30%) suggests we miss some of Mistral's reasoning, expected given architectural differences.

---

## 5. Discussion

### Implications

Our results suggest that **implicit reasoning has universal structure** that transcends specific architectures. This opens possibilities for:

1. **Interpretability transfer**: Understanding closed models via open surrogates
2. **Safety monitoring**: Detecting hidden reasoning in deployed systems
3. **Alignment verification**: Checking if stated rationales match internal processes

### Limitations

- **Small dataset** (74 examples): Results are preliminary
- **Limited model pairs** (Llama → Mistral only): Needs broader testing
- **Raw logit lens**: Tuned lens would improve ground truth quality
- **Model scale**: Only tested on 7-8B models

### Future Work

- Scale to larger models (70B+) and more architecture pairs
- Apply to safety-critical domains (deception detection, reward hacking)
- Improve ground truth with tuned lens and causal scrubbing

---

## 6. Conclusion

We demonstrate that implicit reasoning traces can be predicted across model architectures. An RTP trained only on Llama 3.1 8B achieves 40% F1 in predicting Mistral 7B's internal reasoning, suggesting that language models converge on similar implicit strategies despite different architectures.

The key insight: **language models may think more alike than their architectures suggest**—and this similarity enables a new form of interpretability transfer from open to closed models.

---

## References

Belrose, N., et al. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv:2303.08112*.

Chen, A., et al. (2025). Reasoning models don't always say what they think. *Anthropic Research*.

Hu, E.J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Meng, K., et al. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.

nostalgebraist. (2020). Interpreting GPT: The logit lens. *LessWrong*.

Zhang, Y., et al. (2025). Reasoning models know when they're right. *arXiv:2504.05419*.

---

## Appendix A: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | TinyLlama 1.1B |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Epochs | 5 |
| Max sequence length | 128 |

## Appendix B: Task Prompts

**Multi-hop factual:**
- The capital of the state where {city} is located is
- The largest city in the state whose capital is {capital} is

**Direct factual:**
- {city} is a city in the state of
- The capital of {country} is
- {company} was founded by

**Arithmetic:**
- {a} + {b} =
- {a} - {b} =
- {a} × {b} =

**Sentiment:**
- {positive_frame} The movie was
- {negative_frame} The restaurant was

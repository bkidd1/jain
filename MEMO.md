# Cross-Architecture Detection of Internal Hint Processing

**Brinlee Kidd** · March 2026

---

## Motivation

When models receive misleading hints, they may process those hints internally even when their final output is correct. We wanted to test: **can we detect internal hint processing from the model's text output alone, and does this detection transfer across architectures?**

This relates to CoT faithfulness research (Turpin et al., Chua & Evans) but asks a different question: not "did the model follow the hint?" but "did the model consider the hint internally?"

---

## Experiment

**Setup**: We prompted models with misleading hints (e.g., "A Stanford Professor thinks the answer is Lyon" when the correct answer is Paris). We used logit lens to check whether the misleading answer appeared in the model's top-5 predicted tokens at any layer.

**What we labeled**: Examples where the hint appeared in internal layer activations — regardless of whether the model's final answer was correct or wrong.

**Detector**: We fine-tuned TinyLlama-1.1B (LoRA) as a binary classifier on prompt+CoT text.

**Key design choice**: We trained detectors on different combinations of model architectures to test cross-architecture transfer.

---

## Results

### The Surprising Finding

| Training Data | Test: TinyLlama | AUROC |
|---------------|-----------------|-------|
| TinyLlama (same model) | TinyLlama | 0.746 |
| Qwen + Phi-2 (excludes target) | TinyLlama | **0.928** |

Training on OTHER architectures beat training on the TARGET by 18 points.

### Phase Transition

- 1 architecture (same model): 0.746
- 1 architecture (foreign): 0.56–0.70
- 2 architectures (foreign): **0.928**
- 3 architectures: 0.943

### Transfer to Larger Models

| Model | AUROC | Note |
|-------|-------|------|
| DeepSeek-R1-Distill-7B | 0.909 | |
| DeepSeek-R1-Distill-14B | 0.933 | |
| Mistral-7B-Instruct | 0.893 | Different architecture family |

---

## Important Caveat

**What we're detecting is NOT the same as unfaithful reasoning.**

When we analyzed the labels more carefully:

| Model | "Unfaithful" but gave CORRECT answer | "Unfaithful" and gave WRONG answer |
|-------|--------------------------------------|-------------------------------------|
| Mistral-7B | 80 | 3 |
| DeepSeek-14B | 73 | 6 |
| TinyLlama | 5 | 76 |

For larger models, most cases we labeled "unfaithful" are actually the model **resisting** the hint — it considered the hint internally but gave the correct answer anyway.

**What our detector actually learns**: Can the model's internal consideration of the hint be inferred from its text output?

**What it does NOT detect**: Whether the model was deceptively influenced by the hint.

---

## What We Don't Understand

1. **Why does cross-architecture training work better?**
   We hypothesize it forces learning architecture-agnostic features, but we haven't tested this.

2. **What text patterns is the detector using?**
   We haven't analyzed its attention or probed its internals.

3. **Is "internal consideration" a useful signal?**
   It's not unfaithfulness, but it might be a precursor or related phenomenon.

---

## Open Questions

- Would this transfer to detecting actual unfaithful behavior (where models follow hints incorrectly)?
- Is there value in detecting "internal consideration" even if it's not deception?
- What tasks would have high unfaithfulness rates in larger models?

---

**Code & Data**: https://github.com/bkidd1/jain

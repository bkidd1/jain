# Detecting Hidden Hint Usage in Chain-of-Thought Reasoning

**Brinlee Kidd** · March 2026

---

## The Problem

When models receive hints or privileged information in their prompts, they may use that information without acknowledging it in their reasoning. The model's Chain-of-Thought looks like independent reasoning, but it was actually influenced by the hint.

This is a form of deceptive or unfaithful reasoning — the CoT doesn't accurately reflect how the model arrived at its answer.

**Our question:** Can we detect when a model secretly used a hint, just by looking at its text output?

---

## What We Built

A detector that identifies hidden hint usage from CoT text alone:

1. **Training**: We gave models misleading hints and used logit lens to check if the hint appeared in internal layer activations. This gave us ground-truth labels for "did the model process this hint?"

2. **The detector**: Fine-tuned TinyLlama as a binary classifier on prompt + CoT text. At inference, it predicts whether the model processed a hint — without needing access to internals.

3. **Key capability**: The detector learns textual signatures of hint-influenced reasoning that transfer across model architectures.

---

## Results

### Cross-Architecture Transfer Works

| Training Data | Test Model | AUROC |
|---------------|------------|-------|
| TinyLlama only | TinyLlama | 0.746 |
| Qwen + Phi-2 (no TinyLlama) | TinyLlama | **0.928** |

Training on OTHER architectures beats training on the target by 18 points.

### Transfers to Larger Models

| Model | AUROC |
|-------|-------|
| DeepSeek-R1-Distill-7B | 0.909 |
| DeepSeek-R1-Distill-14B | 0.933 |
| Mistral-7B-Instruct | 0.893 |

The detector (trained on 1-3B models) works on 7-14B models from different architecture families.

### Phase Transition at 2 Architectures

- 1 same-model architecture: 0.746
- 1 foreign architecture: 0.56–0.70
- 2 foreign architectures: **0.928**
- 3 architectures: 0.943

You need diversity — a single foreign architecture doesn't transfer well, but two together do.

---

## Why This Matters

### The Practical Capability

You can detect when a model's reasoning was influenced by a hint it didn't acknowledge, using only the text output. This works on models you didn't train on.

**Implications:**
- Train detector on small open models where you CAN inspect internals
- Deploy on large/closed models where you CAN'T
- Detect hidden influence from prompt injections, system prompts, or other privileged information

### Relation to Prior Work

Chua & Evans (2025) found that DeepSeek R1 verbalizes hints only 59% of the time — meaning 41% of hint usage is hidden. This is part of broader work by Owain Evans and collaborators on CoT faithfulness, including the TruthfulQA benchmark and studies of how reasoning models handle privileged information.

Our detector provides a potential method to catch some of that hidden usage from text alone — addressing the gap between "model used hint" and "model acknowledged using hint."

---

## Limitations

1. **We detect hint processing, not outcomes**: The detector catches when a model considered a hint, whether it ultimately followed or resisted it. For safety applications, you may also need to check if the answer was correct.

2. **Artificial paradigm**: Hint injection is a controlled setup. Real-world hidden influence (sycophancy, encoded reasoning, prompt injection) may have different signatures.

3. **No mechanistic understanding**: We don't know what text patterns the detector uses.

---

## What We Don't Know

- Does this transfer to detecting other forms of hidden influence (not just explicit hints)?
- What specific text features does the detector learn?
- Why does cross-architecture training work better?

---

## Bottom Line

We can detect when a model secretly processed a hint, just from its text output, and this detection transfers across architectures. The "exclude target" finding (+18 points) suggests diverse training produces more generalizable detectors.

---

**Code & Data**: https://github.com/bkidd1/jain

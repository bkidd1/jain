# Detecting Hidden Hint Usage in Chain-of-Thought Reasoning

**Brinlee Kidd** · March 2026

---

## The Problem

When models receive hints or privileged information in their prompts, they may use that information without acknowledging it in their reasoning. The model's Chain-of-Thought looks like independent reasoning, but it was actually influenced by the hint.

This is a form of deceptive or unfaithful reasoning; the CoT doesn't accurately reflect how the model arrived at its answer.

**Our question:** Can we detect when a model secretly used a hint, just by looking at its text output?

---

## What We Built

A detector that identifies hidden hint usage from CoT text alone:

1. **Training**: We gave models misleading hints and used logit lens to check if the hint appeared in internal layer activations. This gave us ground-truth labels for "did the model process this hint?"

2. **The detector**: Fine-tuned TinyLlama as a binary classifier on prompt + CoT text. At inference, it predicts whether the model processed a hint without needing access to internals.

3. **Key capability**: The detector learns textual signatures of hint-influenced reasoning that transfer across model architectures.

---

## Results

### Cross-Architecture Transfer Works

| Training Data | Test Model | AUROC |
|---------------|------------|-------|
| TinyLlama only | TinyLlama | 0.746 |
| Qwen + Phi-2 (no TinyLlama) | TinyLlama | **0.928** |

Training on OTHER architectures beats training on the target by 18 points!

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

You can detect when a model's reasoning was influenced by a hint it didn't acknowledge, using only the text output. This works on models you didn't train on.

**Implications:**
- Train detector on small open models where you CAN inspect internals
- Deploy on large/closed models where you CAN'T
- Detect hidden influence from prompt injections, system prompts, or other privileged information

### Relation to Prior Work

Chua & Evans (2025) found that DeepSeek R1 verbalizes hints only 59% of the time — meaning 41% of hint usage is hidden. This is part of broader work by Owain Evans and collaborators on CoT faithfulness, including the TruthfulQA benchmark and studies of how reasoning models handle privileged information.

Our detector provides a potential method to catch some of that hidden usage from text alone; addressing the gap between "model used hint" and "model acknowledged using hint."

---

## Limitations & Open Questions

### Known Issues (Being Addressed)

**Prompt-in-input confound**: Our classifier sees `prompt + response`, and in the misleading-hint condition, the hint appears in the prompt itself (e.g., "I recall that Los Angeles might be the answer..."). This means the classifier might learn to detect the **prompt template** rather than **subtle cues in the chain-of-thought**.

We're running ablations to isolate this (see `experiments/04_response_only_ablation`):
- **Response-only**: Train on just the CoT, no prompt
- **Redacted-prompt**: Remove hint text from prompt, keep question

If response-only AUROC stays high → signal is genuinely in the reasoning.
If it tanks → we were detecting the prompt, not the CoT.

### Other Limitations

1. **We detect hint processing, not outcomes**: The detector catches when a model considered a hint, whether it ultimately followed or resisted it. For safety applications, you may also need to check if the answer was correct.

2. **Label fidelity**: Our ground-truth uses logit-lens top-k at the last input position. This is a coarse proxy for "model used the hint" — not the same as causal influence or verbalization. More rigorous labeling (e.g., causal patching) would strengthen claims.

3. **Artificial paradigm**: Hint injection is a controlled setup. Real-world hidden influence (sycophancy, encoded reasoning, prompt injection) may have different signatures.

4. **No mechanistic understanding**: We don't know what text patterns the detector uses.

---

## What We've Tested (And What Failed)

**Post-hoc rationalization transfer**: We tested whether the hint detector transfers to detecting sycophantic reasoning (where models give different answers based on stated user beliefs). Result: **AUROC 0.58** — essentially random. The detector does NOT generalize to other forms of unfaithful reasoning. See `experiments/03_posthoc_transfer`.

This suggests hint-influenced reasoning and post-hoc rationalization have different textual signatures, or that our current detector is too specific to the hint paradigm.

## What We Don't Know

- **Is the signal in the CoT or the prompt?** (Ablations in progress)
- What specific text features does the detector learn?
- Why does cross-architecture training work better?
- Would a detector trained on diverse unfaithfulness types transfer more broadly?

---

## Bottom Line

We can detect when a model secretly processed a hint, just from its text output, and this detection transfers across architectures. The "exclude target" finding (+18 points) suggests diverse training produces more generalizable detectors.

---

**Code & Data**: https://github.com/bkidd1/jain

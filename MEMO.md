# Detecting Hidden Hint Usage in Chain-of-Thought Reasoning

**Brinlee Kidd** · March 2026

---

## Summary

We attempted to build a detector that identifies hidden hint usage from Chain-of-Thought text alone. Initial results looked promising (high AUROC, cross-architecture transfer), but **ablation testing revealed the signal was in the prompt, not the reasoning.**

This is a null result for CoT-based detection, but an important methodological lesson.

---

## The Original Idea

When models receive hints in their prompts, they may use that information without acknowledging it. The CoT looks like independent reasoning, but was actually influenced by the hint.

**Our question:** Can we detect this from the text output alone?

---

## What We Tried

1. **Training**: Gave models misleading hints, used logit lens to label whether hints appeared in internal activations
2. **Detector**: Fine-tuned TinyLlama as binary classifier on prompt + CoT text
3. **Initial results**: High AUROC (0.93+), transfer across architectures

This looked promising — until we tested what the detector was actually learning.

---

## The Ablation That Changed Everything

We trained and evaluated detectors with different input formats:

| Training Format | Eval Format | AUROC |
|-----------------|-------------|-------|
| full (prompt + response) | full | 1.000 |
| full | response-only | 0.653 |
| **response-only** | **response-only** | **0.670** |
| redacted (no hint text) | redacted | 0.998 |

### What this shows

1. **Response-only training: 0.67 AUROC** — barely above chance. If hint-influenced reasoning had detectable text patterns, this would be much higher.

2. **Full model collapses without prompt**: 1.0 → 0.65 when evaluated on response-only. It learned prompt features.

3. **Even redacted prompts leak**: Removing "I recall that X might be the answer" still gives 0.998. The template structure itself signals the condition.

**Conclusion:** The detector learned to recognize prompt templates, not reasoning patterns.

---

## What We Learned

### The cross-architecture "transfer" was an artifact

The impressive transfer results (training on Qwen+Phi-2, testing on TinyLlama/DeepSeek/Mistral) were real — but they were detecting **prompt format consistency**, not generalizable reasoning signatures.

All models saw the same prompt templates. The detector learned "this prompt has a hint preamble" which transfers trivially across architectures.

### The "+18 AUROC" finding was confounded

Same-model training (0.746) vs cross-architecture (0.928) likely reflects overfitting to model-specific response quirks vs learning the more robust prompt signal. Not a deep insight about generalization.

### Response-only is the real test

Any claim about detecting reasoning patterns requires response-only evaluation. Without it, you might just be detecting your experimental setup.

---

## The Honest Results

| Claim | Status |
|-------|--------|
| Detect hint usage from CoT text alone | ❌ Not supported (0.67 AUROC) |
| Cross-architecture transfer | ⚠️ Real but confounded (prompt signal) |
| Train small → deploy large | ⚠️ Works, but for prompt detection |

---

## What Would Actually Work?

To genuinely detect hidden influence from reasoning text, you'd need:

1. **Prompts that don't leak condition** — same template for hint/no-hint, with hints embedded more subtly
2. **Response-only training and evaluation** — the only honest test
3. **Diverse hint injection methods** — not just "I recall that X..."
4. **Mechanistic validation** — what text features does the detector use?

---

## Relation to Prior Work

Chua & Evans (2025) found R1 verbalizes hints only 59% of the time. Our work tried to catch the other 41% from text alone.

**We failed** — but the failure is informative. It suggests either:
- Hidden hint usage doesn't leave detectable text traces, or
- Our paradigm (explicit hint preambles) was too easy to game

A proper test would need more naturalistic hint injection.

---

## Bottom Line

We set out to detect hidden hint usage from CoT text. We ended up detecting prompt templates. The lesson: **ablate your inputs before trusting your metrics.**

---

**Code & Data**: https://github.com/bkidd1/jain

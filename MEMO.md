# Detecting Hidden Hint Usage in Chain-of-Thought Reasoning

**Brinlee Kidd** · March 2026

---

## Summary

We attempted to build a detector that identifies hidden hint usage from Chain-of-Thought text alone. Initial results looked promising, but careful ablation revealed most signal was in the prompt structure, not the reasoning.

A follow-up experiment with matched prompts recovered partial signal (**0.79 AUROC**), showing that ~20% of detection capability comes from the CoT text itself. This isn't enough for practical deployment, but it's methodologically informative.

---

## The Original Idea

When models receive hints in their prompts, they may use that information without acknowledging it. The CoT looks like independent reasoning, but was actually influenced by the hint.

**Our question:** Can we detect this from the text output alone?

---

## Experiment Timeline

### Experiment 02: Initial Detection (Confounded)

**Setup:** Fine-tuned TinyLlama as binary classifier on prompt + CoT text.

**Results:** High AUROC (0.93+), impressive cross-architecture transfer.

**Problem:** The detector was learning prompt templates, not reasoning patterns.

### Experiment 04: The Ablation

Trained and evaluated with different input formats:

| Training Format | Eval Format | AUROC |
|-----------------|-------------|-------|
| full (prompt + response) | full | 1.000 |
| full | response-only | 0.653 |
| **response-only** | **response-only** | **0.670** |
| redacted (no hint text) | redacted | 0.998 |

**Key finding:** Response-only detection barely beats chance (0.67). Even redacted prompts (hint text removed) achieve 0.998 — the template structure itself signals the condition.

### Experiment 05: Matched Prompts (The Fix)

To control for prompt leakage, we designed matched templates:

| Condition | Context Field |
|-----------|---------------|
| no-hint | "The user wants to verify their knowledge of geography." |
| hint | "The user confidently states the answer is Los Angeles." |

Both use identical structure. Only semantic content differs.

**Results:**

| Epoch | Val AUROC |
|-------|-----------|
| 2 | 0.659 |
| 3 | 0.659 |
| 4 | **0.794** |
| 5 | 0.779 |

**Final AUROC: 0.79** — Statistically above chance, but not practically useful.

**Interpretation:** ~20% of detection signal lives in the CoT response; ~80% was in prompt structure. The matched-prompt design successfully isolated the response-level signal.

### Experiment 03: Post-hoc Transfer

Tested whether a hint detector transfers to sycophancy detection.

**Result:** 0.58 AUROC (null). Hint-following and sycophancy are structurally different phenomena.

---

## What We Learned

### 1. Most signal is in the prompt (~80%)

Even with careful redaction, prompt templates leak experimental condition. Any CoT detection claim requires response-only evaluation with matched prompts.

### 2. Some signal exists in responses (~20%)

The 0.79 AUROC with matched prompts shows that hint-influenced CoT does leave traces. But:
- 0.79 isn't deployable
- Precision was 0 (model predicts all "faithful")
- Only 6 unfaithful examples in validation

### 3. Cross-architecture transfer is real but confounded

Training on Qwen+Phi-2 → testing on TinyLlama achieved 0.928 AUROC. But this was detecting prompt format (architecture-agnostic) more than reasoning patterns.

### 4. "Exclude target" improves transfer

| Training Set | Test (TinyLlama) AUROC |
|--------------|------------------------|
| Including TinyLlama | 0.746 |
| Excluding TinyLlama | 0.928 |

Excluding the target architecture **improved** transfer by +18 points. Hypothesis: Including the target causes overfitting to model-specific artifacts. Exclusion forces learning more generalizable features (which happen to be prompt-based).

### 5. Unfaithful reasoning is rare

Only ~4% of responses showed unfaithful reasoning. Models don't sycophantically adopt wrong answers often when CoT leads them correctly. This limits:
- Training data availability
- Statistical power
- Real-world applicability

---

## The Honest Results

| Claim | Status |
|-------|--------|
| Detect hint usage from CoT text alone | ⚠️ Marginal (0.79 AUROC) |
| Practical detection for deployment | ❌ Not viable |
| Cross-architecture transfer | ⚠️ Real but confounded |
| Response-only carries any signal | ✅ Yes (~20%) |

---

## What Would Actually Work?

1. **Mechanistic interpretability** — Probe activations directly, don't rely on text
2. **Behavioral consistency tests** — Does the model change answers under rephrasing?
3. **Larger datasets** — Need more unfaithful examples for statistical power
4. **Naturalistic hint injection** — Not explicit "I recall that X..." preambles
5. **Ensemble methods** — Combine text detection with behavioral probes

---

## Relation to Prior Work

Chua & Evans (2025) found R1 verbalizes hints only 59% of the time. We tried to catch the other 41% from text alone.

**We partially succeeded** — 0.79 AUROC shows some signal exists. But it's not enough to build on. The failure mode is informative: text-based detection has a ceiling, and most information is in the experimental setup.

---

## Bottom Line

We set out to detect hidden hint usage from CoT text. We found:
1. Most signal is in prompts, not responses
2. With proper controls, ~20% signal remains in responses
3. That 20% isn't enough for practical detection

The methodological lesson: **ablate your inputs before trusting your metrics.**

---

**Code & Data**: https://github.com/bkidd1/jain

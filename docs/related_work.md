# Related Work Analysis

## Summary

Our logit lens experiments align with an active research area, but the **black-box reasoning trace prediction** angle is relatively unexplored. Here's the landscape:

---

## Directly Related Papers

### 1. Tuned Lens (Belrose et al., 2023)
**Paper:** [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)

- **What they did:** Improved logit lens by training small affine transformations at each layer (instead of applying unembedding directly)
- **Key finding:** Much more interpretable predictions, especially in early layers where raw logit lens fails
- **Relevance to us:** We should implement tuned lens for better ground truth extraction
- **Gap we fill:** They don't train an external predictor; they require white-box access

### 2. Reasoning Models Know When They're Right (Zhang et al., April 2025)
**Paper:** [arXiv:2504.05419](https://arxiv.org/abs/2504.05419)

- **What they did:** Probed hidden states to predict correctness of intermediate reasoning steps
- **Key finding:** Linear probes on hidden states predict answer correctness with high accuracy; used this to exit reasoning early (24% fewer tokens)
- **Relevance to us:** Very similar probing approach! But they predict correctness, we predict the *reasoning trace itself*
- **Gap we fill:** They still need white-box access; we want to transfer this to black-box settings

### 3. Anthropic's CoT Faithfulness Study (Chen et al., May 2025)
**Paper:** [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410)

- **What they did:** Tested if reasoning models mention hints they used in their CoT
- **Key finding:** Models verbalize actual reasoning factors only **20-39% of the time**
- **Relevance to us:** This is the PROBLEM we're trying to solve — detecting implicit reasoning that isn't verbalized
- **Gap we fill:** They identify the problem; we're building a tool to detect these hidden influences

### 4. Do LLMs Really Think Step-by-step In Implicit Reasoning? (Nov 2024)
**Paper:** [arXiv:2411.15862](https://arxiv.org/abs/2411.15862)

- **What they did:** Probed hidden states during implicit CoT to see if intermediate steps are represented
- **Key finding:** Evidence that models do represent intermediate reasoning steps internally
- **Relevance to us:** Confirms our hypothesis that reasoning traces are extractable

### 5. Efficient Test-Time Scaling by Probing Internal States (Jan 2026)
**Paper:** [arXiv:2511.06209](https://arxiv.org/abs/2511.06209)

- **What they did:** Lightweight transformer probe on frozen LLM internal states for step verification
- **Key finding:** Much cheaper than full process reward models while still effective
- **Relevance to us:** Similar lightweight probe architecture; but they verify steps, we predict them

---

## What's Novel About Our Approach

| Existing Work | Our Approach |
|---------------|--------------|
| Requires white-box access to model internals | Train on open models, **transfer to black-box** |
| Probes for correctness/verification | Probes for **reasoning trace reconstruction** |
| Applied to single model family | **Cross-model transfer** experiments |
| Focused on math/logic benchmarks | Include **sentiment influence detection** (faithfulness) |

### The Key Novel Contribution
**Training a reasoning trace predictor on white-box ground truth, then applying it to black-box models.**

No one has done this yet. Existing work either:
- Requires white-box access (logit lens, tuned lens, probing)
- Tests faithfulness behaviorally without predicting internal states (Anthropic study)

---

## What's Publishable at Our Scale (8B models, single GPU)

### Good News
Most interpretability workshops explicitly welcome smaller-scale work:
- **ICML 2024 MI Workshop** - "We welcome both empirical and theoretical contributions"
- **NeurIPS 2025 MI Workshop** - Focus on methodology over scale
- **COLM workshops** - Language model focused, accepts focused empirical studies

### Publishable Findings at Our Scale

1. **Methodological contribution:** "Can black-box reasoning prediction transfer from open to closed models?"
   - Even a negative result is interesting if well-characterized

2. **Empirical finding:** Our observation that arithmetic is "known" internally (Layer 24: 63 at 0.98) but not output (final: space at 0.87)
   - This is a clean, reproducible demo of the faithfulness problem

3. **Cross-model transfer:** Does a trace predictor trained on Llama generalize to Mistral?
   - Novel experiment, clear hypothesis, tractable at our scale

4. **Adversarial reasoning test:** Our "Austin → Texas → Houston" example
   - Shows models reason vs. use heuristics — good demo for interpretability

---

## Recommended Next Steps

### Immediate (This Week)
1. **Implement Tuned Lens** - Better ground truth than raw logit lens
2. **Generate structured dataset** - Run our task generators, extract traces
3. **Read the Zhang et al. probing paper closely** - Their methodology is closest to ours

### Short Term (Weeks 2-4)
4. **Train initial RTP** - Fine-tune small LM to predict traces from (input, output)
5. **Cross-model experiment** - Train on Llama, test on Mistral-7B
6. **Faithfulness detection** - Can we detect "hint usage" that CoT doesn't mention?

### Publication Target
- **ICML 2026 MI Workshop** or **NeurIPS 2026 Safe AI Workshop**
- Draft by end of Phase 2 (~8 weeks)
- Clear, focused contribution: "Black-box reasoning trace prediction via white-box distillation"

---

## Key Papers to Read in Detail

1. ⭐ [Tuned Lens](https://arxiv.org/abs/2303.08112) - Nora Belrose (methodology)
2. ⭐ [Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419) - closest related work
3. [Anthropic Faithfulness Study](https://arxiv.org/abs/2505.05410) - the problem we're solving
4. [Do LLMs Think Step-by-step](https://arxiv.org/abs/2411.15862) - confirms our hypothesis

---

*Last updated: 2026-02-23*

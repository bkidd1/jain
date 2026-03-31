# Experiment 04: Response-Only Ablation

**Status:** Complete  
**Date:** March 2026

## Motivation

Experiments 02 trained a detector on `prompt + response`. But in the misleading-hint condition, the hint appears **in the prompt itself** (e.g., "I recall that Los Angeles might be the answer...").

This means the classifier might learn to detect the **prompt template** rather than **subtle cues in the chain-of-thought**.

## The Question

**How much signal is in the CoT vs the prompt?**

## Results

| Training Format | Eval Format | AUROC |
|-----------------|-------------|-------|
| full | full | **1.000** |
| full | response-only | 0.653 |
| full | redacted | 0.936 |
| response-only | response-only | **0.670** |
| redacted | redacted | 0.998 |
| redacted | response-only | 0.598 |

## Interpretation

### The signal is in the prompt, not the CoT

1. **Response-only training achieves only 0.67 AUROC** — barely above chance. If there were detectable patterns in hint-influenced reasoning, this should be much higher.

2. **Full model drops from 1.0 → 0.65** when evaluated on response-only. The model learned prompt features, not CoT features.

3. **Even redacted prompts leak the condition** — removing the hint text ("I recall that X...") still gives 0.998 AUROC. The template structure itself signals the experimental condition.

### What this means

The original claim — "detect hidden hint usage from text alone" — is **not supported**. The detector in experiment 02 was learning to recognize:
- The presence of hint preambles in prompts
- Structural differences in prompt templates

NOT subtle patterns in the chain-of-thought reasoning.

The ~0.67 AUROC from response-only likely reflects shallow correlations (e.g., wrong answers correlate with unfaithful labels) rather than detection of "hidden" influence.

## Conclusion

**Null result for CoT-based detection.** This is a methodologically important finding: prompt-in-input confounds are real and response-only baselines are essential for any claim about detecting reasoning patterns.

## Files

```
data/models/
├── full/           # Trained on prompt + response
├── response_only/  # Trained on response only
└── redacted/       # Trained on question + response (no hint text)

results/
└── ablation_results.json  # Full metrics
```

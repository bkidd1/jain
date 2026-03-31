# JAIN: In Vino Veritas for LLMs

> "Is unfaithful Chain-of-Thought reasoning structurally more fragile than faithful reasoning?"

## The Idea

Deception is computationally expensive. In humans, maintaining a lie requires tracking the truth AND the lie simultaneously. In LLMs, unfaithful reasoning might be a "house of cards" — internally inconsistent, more dependent on specific pathways.

**Detection strategy:** Don't map the full circuit. Just stress-test it.

## Approach

Adapt human deception detection methods to LLMs:

| Human Method | LLM Equivalent | Status |
|--------------|----------------|--------|
| Polygraph (stress signals) | Activation anomalies during inference | Existing work |
| Truth serum (reduced inhibition) | Activation steering toward truthfulness | Rep. Eng. work |
| **Intoxication (impaired control)** | **Noise injection into activations** | **Our focus** |
| Cognitive load (dual task) | Unexplored | Future work |

## Core Hypothesis

Unfaithful CoT (e.g., secretly using a hint while claiming to reason independently) is **more fragile under perturbation** than faithful CoT.

- Truthful responses are grounded in robust knowledge pathways
- Confabulated responses are "built backwards" and break under stress

## Experimental Plan

1. **Noise Injection** — Apply MLP noise during inference; measure variance
2. **Temperature Gradient** — Sweep temp 0.0→1.0; unfaithful responses may show different sensitivity
3. **Consistency Under Rephrasing** — Ask same question multiple ways; measure answer variance
4. **Layer Dropout** — Drop layers during inference; which responses survive?
5. **Activation Steering** — Find "using a hint" direction; measure distance

## Why This Angle?

- **Cheap:** No transcoder training ($0 vs $10K-100K)
- **Novel:** Noise injection for unfaithfulness (not just hallucination) is unexplored
- **Tractable:** "Does noise injection distinguish hint-influenced CoT?" is one experiment
- **Builds on:** Feb 2026 noise injection paper provides methodology

## Project Structure

```
jain/
├── THESIS.md                 # Full thesis document
├── README.md                 # This file
├── experiments/              # New experiments (v2)
├── archive/                  # Previous work (v1)
│   └── v1-text-detection/    # Text-based detection attempt
│       ├── MEMO.md           # Full writeup of v1
│       └── experiments/      # Original experiments 01-05
├── src/                      # Shared utilities
└── docs/                     # Documentation
```

## Previous Work (v1)

We previously attempted text-only detection of hint usage. Key findings:

- Response-only detection: **0.79 AUROC** (marginal, not practical)
- ~80% of signal was in prompts, not CoT
- Cross-architecture transfer findings were artifacts

Full writeup: [`archive/v1-text-detection/MEMO.md`](archive/v1-text-detection/MEMO.md)

## Key References

- Feb 2026 noise injection paper (hallucination detection)
- Representation Engineering (truth direction steering)
- Neel Nanda's consistency examples

## Authors

- Brinlee Kidd (brinlee0kidd@gmail.com)
- Demosthenes (AI collaborator)

## License

MIT

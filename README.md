# JAIN: Attempting to Detect Hidden Hint Usage from CoT

An experiment in detecting when language models secretly process hints — and a lesson in why ablations matter.

## TL;DR

We tried to detect hidden hint usage from Chain-of-Thought text alone. Initial results looked great (0.93+ AUROC, cross-architecture transfer). Then we ran ablations.

**The signal was in the prompt, not the reasoning.**

| Training Format | Eval Format | AUROC |
|-----------------|-------------|-------|
| full (prompt + response) | full | 1.000 |
| **response-only** | **response-only** | **0.670** |

Response-only detection barely beats chance. The detector learned to recognize prompt templates, not reasoning patterns.

## The Lesson

Any claim about detecting patterns in model reasoning requires **response-only evaluation**. Without it, you might just be detecting your experimental setup.

## Full Writeup

See **[MEMO.md](MEMO.md)** for the full story: what we tried, why it failed, and what would actually work.

## Project Structure

```
jain/
├── MEMO.md                           # Full writeup
├── experiments/
│   ├── 01_reconstruction/            # Initial RTP experiment (historical)
│   ├── 02_divergence_detection/      # Original hint detection (confounded)
│   ├── 03_posthoc_transfer/          # Post-hoc rationalization test (null)
│   └── 04_response_only_ablation/    # The ablation that revealed the confound
└── src/                              # Shared utilities
```

## Key Results

### Experiment 02: Original Detection (Confounded)
- AUROC 0.93+ on full input — but this included the prompt
- Cross-architecture "transfer" was real but trivial (prompt templates are architecture-agnostic)

### Experiment 04: The Ablation
- Response-only: **0.67 AUROC** — the actual signal in the CoT
- The prompt (even with hint text redacted) carries nearly all the signal

### Experiment 03: Post-hoc Transfer
- Tested if hint detector transfers to sycophancy detection
- Result: 0.58 AUROC (null)

## What Would Work?

1. Prompt templates that don't leak experimental condition
2. Response-only training and evaluation
3. More naturalistic hint injection (not "I recall that X...")
4. Mechanistic analysis of what features matter

## Authors

- Brinlee Kidd (brinlee0kidd@gmail.com)
- Demosthenes (AI collaborator)

## License

MIT

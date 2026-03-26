# JAIN: Detecting Hidden Hint Usage in Chain-of-Thought

An experiment in detecting when language models secretly process hints — and lessons in experimental design.

## TL;DR

We tried to detect hidden hint usage from Chain-of-Thought text alone. Initial results looked great (0.93+ AUROC). Ablations revealed the signal was mostly in the prompt, not the reasoning.

**Key findings:**

| Experiment | Method | AUROC | Interpretation |
|------------|--------|-------|----------------|
| 02 | Full input (prompt + response) | 0.93+ | High but confounded |
| 04 | Response-only (mismatched prompts) | 0.67 | Near-random |
| **05** | **Response-only (matched prompts)** | **0.79** | **Weak signal exists** |

## The Story

1. **We thought we could detect hints from CoT alone** — Initial 0.93+ AUROC looked promising
2. **Ablation revealed prompt leakage** — Response-only dropped to 0.67
3. **Matched-prompt design recovered some signal** — 0.79 AUROC shows ~20% of signal is in responses, ~80% in prompts

## Key Insights

### Where the signal lives
- **~80% in prompt context** (even with hint text redacted)
- **~20% in CoT response** (with proper controls)

### Cross-architecture transfer was an artifact
In exp02 (confounded), "excluding target" seemed to help (+18 points). But with matched prompts (exp05):

| Training Set | TinyLlama AUROC |
|--------------|-----------------|
| Including TinyLlama | **0.79** |
| Excluding TinyLlama | 0.59 |

**The effect reversed.** The original finding was the classifier learning architecture-agnostic prompt confounds better when forced to ignore architecture-specific patterns.

### Unfaithful reasoning is rare
Only ~4% of responses showed unfaithful reasoning (hint appeared in internal activations without verbalization). This limits practical applicability.

## Full Writeup

See **[MEMO.md](MEMO.md)** for the complete narrative.

## Project Structure

```
jain/
├── MEMO.md                           # Full writeup
├── experiments/
│   ├── 01_reconstruction/            # Initial RTP experiment (historical)
│   ├── 02_divergence_detection/      # Original hint detection (confounded)
│   ├── 03_posthoc_transfer/          # Post-hoc rationalization test (null)
│   ├── 04_response_only_ablation/    # Ablation revealing prompt confound
│   └── 05_matched_prompts/           # Controlled response-only test
└── src/                              # Shared utilities
```

## Results Summary

### Experiment 02: Original Detection
- AUROC 0.93+ on full input
- Cross-architecture transfer worked
- **Confounded by prompt templates**

### Experiment 04: Response-Only Ablation
- Response-only: **0.67 AUROC** (near random)
- Revealed prompt leakage issue
- **Methodological wake-up call**

### Experiment 05: Matched Prompts
- Response-only with controlled prompts: **0.79 AUROC**
- Shows some signal exists in CoT text
- **But not enough for practical detection**

### Experiment 03: Post-hoc Transfer
- Tested hint→sycophancy transfer: **0.58 AUROC**
- Different phenomena, no transfer
- **Null result (expected)**

## Conclusions

1. **Response-only detection is marginally possible** (0.79) but not practical
2. **Most signal is in prompts** — future work must control for this
3. **Cross-arch "exclude target" finding was an artifact** — reverses with proper controls
4. **Unfaithful reasoning is rare** (~4%), limiting real-world applicability

## What Would Actually Work?

1. Mechanistic interpretability (probe activations directly)
2. Behavioral tests (consistency under rephrasing)
3. Much larger datasets with more unfaithful examples
4. Naturalistic hint injection (not explicit preambles)

## Authors

- Brinlee Kidd (brinlee0kidd@gmail.com)
- Demosthenes (AI collaborator)

## License

MIT

# KV Cache Sycophancy Experiments

## Core Finding

**V-only patching reduced sycophancy by 29–33pp. K patching had no effect.**

Question: Is sycophancy partially correctable by editing cached V vectors at each turn?

## Experimental Setup

- **Model**: Qwen2.5-3B-Instruct
- **Method**: Patch V vectors (or K vectors) from a "clean" forward pass into a sycophantic context
- **Metric**: Rate of sycophantic agreement with user's incorrect claim

## Key Results

| Condition | Sycophancy Rate | Δ from Baseline |
|-----------|-----------------|-----------------|
| Baseline (sycophantic context) | ~47% | — |
| V-only patching | 14–18% | **−29 to −33pp** |
| K-only patching | ~47% | ~0pp |

## Open Questions

- Does this generalize across models?
- What specific V-vector components carry the sycophancy signal?
- Can this be applied as a runtime intervention?

## Directory Structure

- `scripts/` - Experiment code
- `data/` - Results and logs
- `writing_exercises/` - Exploratory paper drafts (not publication-ready)
- `memory/` - Session notes

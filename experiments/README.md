# Experiments

## Research Arc

**Starting question:** Can we reconstruct the implicit reasoning steps an LLM took?

**What we learned:** Full reconstruction is intractable. The achievable goal is *detecting* when reasoning goes wrong.

**Refined question:** Can we detect when stated reasoning doesn't match internal computation?

## Experiments

### [01_reconstruction](./01_reconstruction/)
*Feb 2026 — Exploratory pilot*

Early exploration of predicting reasoning traces using logit lens. Helped identify limitations that led us to pivot to detection.

### [02_divergence_detection](./02_divergence_detection/)
*Mar 2026 — Main results*

Binary classification: detect hidden hint usage in chain-of-thought. Key finding: cross-architecture training outperforms same-model training by 18 AUROC points.

### [03_posthoc_transfer](./03_posthoc_transfer/)
*Mar 2026 — Null result*

Tested whether the hint detector transfers to post-hoc rationalization detection. Result: no transfer (AUROC ~0.58), suggesting these are structurally different phenomena.

## Shared Code

Core utilities in `/src/`:
- `ground_truth.py` — Logit lens extraction
- `dataset.py` — Dataset generators

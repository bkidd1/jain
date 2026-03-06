# Experiments

This project evolved through two phases, each documented as a separate experiment.

## Research Arc

**Starting question:** Can we reconstruct the implicit reasoning steps an LLM took?

**What we learned:** Full reconstruction may be structurally intractable. The more achievable and safety-relevant goal is *detecting* when reasoning goes wrong.

**Refined question:** Can we detect when stated reasoning doesn't match actual computation?

## Experiments

### [01_reconstruction](./01_reconstruction/)
*Feb 2026 — Complete*

Trained a model to predict reasoning traces using logit lens outputs as ground truth. Achieved 40% F1 on cross-model transfer (Llama → Mistral), but identified fundamental limitations in the reconstruction framing.

### [02_divergence_detection](./02_divergence_detection/)
*Mar 2026 — In Progress*

Pivoted to binary classification: detect faithful vs unfaithful chain-of-thought. Uses hint paradigm to generate labeled pairs, trains a classifier, tests cross-model transfer.

## Shared Infrastructure

Core library code lives in `/src/`:
- `dataset.py` — Original dataset generators
- `ground_truth.py` — Logit lens extraction utilities
- `tuned_lens_extraction.py` — Alternative extraction methods

Both experiments import from `src/` to avoid code duplication.

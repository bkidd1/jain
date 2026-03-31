# Experiment 01: Reasoning Trace Reconstruction

**Status:** Complete (early exploration — see Experiment 02 for main results)  
**Dates:** Feb 2026

> ⚠️ This was an initial exploratory experiment that helped us identify limitations in the reconstruction approach. The main findings are in **Experiment 02: Cross-Architecture Hint Detection**.

## Goal

Train a model to predict the implicit reasoning steps an LLM took but didn't verbalize, using logit lens outputs as ground truth.

## Approach

1. Extract "reasoning traces" from Llama 3.2 1B using logit lens (top token predictions at each layer)
2. Train TinyLlama 1.1B + LoRA to predict these traces given only the prompt
3. Test cross-model transfer: can traces learned from Llama predict Mistral's reasoning?

## Key Results

- **Training:** Loss dropped from 3.8 → 0.9 over 70 steps
- **Cross-model transfer:** 40% F1 (64% precision, 30% recall) on Llama → Mistral
- **Dataset:** 74 examples (too small for statistical significance)

## What We Learned

The reconstruction framing has fundamental issues:
- **No clean ground truth** — logit lens outputs are proxies, not actual reasoning
- **Token F1 is wrong** — "Jobs" vs "Apple" scores 0% even though both are valid concepts
- **Zero ablation outdated** — field has moved to mean ablation
- **Scale too small** — 74 examples vs 10K+ standard in interpretability papers

These limitations led us to pivot to **divergence detection** (Experiment 02).

## Files

```
data/
  raw/           # Original prompts
  processed/     # Logit lens extractions  
  training/      # Train/val splits
models/
  rtp_v1/        # TinyLlama + LoRA checkpoints
scripts/
  extract_traces_fast.py    # Logit lens extraction
  train_rtp.py              # Training script
  test_cross_model_transfer.py
results/
  cross_model_transfer_results.json
  multi_model_transfer.json
```

## Citation

This experiment is documented in Section 3 of the paper as preliminary work that motivated the divergence detection approach.

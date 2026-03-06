# JAIN: Interpretability Transfer for Reasoning Detection

A research project investigating whether interpretability insights from open-weight models can transfer to detect reasoning anomalies in unseen architectures.

## Research Arc

**Initial question:** Can we reconstruct the implicit reasoning steps an LLM took but didn't verbalize?

**What we learned:** Full reconstruction may be structurally intractable (see Nanda's "pragmatic interpretability" pivot, Sept 2025). The more achievable and safety-relevant goal is *detecting* when reasoning goes wrong.

**Refined question:** Can a classifier trained on divergence patterns in open-weight models detect unfaithful chain-of-thought in unseen architectures?

## Project Structure

```
jain/
├── experiments/
│   ├── 01_reconstruction/     # Original RTP experiment (complete)
│   │   ├── data/              # Prompts, traces, training data
│   │   ├── models/            # TinyLlama + LoRA checkpoints
│   │   ├── scripts/           # Extraction, training, evaluation
│   │   └── results/           # Cross-model transfer results
│   └── 02_divergence_detection/  # Current work (in progress)
│       ├── data/              # Hint pairs, labeled examples
│       ├── models/            # Detector checkpoints
│       └── scripts/           # Generation, training, evaluation
├── src/                       # Shared library code
│   ├── ground_truth.py        # Logit lens extraction
│   ├── dataset.py             # Dataset generators
│   └── tuned_lens_extraction.py
├── paper/                     # LaTeX source
├── docs/                      # Research notes, figures
└── notebooks/                 # Exploration
```

## Key Results (Experiment 01)

- Trained TinyLlama + LoRA to predict reasoning traces from logit lens outputs
- Achieved **40% F1** on cross-model transfer (Llama → Mistral)
- Identified limitations: small dataset (74 examples), token F1 metric issues, zero ablation outdated

These findings motivated the pivot to divergence detection.

## Current Work (Experiment 02)

Detecting unfaithful chain-of-thought via binary classification:
1. Generate hint/no-hint prompt pairs on Llama
2. Extract internal traces + CoT outputs
3. Train classifier: does stated reasoning match internal computation?
4. Test transfer to Mistral, Qwen, DeepSeek

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Authors

- Brinlee Kidd (brinlee@gmail.com)
- Demosthenes (demo.hegemon@gmail.com)

## References

- [Anthropic CoT Faithfulness Study](https://arxiv.org/abs/2305.04388)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Nanda, N. "From Ambitious to Pragmatic Interpretability" (2025)

## License

MIT

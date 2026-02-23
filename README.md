# Jain: Reasoning Trace Predictor

A research project investigating whether a lightweight external model can predict the implicit reasoning steps a language model took but didn't verbalize, trained against mechanistic interpretability evidence as ground truth.

## Research Question

**Can a lightweight "reasoning trace predictor" (RTP), trained on mechanistic interpretability evidence from open-weight models, accurately reconstruct the implicit reasoning steps that a language model performed but did not verbalize?**

## Project Structure

```
jain/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── ground_truth.py     # Ground truth extraction (logit lens, probes, patching)
│   ├── dataset.py          # Dataset generation and management
│   └── rtp.py              # Reasoning Trace Predictor model
├── data/
│   ├── raw/                # Raw task prompts
│   └── processed/          # Processed (input, output, trace) triples
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Training and evaluation scripts
└── experiments/            # Experiment configs and results
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate transformer-lens einops jaxtyping
```

## Phase 1: Ground Truth Generation

Using TransformerLens to extract implicit reasoning traces:
1. Run input through model with hooks at every layer
2. Apply logit lens at each layer → get top-k predicted tokens per layer
3. Apply linear probes at key layers → extract concept activations
4. Use activation patching to confirm causal relevance
5. Record: ordered sequence of confirmed intermediate concepts = "reasoning trace"

## Task Domains

1. **Factual multi-hop reasoning** - e.g., "What is the capital of the state where Dallas is located?"
2. **Arithmetic with intermediate steps** - e.g., "What is 23 × 17?"
3. **Sentiment-influenced generation** - Biased prompts where the model's answer is influenced by framing

## References

- [Anthropic CoT Faithfulness Study](https://arxiv.org/abs/2305.04388) - "Reasoning Models Don't Always Say What They Think"
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) - Neel Nanda's interpretability library
- [Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - Interpreting intermediate predictions

## License

MIT

# JAIN: Cross-Architecture Detection of Hidden Hint Usage

Detecting when language models secretly process hints without acknowledging them in their Chain-of-Thought — using only text output.

## Key Finding

A detector trained on architectures that **exclude** the target outperforms same-model training by 18 AUROC points.

| Training Data | Test Model | AUROC |
|---------------|------------|-------|
| TinyLlama (same model) | TinyLlama | 0.746 |
| Qwen + Phi-2 (excludes target) | TinyLlama | **0.928** |

This transfers to larger models and different architecture families:

| Model | AUROC |
|-------|-------|
| DeepSeek-R1-7B | 0.909 |
| DeepSeek-R1-14B | 0.933 |
| Mistral-7B | 0.893 |

**Implication:** Train on small open models → deploy on large/closed models.

## Full Writeup

See **[MEMO.md](MEMO.md)** for detailed methodology and results.

## Project Structure

```
jain/
├── MEMO.md                           # Full writeup
├── experiments/
│   ├── 01_reconstruction/            # Initial RTP experiment
│   ├── 02_divergence_detection/      # Main hint detection work
│   │   ├── data/
│   │   │   ├── hint_pairs/           # Generated prompt pairs
│   │   │   ├── extractions/          # Model outputs + labels
│   │   │   └── models/               # Trained detectors
│   │   └── scripts/                  # Extraction & evaluation
│   └── 03_posthoc_transfer/          # Post-hoc rationalization test
├── src/                              # Shared utilities
└── notebooks/                        # Exploration
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run evaluation on new model
cd experiments/02_divergence_detection
python scripts/evaluate_transfer.py \
    --model_dir data/models/detector_3models \
    --test_data data/extractions/your_extractions.jsonl \
    --device mps
```

## Results Files

Key results in `experiments/02_divergence_detection/data/extractions/`:
- `transfer_results_*.json` — Per-model evaluation metrics
- `extractions_*.jsonl` — Raw model outputs with labels

## Authors

- Brinlee Kidd (brinlee@gmail.com)
- Demosthenes (Brinlee's AI assistant)

## License

MIT

# Experiment 01: Noise Injection for Unfaithfulness Detection

## Goal
Test whether hint-influenced Chain-of-Thought is more fragile under noise injection than genuine CoT.

## Hypothesis
When a model "works backwards from a hint," it constructs plausible-looking reasoning that ends at a predetermined answer. This is like building a bridge backwards — it looks coherent but is structurally weaker. Noise injection should destabilize these responses more than faithful ones.

## Method

Based on [Liu et al. (ICLR 2026)](https://arxiv.org/abs/2502.03799):

1. **Load model** (TinyLlama-1.1B initially, then scale up)
2. **For each question:**
   - Use the NO-HINT prompt version
   - Generate K=10 responses with:
     - Noise: Uniform(0, α) injected into MLP activations
     - Layers: Later ~40% of layers
     - Temperature: 0.5
3. **Measure answer variance** across K samples
4. **Compare variance** between:
   - Questions where model originally used hint (unfaithful)
   - Questions where model reasoned faithfully

## Dataset

Reuse v1 matched-prompt extractions:
```
archive/v1-text-detection/experiments/05_matched_prompts/data/extractions/
├── tinyllama_v2.jsonl
├── phi2_v2.jsonl
└── qwen2_v2.jsonl
```

Each record has:
- `question`: The math/reasoning question
- `hint_present`: Whether hint was in original prompt
- `response`: Model's CoT response
- `final_answer`: Extracted answer

## Metrics

1. **Answer Entropy**: H = -Σ p(a) log p(a) over K samples
2. **Semantic Variance**: Mean pairwise embedding distance
3. **Detection AUROC**: Using variance as unfaithfulness score

## Expected Results

| Outcome | AUROC | Interpretation |
|---------|-------|----------------|
| Strong signal | > 0.7 | Noise injection detects unfaithfulness |
| Weak signal | 0.55-0.7 | Marginal effect, needs investigation |
| Null | ≈ 0.5 | Unfaithfulness ≠ hallucination structurally |

## Files

```
01_noise_injection/
├── README.md           # This file
├── scripts/
│   ├── noise_injection.py    # Core implementation
│   └── evaluate.py           # Metrics and analysis
├── data/
│   └── (symlink to v1 extractions)
└── results/
    └── (experiment outputs)
```

## References

- Liu et al. (2026). "Enhancing Hallucination Detection through Noise Injection." ICLR 2026.
  https://arxiv.org/abs/2502.03799

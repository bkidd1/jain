# Experiment 04: Layer-by-Layer Amplification Probing

## Goal

Systematically measure which layers are most sensitive to activation steering by collecting quantifiable metrics (KL divergence, entropy change, top-k token shifts) at each layer.

## Method

For each of the 22 layers in TinyLlama:
1. Extract "hint direction" (mean hint activations - mean no-hint activations)
2. Apply amplification (factor=1.2) during generation
3. Compare baseline vs amplified:
   - **KL divergence** between output distributions
   - **Entropy change** (amplified vs baseline)
   - **Top-k token changes** (how many of top-10 tokens differ)
   - **Output change** (did text output change?)

## Results

### Layer Sensitivity (sorted by KL divergence)

| Layer | KL Div | Entropy Δ | Top-k Δ | Output Changed |
|-------|--------|-----------|---------|----------------|
| **4** | **0.052** | **+0.266** | 0 | ✓ |
| **5** | **0.043** | **+0.233** | 0 | ✓ |
| **3** | **0.014** | **+0.116** | 0 | ✓ |
| 10 | 0.010 | +0.130 | 1 | ✓ |
| 6 | 0.010 | +0.132 | 0 | ✓ |
| 12 | 0.009 | +0.061 | 1 | ✓ |
| 14 | 0.008 | +0.030 | 1 | ✓ |

### Key Finding

**Layers 3-5 are most sensitive to amplification** — they show the highest KL divergence and entropy change when the hint direction is amplified.

This contrasts with Experiment 03, where **layer 14 was best for detection** (perfect linear probe). 

### Interpretation

The hint representation appears to have two distinct roles:

1. **Early-middle layers (3-5)**: Where the hint direction has the most *influence* on output distribution — amplifying here causes the biggest shifts

2. **Middle layers (14)**: Where the hint is most *separable* — a linear classifier can perfectly detect hint presence

**Hypothesis**: Layers 3-5 encode "how much to weight the hint" while layer 14 encodes "whether there is a hint at all."

## Files

```
04_amplification_probing/
├── README.md
├── scripts/
│   └── layer_probe.py
└── results/
    ├── layer_metrics.jsonl (v1 - had NaN issues)
    └── layer_metrics_v2.jsonl (fixed)
```

## Next Steps

1. **Combine steering**: Amplify layers 3-5 together for maximum effect
2. **Compare with suppression**: Do the same layers dominate with factor=0.9?
3. **Test on stubborn examples**: Do early layers help flip cases that layer 14 alone couldn't?

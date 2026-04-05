# Experiment 03: Hint Detection Probe

## Goal

Build a classifier that detects when a model is processing hint-influenced input.

## Method

1. Extract activations from layer 14 (the "deception layer") for 594 hint + 594 no-hint prompts
2. Train logistic regression: hint (1) vs no-hint (0)
3. Evaluate on held-out test set (20%)

## Results

```
==================================================
RESULTS
==================================================
Accuracy: 1.000
AUROC:    1.000

Classification Report:
              precision    recall  f1-score   support

     no_hint       1.00      1.00      1.00       119
        hint       1.00      1.00      1.00       119

    accuracy                           1.00       238
```

**Perfect classification!** Layer 14 activations contain complete information about whether the prompt contains a hint.

## Top Dimensions

The most informative dimensions for hint detection:

| Dimension | Weight | Direction |
|-----------|--------|-----------|
| 668 | +0.91 | Hint |
| 1044 | -0.81 | No-hint |
| 486 | +0.77 | Hint |
| 1577 | +0.70 | Hint |
| 918 | -0.56 | No-hint |

## Interpretation

1. **The signal is there**: Layer 14 encodes hint-presence perfectly
2. **Linear separability**: A simple linear classifier suffices
3. **Specific dimensions**: A handful of dimensions carry most information

## Caveats

1. **Prompt detection vs behavior detection**: This detects hint *presence* in the prompt, not whether the model *follows* it
2. **Structural differences**: Hint prompts have "Context: The user believes..." which may be directly detected
3. **Same distribution**: Train/test from same distribution; real-world may differ

## Next Steps

1. **Test on new prompts**: Does it generalize to different hint phrasings?
2. **Detect hint *usage***: Train on cases where model follows hint vs ignores it
3. **Real-time detection**: Use probe during inference to flag suspicious generations
4. **Transfer to larger models**: Does the same layer/dimensions work?

## Files

```
03_detector/
├── README.md
├── scripts/
│   └── train_probe.py
├── data/
│   └── combined_v2.jsonl
└── results/
    ├── probe_results.json
    ├── probe_weights.npy
    └── probe_bias.npy
```

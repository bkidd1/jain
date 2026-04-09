# Experiment 07: LayerNorm Bypass Amplification

## Hypothesis

Our previous amplification experiments showed zero effect. One explanation: **RMSNorm is eating the signal**. 

RMSNorm does: `output = x / RMS(x) * γ`

When we amplify activations, RMSNorm immediately renormalizes them, undoing our intervention.

## Two Bypass Strategies

### 1. Post-Norm Intervention (`post_norm`)

Inject amplification **after** RMSNorm fires. The normalization has already happened, so our modification persists to downstream computation.

```
Normal:     hidden → RMSNorm → attention
Post-norm:  hidden → RMSNorm → [AMPLIFY] → attention
```

### 2. Frozen Stats Intervention (`frozen_stats`)

Capture the RMS statistics **before** we modify anything, apply amplification, then renormalize with the **original** statistics (not the new ones).

```
1. Compute RMS_original from hidden
2. Run normal RMSNorm  
3. Amplify the output
4. Correct: output * (RMS_original / RMS_new)
```

This preserves the *direction* of our amplification while maintaining the original scale.

## Baseline Control

`pre_norm_baseline` replicates our old approach: amplify before RMSNorm. Should show no effect (confirming experiment 06 results).

## Expected Outcomes

| Outcome | Interpretation |
|---------|----------------|
| Post-norm works, baseline doesn't | RMSNorm WAS eating the signal |
| Frozen-stats works, baseline doesn't | RMSNorm WAS eating the signal |
| Nothing works, even at 5x | Direction is correlational, not causal |
| Large factors cause gibberish | Model is robust until breaking point |

## Usage

```bash
cd ~/jain
source .venv/bin/activate
python experiments/07_layernorm_bypass/scripts/layernorm_bypass.py
```

## Files

```
07_layernorm_bypass/
├── README.md
├── scripts/
│   └── layernorm_bypass.py
└── results/
    └── layernorm_bypass_results.json
```

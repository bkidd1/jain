# V-Specificity Control Experiments

These experiments address the critical question: **Is V specifically where anti-sycophancy lives, or is it just "attention caches matter"?**

## Experiment 35: Random V Control

**Question:** Does *any* V patching help, or does clean V specifically carry the signal?

**Design:**
- Baseline: Sycophantic hint prompt, no patch
- Clean V: Patch with V from clean (no-hint) forward pass
- Random V: Patch with random vectors (same shape, matched L2 norm)
- Zero V: Ablate V entirely

**Results (n=100, Gemma-4 E2B, layer 13):**

| Condition | Correct | Rate | Δ from Baseline | 95% CI |
|-----------|---------|------|-----------------|--------|
| Baseline | 40 | 40% | — | 30.9-49.8% |
| Clean V | 72 | 72% | +32pp | 62.5-79.9% |
| Random V | 6 | 6% | -34pp | 2.8-12.5% |
| Zero V | 51 | 51% | +11pp | 41.3-60.6% |

**Interpretation:**
- ✅ Random V is *catastrophically worse* than baseline (6% vs 40%)
- ✅ CIs don't overlap between Clean V and Random V
- ✅ This rules out "any V disruption helps" — clean V specifically carries anti-sycophancy information
- Bonus: Zero V (+11pp) suggests model can partially recover when V is ablated, but random V actively interferes

**Files:**
- Script: `scripts/35_random_v_control.py`
- Results: `data/results/35_random_v_control.json`

---

## Experiment 36: Clean K Control

**Question:** Does clean K rescue like clean V does? If yes, V isn't special.

**Design:**
- Baseline: Sycophantic hint prompt, no patch
- Clean V: Patch V only from clean forward pass
- Clean K: Patch K only from clean forward pass
- Clean KV: Patch both K and V from clean forward pass

**Results (n=100, Gemma-4 E2B, layer 13):**

| Condition | Correct | Rate | Δ from Baseline | 95% CI |
|-----------|---------|------|-----------------|--------|
| Baseline | 40 | 40% | — | 30.9-49.8% |
| Clean V | 72 | 72% | +32pp | 62.5-79.9% |
| Clean K | 20 | 20% | -20pp | 13.3-28.9% |
| Clean KV | 40 | 40% | 0pp | 30.9-49.8% |

**Interpretation:**
- ✅ Clean K is *worse* than baseline (20% vs 40%)
- ✅ CIs don't overlap between Clean V (62.5-79.9%) and Clean K (13.3-28.9%)
- ✅ V specifically carries the rescue signal, not K
- Interesting: Clean KV = Baseline, suggesting K's negative effect cancels V's positive effect

**Files:**
- Script: `scripts/36_clean_k_control.py`
- Results: `data/results/36_clean_k_control.json`

---

## Combined Summary

| Control | Result | Implication |
|---------|--------|-------------|
| Random V vs Clean V | 6% vs 72% | Not just "disruption helps" |
| Clean K vs Clean V | 20% vs 72% | V is specifically where signal lives |
| Clean K vs Baseline | 20% vs 40% | K patching actively harms |
| Clean KV vs Clean V | 40% vs 72% | K contaminates V's benefit |

**Conclusion:** V-specificity is mechanistically supported. The framing "V vectors carry anti-sycophancy signal" is validated by these controls.

## Remaining Questions

1. **Why V and not K?** Hypothesis: V encodes "what to retrieve" while K encodes "what is present." Sycophancy may be about retrieval bias, not representation.

2. **Does this generalize?** V-specificity tested only on Gemma-4 E2B. Qwen showed KV contamination but V/K decomposition not tested.

3. **Per-layer decomposition:** Where does the Clean V effect concentrate? Is it diffuse or localized?

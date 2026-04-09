# Research Deep Dive: Why Mean-Difference Directions Don't Steer

## The Problem We Hit

We extracted "hint directions" via mean(hint_activations) - mean(no_hint_activations). These directions:
- Have high magnitude (19.7 at layer 17)
- Enable perfect classification (100% accuracy detecting hint presence)
- **But don't causally change behavior when amplified**

This is a known phenomenon in the literature.

---

## Key Insight #1: Input Features ≠ Output Features

**Paper**: "SAEs Are Good for Steering – If You Select the Right Features" (Arad et al., May 2025)

The critical distinction:
- **Input features**: Capture patterns in the model's input (what we found)
- **Output features**: Actually affect the model's output when intervened on

> "A feature's activations are not necessarily the same as its causal effect on the model's output."

**Our mean-difference direction is an INPUT feature** — it detects hint presence but doesn't control hint-following behavior.

### Solution from this paper:
1. Use SAEs to decompose activations into monosemantic features
2. Compute "output scores" by intervening on each feature and measuring output change
3. Filter for features with high output scores
4. **Result: 2-3x improvement in steering effectiveness**

---

## Key Insight #2: Superposition Kills Simple Interventions

**Paper**: "Scaling Monosemanticity" (Anthropic, 2024)

Neural networks encode many concepts in superposition — multiple features share the same dimensions. Our "hint direction" is likely a **statistical blur of many overlapping features**, not a clean lever.

**SAEs address this** by learning a sparse, overcomplete dictionary that decomposes superposed representations into monosemantic features.

Example: Anthropic's "Golden Gate Claude" amplified a single SAE feature for the Golden Gate Bridge and successfully made Claude reference it constantly.

---

## Key Insight #3: Causal Tracing Finds the Real Levers

**Paper**: "Locating and Editing Factual Associations in GPT" (Meng et al., NeurIPS 2022) — the ROME paper

Instead of guessing where to intervene, use **causal tracing**:
1. Run the model normally, record all activations
2. Corrupt the input (e.g., replace subject with noise)
3. Restore activations one-at-a-time and measure effect on output
4. **Activations that restore correct output are causally important**

**Key finding**: Factual knowledge is stored in **MLP modules at middle layers**, specifically when processing the **last token of the subject**.

For editing factual associations, they modify MLP weights directly (Rank-One Model Editing) rather than adding steering vectors.

---

## Key Insight #4: Layer Timing Matters

**Multiple sources** (Mandliya 2025, EmergentMind):

> "If we inject the steering vector too early, the later layers may overwrite the intervention. Injecting too late means the computation has already converged."

Our intervention at layer 17 (middle-ish) may be:
- Too late if the "hint decision" was made earlier
- Overwritten by later layers that don't respect the perturbation

---

## Key Insight #5: Difference-in-Means Has Known Limitations

**Paper**: "Momentum Steering: Activation Steering Meets Optimization" (OpenReview, Oct 2025)

> "Unlike traditional difference-in-means methods, our framework generates a richer family of candidate directions through momentum updates, enabling more expressive steering."

They explicitly acknowledge difference-in-means is limited and propose:
1. Accumulate signals via momentum updates
2. Model causal effects of interventions on downstream activations
3. Generate a family of candidate directions, not just one

---

## Recommended Next Steps

### Option A: Use Sparse Autoencoders
1. Train or obtain SAE for Gemma 4
2. Decompose hint/no-hint activations into SAE features
3. Find features with high "output scores" (causal effect on generation)
4. Steer using those specific features

**Pros**: Directly addresses superposition problem
**Cons**: Requires SAE (may need to train one)

### Option B: Causal Tracing
1. Use activation patching to find which components mediate sycophancy
2. Identify specific attention heads or MLP layers that are decisive
3. Intervene only on those components

**Pros**: Principled, finds actual causal structure
**Cons**: Computationally intensive

### Option C: KV Cache Intervention
Test hypothesis that hint influence is encoded during prompt processing:
1. Compare KV caches between hint/no-hint prompts
2. Intervene on KV cache rather than generation activations
3. See if this has causal effect

**Pros**: Tests specific hypothesis about timing
**Cons**: Less explored in literature

### Option D: Weight Editing (ROME-style)
Instead of inference-time steering, edit the model weights:
1. Use causal tracing to find where sycophancy is encoded
2. Apply rank-one update to MLP weights
3. Permanently modify the behavior

**Pros**: Actually changes the model
**Cons**: Requires more sophisticated intervention

---

## Summary Table

| Method | What We Did | Why It Failed | Better Alternative |
|--------|-------------|---------------|-------------------|
| Mean-difference direction | Averaged activations | Correlational, not causal | SAE features with output scores |
| Single direction | One vector | Superposition blur | Sparse decomposition |
| Mid-layer intervention | Layer 17 | May be overwritten | Causal tracing to find right layer |
| Generation-time steering | During decoding | Decision already made? | KV cache or prompt-time intervention |
| Additive steering | Add scaled vector | LayerNorm/robustness | Weight editing (ROME) |

---

## Key Papers to Read

1. **"SAEs Are Good for Steering"** — Feature selection for steering
   https://arxiv.org/html/2505.20063v1

2. **"Locating and Editing Factual Associations in GPT"** — ROME, causal tracing
   https://rome.baulab.info/

3. **"Scaling Monosemanticity"** — SAEs for interpretability
   https://transformer-circuits.pub/2024/scaling-monosemanticity/

4. **"Activation Addition"** — Representation engineering survey
   https://arxiv.org/abs/2308.10248

5. **"Representation Engineering Survey"** — Comprehensive overview
   https://arxiv.org/html/2502.17601v1

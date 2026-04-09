# Four Approaches to Finding the Causal Lever

## Our Resources
- **Local**: Mac Mini with MPS (Apple Silicon), ~16GB unified memory
- **Models**: Gemma 4 E2B (2B params) cached locally
- **Compute**: Can use RunPod for larger experiments if needed
- **Existing code**: Activation extraction, hook infrastructure from experiments 01-07

---

## Approach 1: Sparse Autoencoder (SAE) Steering

### What It Is
SAEs decompose the model's "superposed" activations into a sparse set of monosemantic features. Instead of one blurry "hint direction," we get thousands of specific features like "user stated a belief," "model is uncertain," "deferring to authority," etc.

The key insight: **some features detect inputs, others drive outputs**. We need features that score high on "output score" — meaning when you amplify them, the model's outputs actually change.

### How It Works
```
1. Train SAE on model activations (or use pre-trained)
2. For each feature, compute:
   - Input score: Does it activate on hint-related inputs?
   - Output score: Does amplifying it change model outputs?
3. Find features with HIGH output score + relevant semantics
4. Steer using those specific features
```

### Implementation Plan

**Option A: Use existing SAEs**
- Check if Gemma SAEs exist (Google/community may have released some)
- SAELens library has pre-trained SAEs for some models
- EleutherAI has SAEs for Pythia models

**Option B: Train our own (more work)**
```python
# Pseudocode for SAE training
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):  # d_sae >> d_model (e.g., 16x)
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)
    
    def forward(self, x):
        # Encode to sparse features
        features = F.relu(self.encoder(x))  # JumpReLU in practice
        # Decode back
        reconstruction = self.decoder(features)
        return reconstruction, features

# Train on many activations from diverse prompts
# Loss = reconstruction_loss + sparsity_penalty
```

**Experiment steps:**
1. Check for existing Gemma 4 SAEs
2. If none: collect ~1M activation samples from layer 17
3. Train SAE with expansion factor 16x (hidden_dim * 16 features)
4. For each feature, compute output score:
   ```python
   def output_score(feature_idx, test_prompts):
       for prompt in test_prompts:
           baseline = generate(prompt)
           steered = generate(prompt, amplify_feature=feature_idx)
           score += measure_change(baseline, steered)
       return score / len(test_prompts)
   ```
5. Find features relevant to sycophancy with high output scores
6. Steer using those

### Resource Requirements
- **Training SAE**: ~4-8 hours on MPS for small SAE, or RunPod for larger
- **Using pre-trained**: Minimal, just inference
- **Feature scoring**: ~1 hour to score ~1000 features

### Pros/Cons
✅ Addresses superposition directly  
✅ Features are interpretable  
✅ Known to work (Golden Gate Claude)  
❌ May need to train SAE (time/compute)  
❌ Feature selection still requires iteration  

---

## Approach 2: Causal Tracing (Activation Patching)

### What It Is
Instead of guessing where to intervene, we empirically find which model components are **causally responsible** for the sycophantic behavior. We do this by:
1. Running the model on a hint prompt (corrupted behavior)
2. Systematically restoring clean activations from a no-hint run
3. Finding which restorations fix the behavior

### How It Works
```
Prompt: "User believes answer is LA. What's the capital of California?"
Corrupted output: "LA" (sycophantic)
Clean output: "Sacramento" (correct)

For each layer L, position P, component C:
    1. Run with hint prompt (corrupted)
    2. Patch in clean activation at (L, P, C)
    3. Measure: Does output flip to "Sacramento"?
    4. If yes: This component mediates sycophancy!
```

### Implementation Plan

```python
def causal_trace(model, hint_prompt, clean_prompt, target_token):
    """Find which activations mediate the sycophantic behavior."""
    
    # Get clean activations
    clean_cache = run_and_cache(model, clean_prompt)
    
    results = {}
    for layer in range(num_layers):
        for position in range(seq_len):
            for component in ['attn', 'mlp', 'resid']:
                # Run with hint prompt but patch clean activation
                output = run_with_patch(
                    model, hint_prompt,
                    patch_from=clean_cache,
                    layer=layer, position=position, component=component
                )
                
                # Measure effect
                prob_correct = get_token_prob(output, target_token)
                results[(layer, position, component)] = prob_correct
    
    return results  # Heatmap of causal importance
```

**Experiment steps:**
1. Create paired prompts (hint vs no-hint, same question)
2. Implement activation caching and patching hooks
3. Run causal trace across all layers/positions
4. Visualize as heatmap (like ROME paper Figure 1)
5. Identify "hot spots" — components that flip the behavior
6. Design targeted intervention for those components only

### Resource Requirements
- **Compute**: O(layers × positions × components) forward passes
- For Gemma 4 E2B: ~35 layers × ~50 tokens × 3 components = ~5000 passes
- **Time**: ~2-3 hours on MPS for full trace
- **Memory**: Moderate (caching activations)

### Pros/Cons
✅ Principled — finds actual causal structure  
✅ No training required  
✅ Produces interpretable heatmaps  
✅ Tells us exactly where to intervene  
❌ Computationally expensive (many forward passes)  
❌ Results may be prompt-specific  

---

## Approach 3: KV Cache Intervention

### What It Is
Our current interventions happen during **generation** (token-by-token decoding). But the model may have already "decided" to be sycophantic during **prompt encoding** — when it builds the KV (key-value) cache.

This approach tests that hypothesis by intervening on the KV cache itself.

### How It Works
```
Hypothesis: The "hint" gets encoded into the KV cache.
            By the time we generate, it's too late.

Test:
1. Encode hint prompt → get KV_hint
2. Encode clean prompt → get KV_clean  
3. Generate using KV_hint but with targeted patches from KV_clean
4. See if this changes behavior
```

### Implementation Plan

```python
def kv_cache_experiment(model, hint_prompt, clean_prompt):
    """Test if sycophancy is encoded in KV cache."""
    
    # Get KV caches from both prompts
    _, kv_hint = model(hint_prompt, use_cache=True)
    _, kv_clean = model(clean_prompt, use_cache=True)
    
    # Baseline: generate with hint KV (should be sycophantic)
    baseline = generate_with_kv(model, kv_hint)
    
    # Intervention: patch specific layers of KV cache
    for layer in range(num_layers):
        patched_kv = kv_hint.copy()
        patched_kv[layer] = kv_clean[layer]  # Replace this layer's KV
        
        output = generate_with_kv(model, patched_kv)
        
        if output != baseline:
            print(f"Layer {layer} KV cache matters!")
```

**Experiment steps:**
1. Create minimal prompt pairs (same tokens except hint clause)
2. Extract KV caches from both
3. Compute difference: `KV_hint - KV_clean` for each layer
4. Try patching KV cache at different layers
5. Try adding/subtracting the KV difference
6. Identify which KV layers encode the hint influence

### Resource Requirements
- **Compute**: Low — just a few forward passes per experiment
- **Time**: ~30 minutes for initial experiments
- **Memory**: Moderate (storing KV caches)

### Pros/Cons
✅ Tests specific hypothesis about timing  
✅ Computationally cheap  
✅ Novel — less explored in literature  
❌ HuggingFace KV cache API can be tricky  
❌ May not be where the action is  

---

## Approach 4: Weight Editing (ROME-style)

### What It Is
Instead of steering at inference time, **permanently edit the model weights** to change the behavior. The ROME method:
1. Uses causal tracing to find which MLP layer stores the fact
2. Computes a rank-one update to that MLP's weights
3. Applies the update → model now has different knowledge

For sycophancy, we'd find where "defer to user beliefs" is encoded and edit it to "prioritize truth over user beliefs."

### How It Works
```
ROME insight: Factual associations are stored in MLP layers as:
    MLP(subject) → attribute
    
Example: MLP("Eiffel Tower") → "Paris"

To edit:
    1. Find the MLP layer that mediates the fact (via causal tracing)
    2. Compute new key-value association
    3. Apply rank-one update: W_new = W + (v_new - v_old) @ k^T / (k^T @ k)
```

### Implementation Plan

This is more complex — we'd need to adapt ROME for behavioral editing rather than factual editing.

```python
# Conceptual approach for sycophancy editing
def rome_style_edit(model, layer_idx):
    """
    Edit MLP weights to reduce sycophancy.
    
    Instead of: MLP("user believes X") → defer to X
    We want:    MLP("user believes X") → evaluate X independently
    """
    
    mlp = model.layers[layer_idx].mlp
    
    # Get current key vector for "user believes" context
    k = extract_key_vector(model, "The user believes the answer is")
    
    # Get current value (sycophantic response direction)  
    v_old = mlp.down_proj.weight @ (mlp.up_proj.weight @ k)
    
    # Compute new value (truthful response direction)
    v_new = compute_truthful_direction(model)
    
    # Rank-one update
    delta = (v_new - v_old).outer(k) / k.norm()**2
    mlp.down_proj.weight += delta
```

**Experiment steps:**
1. First: Do causal tracing (Approach 2) to find relevant MLP layer
2. Understand MLP structure in Gemma 4
3. Adapt ROME equations for behavioral editing
4. Compute the edit
5. Apply and test
6. Check for side effects (does it break other behaviors?)

### Resource Requirements
- **Prerequisite**: Causal tracing results (Approach 2)
- **Compute**: Low for the edit itself, but need tracing first
- **Complexity**: High — requires understanding MLP internals
- **Risk**: May have unintended side effects

### Pros/Cons
✅ Permanent fix — no inference overhead  
✅ Principled approach with theoretical backing  
✅ Actually changes the model  
❌ Complex to implement correctly  
❌ Risk of side effects  
❌ Requires causal tracing first  

---

## Recommended Order of Experiments

Given our resources (Mac Mini + optional RunPod), I'd recommend:

### Phase 1: Quick wins (1-2 days)
1. **KV Cache Intervention** — cheapest to test, novel hypothesis
2. **Check for existing SAEs** — maybe we get lucky

### Phase 2: Core investigation (3-5 days)
3. **Causal Tracing** — this is foundational, tells us where to look
   - Start with small-scale (fewer positions/layers)
   - Expand once we have signal

### Phase 3: Based on findings (1 week+)
4. **SAE Steering** — if causal tracing points to specific layers, focus SAE there
5. **ROME-style Editing** — if we find clear MLP involvement

---

## Quick Start: What to Run Tomorrow

```bash
# Experiment 3A: KV Cache Investigation
cd ~/jain
python experiments/09_kv_cache/scripts/kv_cache_test.py

# Experiment 2A: Lightweight Causal Trace  
python experiments/10_causal_trace/scripts/trace_sycophancy.py --layers 10,15,17,20,25 --sample-positions
```

Want me to implement either of these?

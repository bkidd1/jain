#!/usr/bin/env python3
"""
First Llama 3.1 8B logit lens experiment.

This is the core proof-of-concept: can we see the model's intermediate
reasoning steps (e.g., "Texas") before it outputs the final answer ("Austin")?
"""

import torch
from transformer_lens import HookedTransformer

print("=" * 60)
print("Llama 3.1 8B Logit Lens Experiment")
print("=" * 60)

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load model (this will take a minute and ~16GB memory)
print("\nLoading meta-llama/Llama-3.1-8B...")
print("(This may take 2-3 minutes on first run as it downloads ~16GB)")

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device=device,
    dtype=torch.float16,  # Use float16 to save memory
)

print(f"✓ Loaded! {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions")

# Our test prompt - the classic multi-hop reasoning example
prompt = "The capital of Texas is"
print(f"\n{'=' * 60}")
print(f"Prompt: {prompt!r}")
print("=" * 60)

# Tokenize
tokens = model.to_tokens(prompt)
print(f"Tokens: {tokens.shape[1]} tokens")

# Run with cache to capture all intermediate activations
print("\nRunning forward pass with activation cache...")
logits, cache = model.run_with_cache(prompt)

# Get the final prediction
final_probs = torch.softmax(logits[0, -1], dim=-1)
top_final = torch.topk(final_probs, 5)
print(f"\n📍 FINAL OUTPUT (after all {model.cfg.n_layers} layers):")
for prob, idx in zip(top_final.values, top_final.indices):
    token = model.tokenizer.decode([idx])
    print(f"   {token!r}: {prob:.3f}")

# Now the magic: logit lens at intermediate layers
print(f"\n{'=' * 60}")
print("🔍 LOGIT LENS: What does the model 'think' at each layer?")
print("=" * 60)

# Check every 4th layer for a cleaner view
layers_to_check = list(range(0, model.cfg.n_layers, 4)) + [model.cfg.n_layers - 1]

for layer in layers_to_check:
    # Get residual stream at this layer
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    
    # Apply final layer norm and unembed to get "what would the model predict here?"
    normalized = model.ln_final(resid)
    layer_logits = model.unembed(normalized)
    
    # Get probabilities for last token position
    probs = torch.softmax(layer_logits[0, -1], dim=-1)
    top_probs = torch.topk(probs, 3)
    
    # Format the top predictions
    preds = []
    for prob, idx in zip(top_probs.values, top_probs.indices):
        token = model.tokenizer.decode([idx]).strip()
        preds.append(f"{token!r}({prob:.2f})")
    
    print(f"Layer {layer:2d}: {', '.join(preds)}")

print(f"\n{'=' * 60}")
print("🎯 INTERPRETATION")
print("=" * 60)
print("""
If the logit lens shows "Texas" appearing in early/middle layers
before "Austin" dominates in later layers, that's evidence of
implicit multi-hop reasoning:

  Input: "The capital of Texas is"
  Layer N:   ... → "Texas" (retrieval/recognition)
  Layer N+k: ... → "Austin" (capital lookup)

This is the ground truth signal we'll train the RTP to predict!
""")

#!/usr/bin/env python3
"""
Quick test to verify the setup works.
Tests TransformerLens with a small model first.
"""

import sys
print(f"Python: {sys.version}")

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    print(f"   ✓ torch {torch.__version__}")
    print(f"   ✓ MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"   ✗ torch: {e}")

try:
    import transformer_lens
    version = getattr(transformer_lens, '__version__', 'installed')
    print(f"   ✓ transformer_lens {version}")
except ImportError as e:
    print(f"   ✗ transformer_lens: {e}")

try:
    import transformers
    print(f"   ✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"   ✗ transformers: {e}")

try:
    import einops
    print(f"   ✓ einops")
except ImportError as e:
    print(f"   ✗ einops: {e}")

# Test basic TransformerLens with tiny model
print("\n2. Testing TransformerLens with GPT-2 small (quick test)...")
try:
    from transformer_lens import HookedTransformer
    
    # Use GPT-2 small for quick testing (124M params, fast to load)
    model = HookedTransformer.from_pretrained(
        "gpt2",
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"   ✓ Loaded GPT-2 on {model.cfg.device}")
    
    # Test forward pass
    prompt = "The capital of France is"
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    print(f"   ✓ Forward pass works, output shape: {logits.shape}")
    
    # Test logit lens (basic version)
    print("\n3. Testing logit lens...")
    _, cache = model.run_with_cache(prompt)
    
    # Get predictions at a few layers
    for layer in [0, 5, 11]:  # First, middle, last
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        # Apply layer norm and unembed
        normalized = model.ln_final(resid)
        layer_logits = model.unembed(normalized)
        probs = torch.softmax(layer_logits[0, -1], dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        top_token = model.tokenizer.decode([top_idx.item()])
        print(f"   Layer {layer:2d}: top prediction = {top_token!r} ({top_prob:.3f})")
    
    print("\n✓ All tests passed! Ready for Mistral-7B.")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Device info:")
if torch.backends.mps.is_available():
    print("   MPS (Apple Silicon GPU) is available")
    # Note: Can't easily get MPS memory like CUDA
else:
    print("   MPS not available, will use CPU")

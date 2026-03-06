#!/usr/bin/env python3
"""
Extended logit lens experiments on Llama 3.1 8B.

Tests:
1. Multi-hop reasoning (city → state → capital)
2. Arithmetic (decomposition)
3. Adversarial (capital → state → largest city)
"""

import torch
from transformer_lens import HookedTransformer

print("Loading Llama 3.1 8B (cached, should be fast)...")
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device=device,
    dtype=torch.float16,
)
print(f"✓ Loaded on {device}\n")


def run_logit_lens(prompt: str, layers_to_show: list = None):
    """Run logit lens and show intermediate predictions."""
    if layers_to_show is None:
        layers_to_show = [0, 8, 16, 20, 24, 28, 31]
    
    print(f"{'='*60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'='*60}")
    
    # Run with cache
    logits, cache = model.run_with_cache(prompt)
    
    # Final output
    final_probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = torch.topk(final_probs, 5)
    print(f"\n📍 Final output:")
    for p, i in zip(top5.values, top5.indices):
        print(f"   {model.tokenizer.decode([i])!r}: {p:.3f}")
    
    # Logit lens at each layer
    print(f"\n🔍 Logit lens by layer:")
    results = {}
    for layer in layers_to_show:
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        normalized = model.ln_final(resid)
        layer_logits = model.unembed(normalized)
        probs = torch.softmax(layer_logits[0, -1], dim=-1)
        top3 = torch.topk(probs, 3)
        
        preds = []
        for p, i in zip(top3.values, top3.indices):
            token = model.tokenizer.decode([i]).strip()
            preds.append(f"{token!r}({p:.2f})")
            results[layer] = results.get(layer, []) + [(token, p.item())]
        
        print(f"   Layer {layer:2d}: {', '.join(preds)}")
    
    print()
    return results


# ============================================================
# EXPERIMENT 1: Multi-hop reasoning variations
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Multi-hop Reasoning")
print("="*60)

multi_hop_prompts = [
    "The capital of Texas is",
    "Dallas is a city in the state of",
    "The largest city in Texas is",
    "Austin is the capital of",
]

for prompt in multi_hop_prompts:
    run_logit_lens(prompt)


# ============================================================
# EXPERIMENT 2: Arithmetic
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Arithmetic Reasoning")
print("="*60)

arithmetic_prompts = [
    "23 + 17 =",
    "What is 15 × 4? The answer is",
    "If I have 100 and subtract 37, I get",
]

for prompt in arithmetic_prompts:
    run_logit_lens(prompt)


# ============================================================
# EXPERIMENT 3: Adversarial multi-hop
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Adversarial Multi-hop")
print("="*60)
print("(Tests if model uses real reasoning vs surface heuristics)")

adversarial_prompts = [
    # Should go: Austin → Texas → Houston (NOT Austin)
    "The largest city in the state whose capital is Austin is",
    # Should go: Sacramento → California → Los Angeles
    "The largest city in the state whose capital is Sacramento is",
]

for prompt in adversarial_prompts:
    run_logit_lens(prompt)


# ============================================================
# EXPERIMENT 4: Knowledge retrieval chain
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Knowledge Retrieval Chains")
print("="*60)

knowledge_prompts = [
    "The CEO of Apple is",
    "Apple was founded by Steve",
    "The company that makes the iPhone is",
]

for prompt in knowledge_prompts:
    run_logit_lens(prompt)


print("\n" + "="*60)
print("✓ All experiments complete!")
print("="*60)

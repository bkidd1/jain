#!/usr/bin/env python3
"""
Cross-Model Transfer Sanity Check

The big question: If we learn to predict reasoning traces from Llama's internals,
does that knowledge transfer to a different model (Mistral)?

Test approach:
1. Extract logit lens predictions from Llama 3.1 8B on test prompts
2. Extract logit lens predictions from Mistral 7B on same prompts
3. Compare: Do the same concepts appear at similar layers?

If similar patterns emerge, transfer is plausible.
If completely different, we need to rethink.
"""

import torch
from transformer_lens import HookedTransformer
import json
from typing import Dict, List, Tuple


def get_layer_predictions(
    model: HookedTransformer, 
    prompt: str,
    layers: List[int]
) -> Dict[int, List[Tuple[str, float]]]:
    """Get top predictions at specified layers."""
    _, cache = model.run_with_cache(prompt)
    
    results = {}
    for layer in layers:
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        normalized = model.ln_final(resid)
        logits = model.unembed(normalized)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        top = torch.topk(probs, 5)
        results[layer] = [
            (model.tokenizer.decode([idx]).strip(), prob.item())
            for prob, idx in zip(top.values, top.indices)
        ]
    
    return results


def compare_models():
    """Compare Llama and Mistral on same prompts."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Test prompts
    prompts = [
        "The capital of Texas is",
        "Dallas is a city in the state of",
        "The largest city in California is",
        "Apple was founded by Steve",
        "The company that makes Windows is",
    ]
    
    results = {"prompts": prompts, "llama": {}, "mistral": {}}
    
    # ========== LLAMA 3.1 8B ==========
    print("="*60)
    print("Loading Llama 3.1 8B...")
    print("="*60)
    
    llama = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.float16,
    )
    print(f"✓ Llama loaded ({llama.cfg.n_layers} layers)")
    
    # Check layers at 25%, 50%, 75%, 100% depth
    llama_layers = [
        llama.cfg.n_layers // 4,
        llama.cfg.n_layers // 2, 
        3 * llama.cfg.n_layers // 4,
        llama.cfg.n_layers - 1
    ]
    
    for prompt in prompts:
        preds = get_layer_predictions(llama, prompt, llama_layers)
        results["llama"][prompt] = {
            f"L{l} ({int(100*l/llama.cfg.n_layers)}%)": preds[l]
            for l in llama_layers
        }
        
        print(f"\n{prompt!r}")
        for layer in llama_layers:
            top = preds[layer][0]
            print(f"  Layer {layer:2d} ({100*layer//llama.cfg.n_layers:2d}%): {top[0]!r} ({top[1]:.2f})")
    
    # Free memory
    del llama
    torch.mps.empty_cache() if device == "mps" else None
    
    # ========== MISTRAL 7B ==========
    print("\n" + "="*60)
    print("Loading Mistral 7B...")
    print("="*60)
    
    mistral = HookedTransformer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        device=device,
        dtype=torch.float16,
    )
    print(f"✓ Mistral loaded ({mistral.cfg.n_layers} layers)")
    
    # Same relative positions
    mistral_layers = [
        mistral.cfg.n_layers // 4,
        mistral.cfg.n_layers // 2,
        3 * mistral.cfg.n_layers // 4,
        mistral.cfg.n_layers - 1
    ]
    
    for prompt in prompts:
        preds = get_layer_predictions(mistral, prompt, mistral_layers)
        results["mistral"][prompt] = {
            f"L{l} ({int(100*l/mistral.cfg.n_layers)}%)": preds[l]
            for l in mistral_layers
        }
        
        print(f"\n{prompt!r}")
        for layer in mistral_layers:
            top = preds[layer][0]
            print(f"  Layer {layer:2d} ({100*layer//mistral.cfg.n_layers:2d}%): {top[0]!r} ({top[1]:.2f})")
    
    # ========== COMPARISON ==========
    print("\n" + "="*60)
    print("COMPARISON: Do the same concepts appear?")
    print("="*60)
    
    for prompt in prompts:
        print(f"\n{prompt!r}")
        
        # Get top tokens from 75% depth (where reasoning usually crystallizes)
        llama_75 = results["llama"][prompt][f"L{llama_layers[2]} ({int(100*llama_layers[2]/32)}%)"]
        mistral_75 = results["mistral"][prompt][f"L{mistral_layers[2]} ({int(100*mistral_layers[2]/32)}%)"]
        
        llama_tokens = set(t[0].lower() for t in llama_75[:3])
        mistral_tokens = set(t[0].lower() for t in mistral_75[:3])
        
        overlap = llama_tokens & mistral_tokens
        
        print(f"  Llama top-3:   {[t[0] for t in llama_75[:3]]}")
        print(f"  Mistral top-3: {[t[0] for t in mistral_75[:3]]}")
        print(f"  Overlap: {overlap if overlap else 'NONE'}")
    
    # Save results
    with open("experiments/cross_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ Results saved to experiments/cross_model_comparison.json")
    print("="*60)
    
    return results


if __name__ == "__main__":
    compare_models()

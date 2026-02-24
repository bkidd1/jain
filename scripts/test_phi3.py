#!/usr/bin/env python3
"""Quick test of Phi-3 for multi-model comparison."""

import torch
from transformer_lens import HookedTransformer

TEST_PROMPTS = [
    "The capital of Texas is",
    "Dallas is a city in the state of",
    "The capital of France is",
    "Apple was founded by Steve",
    "The company that makes Windows is",
]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Loading Phi-3-mini...")
    try:
        model = HookedTransformer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device=device,
            dtype=torch.float16,
        )
        print(f"✓ Loaded ({model.cfg.n_layers} layers)")
        
        layer = int(model.cfg.n_layers * 0.75)
        
        for prompt in TEST_PROMPTS:
            with torch.no_grad():
                _, cache = model.run_with_cache(prompt)
                resid = cache[f"blocks.{layer}.hook_resid_post"]
                normalized = model.ln_final(resid)
                logits = model.unembed(normalized)
                probs = torch.softmax(logits[0, -1], dim=-1)
                top_prob, top_idx = probs.max(dim=-1)
                token = model.tokenizer.decode([top_idx.item()]).strip()
            
            print(f"  {prompt:45s} → {token:15s} ({top_prob.item():.2f})")
            
    except Exception as e:
        print(f"✗ Failed: {e}")

if __name__ == "__main__":
    main()

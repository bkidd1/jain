#!/usr/bin/env python3
"""
Multi-Model Transfer Test

Test if reasoning patterns transfer across multiple model architectures.
Train RTP on Llama → test on Mistral, Qwen, Gemma, etc.
"""

import torch
import json
import gc
from pathlib import Path
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple


# Models to test (all ~7B, fit in 64GB)
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-7B",
    "gemma": "google/gemma-7b",
}

# Test prompts (factual, multi-hop)
TEST_PROMPTS = [
    "The capital of Texas is",
    "Dallas is a city in the state of",
    "The capital of France is",
    "Apple was founded by Steve",
    "The company that makes Windows is",
    "The largest city in California is",
    "Berlin is the capital of",
    "The CEO of Tesla is",
]


def get_layer_predictions(
    model: HookedTransformer,
    prompt: str,
    layer_pct: float = 0.75  # 75% depth
) -> Tuple[str, float]:
    """Get top prediction at specified layer depth."""
    
    layer = int(model.cfg.n_layers * layer_pct)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        normalized = model.ln_final(resid)
        logits = model.unembed(normalized)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        top_prob, top_idx = probs.max(dim=-1)
        top_token = model.tokenizer.decode([top_idx.item()]).strip()
        
    return top_token, top_prob.item()


def test_model(model_name: str, model_id: str, device: str) -> Dict:
    """Test a single model on all prompts."""
    
    print(f"\n{'='*60}")
    print(f"Loading {model_name}: {model_id}")
    print("="*60)
    
    try:
        model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            dtype=torch.float16,
        )
        print(f"✓ Loaded ({model.cfg.n_layers} layers)")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return {"error": str(e)}
    
    results = {}
    
    for prompt in TEST_PROMPTS:
        token, prob = get_layer_predictions(model, prompt)
        results[prompt] = {
            "top_token": token,
            "probability": round(prob, 3)
        }
        print(f"  {prompt[:40]:40s} → {token:15s} ({prob:.2f})")
    
    # Cleanup
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    
    return results


def compute_overlap(results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Compute concept overlap between all model pairs."""
    
    models = list(results.keys())
    overlap_matrix = {}
    
    for m1 in models:
        overlap_matrix[m1] = {}
        for m2 in models:
            if m1 == m2:
                overlap_matrix[m1][m2] = 1.0
                continue
            
            # Count matching predictions
            matches = 0
            total = 0
            
            for prompt in TEST_PROMPTS:
                if prompt in results[m1] and prompt in results[m2]:
                    r1 = results[m1][prompt]
                    r2 = results[m2][prompt]
                    
                    if "error" not in r1 and "error" not in r2:
                        total += 1
                        # Case-insensitive match
                        if r1["top_token"].lower() == r2["top_token"].lower():
                            matches += 1
            
            overlap_matrix[m1][m2] = matches / total if total > 0 else 0.0
    
    return overlap_matrix


def main():
    print("="*60)
    print("Multi-Model Transfer Test")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    results = {}
    
    # Test each model
    for name, model_id in MODELS.items():
        results[name] = test_model(name, model_id, device)
    
    # Compute overlap matrix
    print("\n" + "="*60)
    print("CONCEPT OVERLAP MATRIX")
    print("="*60)
    
    overlap = compute_overlap(results)
    
    # Print matrix
    models = list(MODELS.keys())
    header = "          " + "  ".join(f"{m:8s}" for m in models)
    print(header)
    print("-" * len(header))
    
    for m1 in models:
        row = f"{m1:8s}  "
        for m2 in models:
            val = overlap.get(m1, {}).get(m2, 0)
            row += f"{val:8.1%}  "
        print(row)
    
    # Save results
    output = {
        "models": MODELS,
        "prompts": TEST_PROMPTS,
        "results": results,
        "overlap_matrix": overlap
    }
    
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/multi_model_transfer.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to experiments/multi_model_transfer.json")
    
    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Find highest and lowest overlap pairs
    pairs = []
    for m1 in models:
        for m2 in models:
            if m1 < m2:  # Avoid duplicates
                pairs.append((m1, m2, overlap[m1][m2]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nMost similar pairs:")
    for m1, m2, val in pairs[:3]:
        print(f"  {m1} ↔ {m2}: {val:.0%}")
    
    print("\nLeast similar pairs:")
    for m1, m2, val in pairs[-3:]:
        print(f"  {m1} ↔ {m2}: {val:.0%}")


if __name__ == "__main__":
    main()

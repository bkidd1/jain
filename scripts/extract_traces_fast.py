#!/usr/bin/env python3
"""
Week 2: Fast trace extraction (no full causal patching).

Uses logit lens only — faster but less rigorous.
We'll add causal verification on a smaller subset later.
"""

import json
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from tqdm import tqdm
import gc


def get_logit_lens_trace(model, prompt: str, layers_to_check: list = None):
    """Extract trace using logit lens (no causal patching)."""
    
    if layers_to_check is None:
        # Check 25%, 50%, 75%, final
        n = model.cfg.n_layers
        layers_to_check = [n//4, n//2, 3*n//4, n-1]
    
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        
        concepts = []
        seen = set()
        input_words = set(prompt.lower().split())
        
        for layer in layers_to_check:
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            normalized = model.ln_final(resid)
            logits = model.unembed(normalized)
            probs = torch.softmax(logits[0, -1], dim=-1)
            
            top = torch.topk(probs, 3)
            for prob, idx in zip(top.values, top.indices):
                token = model.tokenizer.decode([idx.item()]).strip()
                p = prob.item()
                
                # Skip if already seen, in input, or low confidence
                if token.lower() in seen or token.lower() in input_words or p < 0.15:
                    continue
                
                concepts.append({
                    "token": token,
                    "layer": layer,
                    "prob": round(p, 3)
                })
                seen.add(token.lower())
        
        # Get final output
        final_logits = model(prompt)
        final_probs = torch.softmax(final_logits[0, -1], dim=-1)
        final_token = model.tokenizer.decode([final_probs.argmax().item()]).strip()
        
        # Clear cache
        del cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Build trace string
    concepts.sort(key=lambda x: x["layer"])
    trace_tokens = [c["token"] for c in concepts if c["prob"] > 0.2]
    if final_token and final_token not in [c["token"] for c in concepts]:
        trace_tokens.append(final_token)
    trace_string = " → ".join(trace_tokens) if trace_tokens else final_token
    
    return {
        "final_output": final_token,
        "trace_string": trace_string,
        "concepts": concepts
    }


def load_prompts(data_dir: str = "data/raw") -> list:
    """Load all generated prompts."""
    prompts = []
    data_path = Path(data_dir)
    
    for file in data_path.glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                item = json.loads(line)
                prompts.append({
                    "task_type": item["task_type"],
                    "input_text": item["input_text"],
                    "expected_output": item.get("expected_output", ""),
                    "expected_trace": item.get("expected_trace", []),
                })
    
    return prompts


def main():
    print("="*60)
    print("Week 2: Fast trace extraction (logit lens only)")
    print("="*60)
    
    # Load prompts
    prompts = load_prompts()
    print(f"\nLoaded {len(prompts)} prompts")
    
    # Load model
    print(f"\nLoading Llama 3.1 8B...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device=device,
        dtype=torch.float16,
    )
    print(f"✓ Loaded on {device}")
    
    # Extract traces
    print(f"\nExtracting traces...")
    results = []
    
    for item in tqdm(prompts, desc="Extracting"):
        try:
            extracted = get_logit_lens_trace(model, item["input_text"])
            
            result = {
                "task_type": item["task_type"],
                "input_text": item["input_text"],
                "expected_output": item["expected_output"],
                "expected_trace": item["expected_trace"],
                "extracted": extracted
            }
            results.append(result)
            
        except Exception as e:
            print(f"\n  Error: {e}")
            continue
    
    # Save results
    output_path = Path("data/processed/traces_llama.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n✓ Saved {len(results)} traces to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    
    by_type = {}
    for r in results:
        t = r["task_type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    for task_type, count in sorted(by_type.items()):
        print(f"  {task_type}: {count}")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("Sample traces")
    print("="*60)
    
    for r in results[:5]:
        print(f"\n  Input: {r['input_text'][:50]}...")
        print(f"  Trace: {r['extracted']['trace_string']}")


if __name__ == "__main__":
    main()

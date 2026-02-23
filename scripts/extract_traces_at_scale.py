#!/usr/bin/env python3
"""
Week 2: Extract reasoning traces at scale.

Runs the causal extraction pipeline on all generated prompts
and saves the results for RTP training.
"""

import json
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.tuned_lens_extraction import ImprovedExtractor


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


def extract_all_traces(
    model_name: str = "meta-llama/Llama-3.1-8B",
    output_file: str = "data/processed/traces_llama.jsonl",
    max_prompts: int = None
):
    """Extract traces from all prompts."""
    
    print("="*60)
    print("Week 2: Large-scale trace extraction")
    print("="*60)
    
    # Load prompts
    prompts = load_prompts()
    print(f"\nLoaded {len(prompts)} prompts")
    
    if max_prompts:
        prompts = prompts[:max_prompts]
        print(f"  (limited to {max_prompts} for this run)")
    
    # Load model
    print(f"\nLoading {model_name}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16,
    )
    print(f"✓ Loaded on {device}")
    
    # Create extractor
    extractor = ImprovedExtractor(model, device)
    
    # Extract traces
    print(f"\nExtracting traces...")
    results = []
    
    for item in tqdm(prompts, desc="Extracting"):
        try:
            trace = extractor.extract_trace(item["input_text"])
            
            result = {
                "task_type": item["task_type"],
                "input_text": item["input_text"],
                "expected_output": item["expected_output"],
                "expected_trace": item["expected_trace"],
                "extracted": {
                    "final_output": trace.final_output,
                    "trace_string": trace.trace_string(),
                    "concepts": [
                        {
                            "token": c.token,
                            "layer": c.layer,
                            "prob": c.logit_lens_prob,
                            "causal_effect": c.causal_effect
                        }
                        for c in trace.concepts
                    ]
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"\n  Error on '{item['input_text'][:30]}...': {e}")
            continue
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n✓ Saved {len(results)} traces to {output_file}")
    
    # Summary stats
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    
    by_type = {}
    for r in results:
        t = r["task_type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    for task_type, count in sorted(by_type.items()):
        print(f"  {task_type}: {count}")
    
    # Count traces with causal concepts
    has_causal = sum(1 for r in results if any(
        c["causal_effect"] > 0.1 for c in r["extracted"]["concepts"]
    ))
    print(f"\n  Traces with causal concepts (>0.1 effect): {has_causal}/{len(results)}")
    
    return results


if __name__ == "__main__":
    # Run extraction
    extract_all_traces()

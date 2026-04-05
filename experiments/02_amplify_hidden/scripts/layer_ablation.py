#!/usr/bin/env python3
"""
Layer Ablation: Find which layer encodes hint-awareness most strongly.

For each layer:
1. Extract hint direction from that layer only
2. Amplify in that layer only (factor 1.2)
3. Measure effect on meta-awareness patterns

Goal: Find the "deception layer" where hint-awareness is concentrated.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


def extract_direction_from_layer(
    model, tokenizer, hint_prompts, no_hint_prompts, layer_idx, device
):
    """Extract hint direction from a single layer."""
    
    activations_hint = []
    activations_no_hint = []
    
    # Set up hook for this layer
    if hasattr(model, 'model'):
        layer = model.model.layers[layer_idx]
    else:
        layer = model.transformer.h[layer_idx]
    
    captured = {}
    
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured['act'] = output[0].detach()
        else:
            captured['act'] = output.detach()
    
    hook = layer.register_forward_hook(capture_hook)
    
    # Extract from hint prompts
    for prompt in hint_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        activations_hint.append(captured['act'].mean(dim=1))
    
    # Extract from no-hint prompts
    for prompt in no_hint_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        activations_no_hint.append(captured['act'].mean(dim=1))
    
    hook.remove()
    
    # Compute direction
    hint_mean = torch.cat(activations_hint, dim=0).mean(dim=0)
    no_hint_mean = torch.cat(activations_no_hint, dim=0).mean(dim=0)
    direction = hint_mean - no_hint_mean
    
    return direction


def generate_with_single_layer_amplification(
    model, tokenizer, prompt, direction, layer_idx, factor, device, max_tokens=256
):
    """Generate with amplification in a single layer."""
    
    if hasattr(model, 'model'):
        layer = model.model.layers[layer_idx]
    else:
        layer = model.transformer.h[layer_idx]
    
    def amplify_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        dir_normalized = direction.to(hidden.device, hidden.dtype)
        dir_normalized = dir_normalized / (dir_normalized.norm() + 1e-8)
        
        projection = (hidden * dir_normalized).sum(dim=-1, keepdim=True)
        amplified = hidden + (factor - 1) * projection * dir_normalized
        
        if rest is not None:
            return (amplified,) + rest
        return amplified
    
    hook = layer.register_forward_hook(amplify_hook)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    hook.remove()
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


def count_meta_patterns(text):
    """Count meta-awareness patterns in text."""
    text_lower = text.lower()
    patterns = {
        'correct_answer': len(re.findall(r'correct answer', text_lower)),
        'wrong_answer': len(re.findall(r'wrong answer', text_lower)),
        'explanation': len(re.findall(r'explanation:', text_lower)),
        'the_answer_is': len(re.findall(r'the answer is', text_lower)),
    }
    return patterns


def run_layer_ablation(
    model_name: str,
    data_path: Path,
    output_path: Path,
    factor: float = 1.2,
    n_test: int = 5,
):
    """Run layer ablation experiment."""
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Get number of layers
    if hasattr(model, 'model'):
        num_layers = len(model.model.layers)
    else:
        num_layers = len(model.transformer.h)
    
    print(f"Model has {num_layers} layers")
    
    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'hint'][:20]
    no_hint_prompts = [d['prompt'] for d in data if d.get('variant') == 'no_hint'][:20]
    test_prompts = hint_prompts[:n_test]
    
    print(f"Using {len(hint_prompts)} hint, {len(no_hint_prompts)} no-hint for direction")
    print(f"Testing on {len(test_prompts)} prompts per layer")
    
    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            # Extract direction from this layer
            direction = extract_direction_from_layer(
                model, tokenizer, hint_prompts, no_hint_prompts, layer_idx, device
            )
            direction_magnitude = direction.norm().item()
            
            # Test amplification in this layer
            layer_results = {
                'layer': layer_idx,
                'direction_magnitude': direction_magnitude,
                'responses': [],
                'meta_patterns': {
                    'correct_answer': 0,
                    'wrong_answer': 0,
                    'explanation': 0,
                    'the_answer_is': 0,
                }
            }
            
            for prompt in test_prompts:
                response = generate_with_single_layer_amplification(
                    model, tokenizer, prompt, direction, layer_idx, factor, device
                )
                
                patterns = count_meta_patterns(response)
                for k, v in patterns.items():
                    layer_results['meta_patterns'][k] += v
                
                layer_results['responses'].append({
                    'prompt': prompt[:100],
                    'response': response[:300],
                })
            
            results.append(layer_results)
            f.write(json.dumps(layer_results) + '\n')
            f.flush()
            
            print(f"Layer {layer_idx}: mag={direction_magnitude:.3f}, "
                  f"wrong_answer={layer_results['meta_patterns']['wrong_answer']}, "
                  f"correct_answer={layer_results['meta_patterns']['correct_answer']}")
    
    # Summary
    print("\n=== LAYER ABLATION SUMMARY ===")
    print(f"{'Layer':<6} {'Magnitude':<10} {'Wrong Ans':<10} {'Correct Ans':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['layer']:<6} {r['direction_magnitude']:<10.3f} "
              f"{r['meta_patterns']['wrong_answer']:<10} "
              f"{r['meta_patterns']['correct_answer']:<12}")
    
    # Find best layer
    best_layer = max(results, key=lambda x: x['meta_patterns']['wrong_answer'])
    print(f"\nBest layer for surfacing 'wrong answer': Layer {best_layer['layer']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, default=1.2)
    parser.add_argument("--n-test", type=int, default=5)
    args = parser.parse_args()
    
    run_layer_ablation(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        factor=args.factor,
        n_test=args.n_test,
    )

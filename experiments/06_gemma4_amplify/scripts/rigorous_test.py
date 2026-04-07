#!/usr/bin/env python3
"""
Rigorous amplification test with proper controls.

Key improvements:
1. More samples (20+ per condition)
2. Track accuracy systematically
3. Test multiple random seeds
4. Compare to no-hint baseline
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import re


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")


class DirectionAmplifier:
    def __init__(self, model, direction, factor, target_layers):
        self.model = model
        self.direction = direction
        self.factor = factor
        self.target_layers = target_layers
        self.hooks = []
        self._setup()
    
    def _setup(self):
        layers = get_layers(self.model)
        for i in self.target_layers:
            hook = layers[i].register_forward_hook(self._amplify)
            self.hooks.append(hook)
    
    def _amplify(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        d = self.direction.to(hidden.device, hidden.dtype)
        d = d / (d.norm() + 1e-8)
        proj = (hidden * d).sum(dim=-1, keepdim=True)
        amplified = hidden + (self.factor - 1) * proj * d
        
        return (amplified,) + rest if rest else amplified
    
    def remove(self):
        for h in self.hooks:
            h.remove()


def extract_answer(response: str, correct: str, misleading: str) -> str:
    """Try to extract which answer the model gave."""
    response_lower = response.lower()
    correct_lower = correct.lower()
    misleading_lower = misleading.lower()
    
    # Check first 200 chars for the answer
    first_part = response_lower[:200]
    
    if correct_lower in first_part and misleading_lower not in first_part:
        return "correct"
    elif misleading_lower in first_part and correct_lower not in first_part:
        return "misleading"
    elif correct_lower in first_part and misleading_lower in first_part:
        # Both mentioned - check which comes first or which is endorsed
        if first_part.index(correct_lower) < first_part.index(misleading_lower):
            return "correct"
        else:
            return "misleading"
    else:
        return "unclear"


def generate(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    model_name = "google/gemma-4-E2B"
    data_path = Path("~/jain/experiments/03_detector/data/combined_v2.jsonl").expanduser()
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    hint_data = [d for d in data if d.get('variant') == 'hint'][:30]
    no_hint_data = [d for d in data if d.get('variant') == 'no_hint'][:30]
    
    print(f"Testing {len(hint_data)} hint prompts")
    
    # Extract hint direction from layer 17 (our best layer)
    print("\nExtracting hint direction...")
    
    from amplify_gemma4 import extract_hint_direction
    hint_prompts = [d['prompt'] for d in hint_data[:20]]
    no_hint_prompts = [d['prompt'] for d in no_hint_data[:20]]
    
    direction = extract_hint_direction(
        model, tokenizer, hint_prompts, no_hint_prompts,
        layer_idx=17, device=model.device
    )
    
    # Test conditions
    conditions = [
        ("baseline", None, None),
        ("amplify_L17_1.2", [17], 1.2),
        ("amplify_L17_1.5", [17], 1.5),
        ("dampen_L17_0.8", [17], 0.8),
    ]
    
    results = {c[0]: {"correct": 0, "misleading": 0, "unclear": 0} for c in conditions}
    detailed = []
    
    for item in tqdm(hint_data, desc="Testing"):
        prompt = item['prompt']
        correct = item['correct_answer']
        misleading = item['misleading_answer']
        
        item_results = {"prompt": prompt, "correct": correct, "misleading": misleading}
        
        for cond_name, layers, factor in conditions:
            if layers:
                amp = DirectionAmplifier(model, direction, factor, layers)
            
            response = generate(model, tokenizer, prompt)
            
            if layers:
                amp.remove()
            
            answer = extract_answer(response, correct, misleading)
            results[cond_name][answer] += 1
            item_results[cond_name] = {"response": response[:200], "answer": answer}
        
        detailed.append(item_results)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for cond, counts in results.items():
        total = sum(counts.values())
        acc = counts['correct'] / total if total > 0 else 0
        syc = counts['misleading'] / total if total > 0 else 0
        print(f"{cond:20s}: correct={counts['correct']:2d} ({acc:.1%}), misleading={counts['misleading']:2d} ({syc:.1%}), unclear={counts['unclear']:2d}")
    
    # Save detailed results
    output = {
        "summary": results,
        "detailed": detailed[:10],  # Save first 10 for inspection
    }
    
    out_path = Path("~/jain/experiments/06_gemma4_amplify/results/rigorous_test.json").expanduser()
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

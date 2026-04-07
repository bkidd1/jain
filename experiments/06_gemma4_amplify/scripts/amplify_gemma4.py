#!/usr/bin/env python3
"""
Activation Amplification on Gemma 4

Can we surface hidden reasoning in Gemma 4 by amplifying the "hint direction"?

Key questions:
1. Does amplification reveal suppressed info in Gemma 4?
2. Which layers are most sensitive to amplification?
3. Is there a "sweet spot" factor for coherent but revealing outputs?
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


def get_layers(model):
    """Get the transformer layers for any architecture."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # Gemma 4 multimodal
        return model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Standard Llama-style
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style
        return model.transformer.h
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


class ActivationExtractor:
    """Extract activations from specific layers."""
    
    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = None
        self.hook = None
        self._setup_hook()
    
    def _setup_hook(self):
        layers = get_layers(self.model)
        self.hook = layers[self.layer_idx].register_forward_hook(self._capture_hook)
    
    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def get_activations(self) -> torch.Tensor:
        return self.activations
    
    def remove(self):
        if self.hook:
            self.hook.remove()


class DirectionAmplifier:
    """Amplify activations along a specific direction during generation."""
    
    def __init__(
        self, 
        model: nn.Module, 
        direction: torch.Tensor,
        amplify_factor: float,
        target_layers: list[int],
    ):
        self.model = model
        self.direction = direction
        self.amplify_factor = amplify_factor
        self.target_layers = target_layers
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        layers = get_layers(self.model)
        for i in self.target_layers:
            hook = layers[i].register_forward_hook(self._amplify_hook)
            self.hooks.append(hook)
    
    def _amplify_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        direction = self.direction.to(hidden.device, hidden.dtype)
        direction = direction / (direction.norm() + 1e-8)
        
        # Project onto direction and amplify
        projection = (hidden * direction).sum(dim=-1, keepdim=True)
        amplified = hidden + (self.amplify_factor - 1) * projection * direction
        
        if rest is not None:
            return (amplified,) + rest
        return amplified
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_hint_direction(
    model: nn.Module,
    tokenizer,
    hint_prompts: list[str],
    no_hint_prompts: list[str],
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract the hint direction = mean(hint) - mean(no_hint)."""
    print(f"Extracting hint direction from layer {layer_idx}...")
    
    extractor = ActivationExtractor(model, layer_idx)
    
    hint_acts = []
    no_hint_acts = []
    
    for prompt in tqdm(hint_prompts, desc="Hint prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        act = extractor.get_activations().mean(dim=1)
        hint_acts.append(act)
    
    for prompt in tqdm(no_hint_prompts, desc="No-hint prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        act = extractor.get_activations().mean(dim=1)
        no_hint_acts.append(act)
    
    extractor.remove()
    
    hint_mean = torch.cat(hint_acts, dim=0).mean(dim=0)
    no_hint_mean = torch.cat(no_hint_acts, dim=0).mean(dim=0)
    
    direction = hint_mean - no_hint_mean
    print(f"Hint direction magnitude: {direction.norm():.4f}")
    
    return direction


def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


def run_layer_sweep(
    model, tokenizer, hint_direction_by_layer, 
    test_prompts, layers_to_test, amplify_factor,
    device, max_new_tokens=150
):
    """Test amplification at different layers."""
    results = []
    
    for layer_idx in tqdm(layers_to_test, desc="Layer sweep"):
        direction = hint_direction_by_layer[layer_idx]
        
        # Amplify only at this layer
        amplifier = DirectionAmplifier(
            model, direction, amplify_factor, [layer_idx]
        )
        
        layer_results = []
        for prompt in test_prompts[:3]:  # Test on a few examples
            response = generate_text(model, tokenizer, prompt, max_new_tokens, device)
            layer_results.append(response)
        
        amplifier.remove()
        
        results.append({
            'layer': layer_idx,
            'responses': layer_results,
        })
        
        # Print sample
        print(f"\n--- Layer {layer_idx} (factor={amplify_factor}) ---")
        print(layer_results[0][:300] + "...")
    
    return results


def run_factor_sweep(
    model, tokenizer, hint_direction, target_layers,
    test_prompts, factors, device, max_new_tokens=150
):
    """Test different amplification factors."""
    results = []
    
    for factor in tqdm(factors, desc="Factor sweep"):
        amplifier = DirectionAmplifier(
            model, hint_direction, factor, target_layers
        )
        
        factor_results = []
        for prompt in test_prompts[:3]:
            response = generate_text(model, tokenizer, prompt, max_new_tokens, device)
            factor_results.append(response)
        
        amplifier.remove()
        
        results.append({
            'factor': factor,
            'responses': factor_results,
        })
        
        print(f"\n--- Factor {factor} ---")
        print(factor_results[0][:300] + "...")
    
    return results


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Get layer count
    layers = get_layers(model)
    num_layers = len(layers)
    print(f"Model has {num_layers} layers")
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    
    hint_data = [d for d in data if d.get('variant') == 'hint']
    no_hint_data = [d for d in data if d.get('variant') == 'no_hint']
    
    hint_prompts = [d['prompt'] for d in hint_data[:20]]
    no_hint_prompts = [d['prompt'] for d in no_hint_data[:20]]
    
    print(f"Using {len(hint_prompts)} hint, {len(no_hint_prompts)} no-hint prompts")
    
    # Extract hint directions at multiple layers
    test_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-2]
    print(f"\nExtracting directions from layers: {test_layers}")
    
    hint_directions = {}
    for layer_idx in test_layers:
        direction = extract_hint_direction(
            model, tokenizer, hint_prompts, no_hint_prompts,
            layer_idx, device
        )
        hint_directions[layer_idx] = direction
    
    # Test prompts for generation
    test_prompts = hint_prompts[:5]
    
    # First: baseline (no amplification)
    print("\n" + "="*60)
    print("BASELINE (no amplification)")
    print("="*60)
    
    baseline_results = []
    for prompt in test_prompts[:3]:
        response = generate_text(model, tokenizer, prompt, 150, device)
        baseline_results.append(response)
        print(f"\n{response[:400]}...")
    
    # Layer sweep: which layer is most sensitive?
    print("\n" + "="*60)
    print(f"LAYER SWEEP (factor={args.amplify})")
    print("="*60)
    
    layer_results = run_layer_sweep(
        model, tokenizer, hint_directions,
        test_prompts, test_layers, args.amplify,
        device
    )
    
    # Factor sweep at middle layer
    middle_layer = num_layers // 2
    print("\n" + "="*60)
    print(f"FACTOR SWEEP (layer={middle_layer})")
    print("="*60)
    
    factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    factor_results = run_factor_sweep(
        model, tokenizer, hint_directions[middle_layer],
        [middle_layer], test_prompts, factors, device
    )
    
    # Save results
    all_results = {
        'model': args.model,
        'num_layers': num_layers,
        'baseline': baseline_results,
        'layer_sweep': layer_results,
        'factor_sweep': factor_results,
        'test_layers': test_layers,
    }
    
    with open(output_dir / 'amplification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved results to: {output_dir / 'amplification_results.json'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--data", type=Path, 
                       default=Path("~/jain/experiments/03_detector/data/combined_v2.jsonl").expanduser())
    parser.add_argument("--output", type=Path,
                       default=Path("~/jain/experiments/06_gemma4_amplify/results").expanduser())
    parser.add_argument("--amplify", type=float, default=1.2)
    args = parser.parse_args()
    
    main(args)
